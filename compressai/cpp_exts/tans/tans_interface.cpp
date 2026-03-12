/*
 * tANS (table-based Asymmetric Numeral Systems) — implementation.
 *
 * Implements the four algorithms from the paper:
 *   1. Symbol spread          — map symbols to states
 *   2. Encoding preparation   — compute per-symbol encoding params
 *   3. Build encoding table   — CTable[symbol][state]
 *   4. Build decoding table   — DTable[state]
 *
 * Plus encoder/decoder with bypass coding for out-of-range symbols.
 */

#include "tans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

/* ── Constants ─────────────────────────────────────────────────────────── */

static constexpr uint16_t bypass_precision = 4;
static constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

/* CDF precision used by CompressAI */
static constexpr int CDF_PRECISION = 16;
static constexpr int CDF_TOTAL = 1 << CDF_PRECISION;  // 65536

/* ── Algorithm 1: Symbol Spread ────────────────────────────────────────── */

/*
 * Spread symbols across L states such that each symbol s appears exactly
 * freq[s] times.  Uses a step-based spread for good distribution.
 */
static std::vector<uint16_t>
symbol_spread(const std::vector<int> &freqs, int num_symbols, int L) {
  std::vector<uint16_t> spread(L);

  // Step chosen to be coprime to L for good distribution.
  // Use (L >> 1) + (L >> 3) + 3, which is odd when L is a power of 2.
  const int step = (L >> 1) + (L >> 3) + 3;
  const int mask = L - 1; // L must be power of 2

  int pos = 0;
  for (int s = 0; s < num_symbols; ++s) {
    for (int i = 0; i < freqs[s]; ++i) {
      spread[pos & mask] = static_cast<uint16_t>(s);
      pos += step;
    }
  }

  return spread;
}

/* ── Normalize CDF to tANS table size ──────────────────────────────────── */

/*
 * Convert a CompressAI CDF (precision=16, total=65536) to symbol frequencies
 * that sum to exactly L = 1<<R.  Every symbol with non-zero probability gets
 * freq >= 1.
 */
static std::vector<int>
cdf_to_frequencies(const std::vector<int32_t> &cdf, int cdf_size, int L) {
  const int num_symbols = cdf_size - 1; // CDF has num_symbols+1 entries
  std::vector<int> freqs(num_symbols);

  // First pass: proportional scaling + floor, ensure min of 1
  int total = 0;
  for (int s = 0; s < num_symbols; ++s) {
    int prob = cdf[s + 1] - cdf[s];
    if (prob <= 0) {
      freqs[s] = 0;
      continue;
    }
    int f = static_cast<int>(
        static_cast<int64_t>(prob) * L / CDF_TOTAL);
    if (f < 1)
      f = 1;
    freqs[s] = f;
    total += f;
  }

  // Adjust to make sum exactly L.
  // When total > L, iteratively reduce the largest frequencies (keeping
  // each >= 1).  When total < L, add the deficit to the largest.
  while (total > L) {
    // Find the symbol with the largest frequency > 1
    int max_idx = -1;
    int max_freq = 1;
    for (int s = 0; s < num_symbols; ++s) {
      if (freqs[s] > max_freq) {
        max_freq = freqs[s];
        max_idx = s;
      }
    }
    if (max_idx == -1) {
      throw std::runtime_error(
          "tANS: more non-zero-probability symbols than table states — "
          "try a larger R");
    }
    int reduce = std::min(total - L, freqs[max_idx] - 1);
    freqs[max_idx] -= reduce;
    total -= reduce;
  }

  if (total < L) {
    int max_idx = 0;
    for (int s = 1; s < num_symbols; ++s) {
      if (freqs[s] > freqs[max_idx])
        max_idx = s;
    }
    freqs[max_idx] += (L - total);
  }

  return freqs;
}

/* ── Build Tables (Algorithms 2–4) ─────────────────────────────────────── */

TansTable build_tans_table(const std::vector<int32_t> &cdf,
                           int cdf_size, int offset, int R) {
  const int L = 1 << R;
  const int num_symbols = cdf_size - 1; // number of codable symbols

  TansTable table;
  table.R = R;
  table.L = L;
  table.num_symbols = num_symbols;
  table.offset = offset;

  // --- Step 1: Normalize frequencies ---
  std::vector<int> freqs = cdf_to_frequencies(cdf, cdf_size, L);

  // --- Step 2: Symbol spread ---
  std::vector<uint16_t> spread = symbol_spread(freqs, num_symbols, L);

  // --- Step 3: Compute cumulative starts for each symbol ---
  // cum_start[s] = sum(freqs[0..s-1])
  std::vector<int> cum_start(num_symbols + 1, 0);
  for (int s = 0; s < num_symbols; ++s) {
    cum_start[s + 1] = cum_start[s] + freqs[s];
  }
  assert(cum_start[num_symbols] == L);

  // --- Step 4: Build encoding table (Algorithm 3) ---
  //
  // For each symbol s and each occurrence of s in the spread table,
  // we compute the encoding table entry.
  //
  // Encoding: given current state x (in [L, 2L)) and symbol s,
  //   1. Compute how many bits to output:  nb_bits
  //   2. Output the low nb_bits of x
  //   3. x >>= nb_bits
  //   4. new_x = encoding_table_lookup
  //
  // CTable layout: ctable[s * L + (x - L)] for x in [L..2L)

  table.ctable.resize(static_cast<size_t>(num_symbols) * L);

  // For the encoding table, we need to know, for each symbol s,
  // which states map to which positions. We use the spread + next-state
  // approach from the paper.

  // Build "next state" table from the spread:
  // For each state position p (0..L-1), the symbol at spread[p] is decoded.
  // We assign states L..2L-1 to these positions.
  // Symbol s gets assigned states L + cum_start[s] .. L + cum_start[s] + freqs[s] - 1
  // via the sorted order.

  // symbol_state_counter[s] tracks the next available state index for symbol s
  std::vector<int> symbol_state_counter(num_symbols, 0);

  // Build the state assignment from spread
  // For each position in the spread table, we assign states to symbols
  // in the order they appear, matching the decoding table.
  // state_for_position[p] = L + position in the sorted per-symbol assignment
  std::vector<int> state_for_pos(L);
  {
    std::vector<int> sym_count(num_symbols, 0);
    for (int p = 0; p < L; ++p) {
      int s = spread[p];
      state_for_pos[p] = L + cum_start[s] + sym_count[s];
      sym_count[s]++;
    }
  }

  // Now build encoding table from the relationship:
  // Encoding symbol s from state x:
  //   fs = freqs[s]
  //   nb_bits = R - floor(log2(fs))     ... but state-dependent
  //
  // More precisely, for tANS:
  //   The valid input states for encoding symbol s are in [fs, 2*fs)
  //   after normalization. So:
  //     nb_bits = (R + 1) when x >= some threshold, R otherwise
  //
  // Standard tANS encoding:
  //   Given state x in [L, 2L) and symbol s with frequency fs:
  //     k = R - floor(log2(fs))         [number of bits per symbol, approximate]
  //     threshold = fs << (k + 1)       [if x >= threshold, output k+1 bits, else k bits]
  //     if x >= threshold:
  //       output (k+1) low bits of x
  //       x >>= (k+1)
  //     else:
  //       output k low bits of x
  //       x >>= k
  //     now x is in [fs, 2*fs), look up next state from symbol_spread

  // Build per-symbol encoding table using the next_state_table approach:
  // For each symbol, we know which spread positions have that symbol,
  // and what state each position maps to.
  // We iterate states x in [L, 2L) for each symbol.

  for (int s = 0; s < num_symbols; ++s) {
    int fs = freqs[s];
    if (fs == 0) {
      // Zero-frequency symbol: fill with dummy entries
      for (int x = 0; x < L; ++x) {
        table.ctable[s * L + x] = {0, 0};
      }
      continue;
    }

    // Collect the states assigned to symbol s from the spread,
    // sorted by state for deterministic encoding.
    std::vector<int> sym_states;
    sym_states.reserve(fs);
    for (int p = 0; p < L; ++p) {
      if (spread[p] == s) {
        sym_states.push_back(state_for_pos[p]);
      }
    }
    std::sort(sym_states.begin(), sym_states.end());
    assert(static_cast<int>(sym_states.size()) == fs);

    // For each input state x in [L, 2L):
    // Determine nb_bits to output, then the reduced state maps to
    // one of sym_states.
    int k = 0;
    while ((fs << (k + 1)) <= (L << 1)) {
      k++;
    }
    // k is such that  fs << k <= L  and  fs << (k+1) > L
    // Actually we want: the range of nb_bits
    // For state x in [L, 2L):
    //   reduced = x >> nb_bits must be in [fs, 2*fs)
    //   nb_bits = R - floor(log2(fs)) or R - floor(log2(fs)) + 1

    int nb_bits_base = R; // will refine below
    {
      int tmp = fs;
      int log2_fs = 0;
      while (tmp > 1) {
        tmp >>= 1;
        log2_fs++;
      }
      nb_bits_base = R - log2_fs;
    }
    // threshold: states >= threshold need (nb_bits_base) bits,
    //            states < threshold need (nb_bits_base - 1) bits
    // Wait — let me use the standard formulation:
    // For x in [L, 2L):
    //   We want x_reduced in [fs, 2*fs)
    //   x_reduced = x >> nb_bits
    //   nb_bits such that x >> nb_bits is in [fs, 2*fs)
    //
    // So nb_bits = floor(log2(x)) - floor(log2(fs))
    //           = number of bits to shift x down to [fs, 2*fs) range
    //
    // But since x in [L, 2L) = [2^R, 2^(R+1)):
    //   If x < 2*fs << nb_bits_base: nb_bits = nb_bits_base - 1 + (x >= L ? 0 : ...)
    //   It simplifies to:

    // Let's use a direct approach:
    // For fs = frequency, the symbol's valid reduced states are [fs, 2*fs)
    // There are fs such states.
    // In the encoding table, we map each (x in [L...2L)) to the right
    // sym_states entry.

    int sym_idx = 0; // index into sym_states
    for (int x = L; x < 2 * L; ++x) {
      // Determine nb_bits: we want x >> nb_bits in [fs, 2*fs)
      int x_reduced = x;
      int nb = 0;
      while (x_reduced >= 2 * fs) {
        x_reduced >>= 1;
        nb++;
      }
      // Now x_reduced is in [fs, 2*fs)
      // The index into sym_states is (x_reduced - fs)
      int idx = x_reduced - fs;
      assert(idx >= 0 && idx < fs);

      CTableEntry entry;
      entry.nb_bits = static_cast<uint8_t>(nb);
      entry.new_state = static_cast<uint32_t>(sym_states[idx]);
      table.ctable[s * L + (x - L)] = entry;
    }
  }

  // --- Step 5: Build decoding table (Algorithm 4) ---
  //
  // DTable[state - L] for state in [L, 2L)
  // Each entry: (symbol, nb_bits, new_state_base)
  //
  // Decoding from state x:
  //   entry = DTable[x - L]
  //   symbol = entry.symbol
  //   nb_bits = entry.nb_bits
  //   new_state_base = entry.new_state
  //   x = new_state_base + read_bits(nb_bits)

  table.dtable.resize(L);

  // The decoding table is built from the encoding table's inverse.
  // For each state s_new that an encoding step produces, we know
  // which symbol was encoded and how many bits were output.
  //
  // From the spread + state assignment:
  // state_for_pos[p] is the state corresponding to spread position p.
  // If we're in state x = state_for_pos[p], the decoded symbol is spread[p].
  //
  // To find nb_bits and new_state_base:
  // After decoding symbol s from state x, we need to reconstruct the
  // previous encoder state. The encoder output nb_bits low bits of the
  // previous state, then shifted right. So the decoder reads nb_bits bits
  // and shifts left:
  //   new_state_base such that:  new_state = new_state_base | read_bits(nb_bits)
  //   where new_state is in [L, 2L)
  //
  // Since the encoder produced state x from reduced state x_reduced:
  //   x_reduced is in [fs, 2*fs) for symbol s with frequency fs
  // And the encoding outputs nb_bits = R - floor_log2(x_reduced) or similar
  //
  // Simpler: use the spread to build the decode table directly.

  {
    std::vector<int> sym_count(num_symbols, 0);
    for (int p = 0; p < L; ++p) {
      int s = spread[p];
      int fs = freqs[s];
      int x = state_for_pos[p]; // This state decodes to symbol s

      // x is in [L, 2L), so index is x - L
      int idx = x - L;

      // Determine nb_bits for decoding:
      // The encoder compressed from some x_prev to x.
      // x_reduced was in [fs, 2*fs), specifically x_reduced = fs + sym_count[s]
      int x_reduced = fs + sym_count[s];

      // nb_bits to read = R - floor(log2(x_reduced))
      int log2_xr = 0;
      {
        int tmp = x_reduced;
        while (tmp >= 2) {
          tmp >>= 1;
          log2_xr++;
        }
      }
      int nb = R - log2_xr;

      // new_state_base: after reading nb bits (value b), new state = x_reduced << nb | b
      // So new_state_base = x_reduced << nb
      uint32_t new_state_base = static_cast<uint32_t>(x_reduced) << nb;

      DTableEntry entry;
      entry.symbol = static_cast<uint16_t>(s);
      entry.nb_bits = static_cast<uint8_t>(nb);
      entry.new_state = new_state_base;

      table.dtable[idx] = entry;

      sym_count[s]++;
    }
  }

  return table;
}

/* ── Table cache management ────────────────────────────────────────────── */

void TansEncoder::ensure_tables(
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets, int R) {

  if (_cached_R == R && _table_cache.size() == cdfs.size()) {
    return; // tables already built for this configuration
  }

  _table_cache.clear();
  _table_cache.reserve(cdfs.size());
  for (size_t i = 0; i < cdfs.size(); ++i) {
    _table_cache.push_back(
        build_tans_table(cdfs[i], cdfs_sizes[i], offsets[i], R));
  }
  _cached_R = R;
}

void TansDecoder::ensure_tables(
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets, int R) {

  if (_cached_R == R && _table_cache.size() == cdfs.size()) {
    return;
  }

  _table_cache.clear();
  _table_cache.reserve(cdfs.size());
  for (size_t i = 0; i < cdfs.size(); ++i) {
    _table_cache.push_back(
        build_tans_table(cdfs[i], cdfs_sizes[i], offsets[i], R));
  }
  _cached_R = R;
}

/* ── Encoder ───────────────────────────────────────────────────────────── */

/*
 * Encoding algorithm:
 *   - Process symbols in FORWARD order, but we write bits to a buffer
 *     and reverse at the end (mirroring rANS convention).
 *   - State x starts at L.
 *   - For each symbol:
 *       1. If out of range: encode escape + bypass value
 *       2. Look up CTable[symbol][x - L]
 *       3. Output nb_bits low bits of x
 *       4. x = new_state from table
 *   - Final state is written to output.
 */

py::bytes TansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols,
    const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets,
    int R) {

  assert(symbols.size() == indexes.size());

  // Auto-raise R if any CDF has more non-zero-probability symbols than 1<<R.
  // Each non-zero symbol needs at least one table state.
  {
    int max_non_zero = 0;
    for (size_t i = 0; i < cdfs.size(); ++i) {
      int ns = cdfs_sizes[i] - 1;
      int nz = 0;
      for (int s = 0; s < ns; ++s) {
        if (s + 1 < static_cast<int>(cdfs[i].size()) &&
            cdfs[i][s + 1] - cdfs[i][s] > 0)
          ++nz;
      }
      max_non_zero = std::max(max_non_zero, nz);
    }
    while ((1 << R) < max_non_zero)
      ++R;
  }

  ensure_tables(cdfs, cdfs_sizes, offsets, R);

  const int L = 1 << R;

  // Bit buffer: we collect bits as (value, nbits) pairs
  struct BitChunk {
    uint32_t bits;
    uint8_t nbits;
  };
  std::vector<BitChunk> bit_buffer;
  bit_buffer.reserve(symbols.size() * 2);

  // Bypass buffer: groups of raw-value chunks for out-of-range symbols.
  // Each group is one bypass value's length + data chunks.
  // Since we encode in reverse, groups are collected in reverse order
  // and must be reversed before serialization so the decoder reads
  // them in forward symbol order.
  std::vector<std::vector<BitChunk>> bypass_groups;

  uint32_t state = static_cast<uint32_t>(L); // initial state

  // Encode in REVERSE order (like rANS) to enable forward decoding
  for (int i = static_cast<int>(symbols.size()) - 1; i >= 0; --i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0 && cdf_idx < static_cast<int>(cdfs.size()));

    const TansTable &tbl = _table_cache[cdf_idx];
    const int max_value = tbl.num_symbols - 1;

    int32_t value = symbols[i] - tbl.offset;

    uint32_t raw_val = 0;
    bool is_bypass = false;

    if (value < 0) {
      raw_val = static_cast<uint32_t>(-2 * value - 1);
      value = max_value;
      is_bypass = true;
    } else if (value >= max_value) {
      raw_val = static_cast<uint32_t>(2 * (value - max_value));
      value = max_value;
      is_bypass = true;
    }

    assert(value >= 0 && value < tbl.num_symbols);

    // Look up encoding table
    assert(state >= static_cast<uint32_t>(L) &&
           state < static_cast<uint32_t>(2 * L));
    const CTableEntry &entry = tbl.ctable[value * L + (state - L)];

    // Output low bits of current state
    if (entry.nb_bits > 0) {
      bit_buffer.push_back(
          {state & ((1u << entry.nb_bits) - 1), entry.nb_bits});
    }

    state = entry.new_state;

    // Handle bypass coding
    if (is_bypass) {
      std::vector<BitChunk> group;
      // Encode the raw value in bypass_precision chunks
      int32_t n_bypass = 0;
      {
        uint32_t tmp = raw_val;
        while (tmp != 0) {
          tmp >>= bypass_precision;
          ++n_bypass;
        }
      }

      // Encode number of bypasses (unary-ish coding)
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        group.push_back({max_bypass_val, bypass_precision});
        val -= max_bypass_val;
      }
      group.push_back(
          {static_cast<uint32_t>(val), bypass_precision});

      // Encode raw value chunks
      for (int32_t j = 0; j < n_bypass; ++j) {
        uint32_t chunk =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        group.push_back({chunk, bypass_precision});
      }

      bypass_groups.push_back(std::move(group));
    }
  }

  // Flatten bypass groups in forward-symbol order (reverse of encoding order)
  std::vector<BitChunk> bypass_buffer;
  {
    std::reverse(bypass_groups.begin(), bypass_groups.end());
    for (auto &group : bypass_groups) {
      for (auto &chunk : group) {
        bypass_buffer.push_back(chunk);
      }
    }
  }

  // --- Pack into byte string ---
  // Format: [R (1 byte)] [final_state (4 bytes)] [bypass_len (4 bytes)]
  //         [bypass bits...] [main bits (reversed)...]

  // Collect all bits into a contiguous bit stream
  std::vector<uint8_t> output;
  output.reserve(symbols.size()); // rough estimate

  // Header: R value
  output.push_back(static_cast<uint8_t>(R));

  // Final state (little-endian 4 bytes)
  output.push_back(static_cast<uint8_t>(state & 0xFF));
  output.push_back(static_cast<uint8_t>((state >> 8) & 0xFF));
  output.push_back(static_cast<uint8_t>((state >> 16) & 0xFF));
  output.push_back(static_cast<uint8_t>((state >> 24) & 0xFF));

  // Number of bypass chunks (little-endian 4 bytes)
  uint32_t n_bypass_chunks = static_cast<uint32_t>(bypass_buffer.size());
  output.push_back(static_cast<uint8_t>(n_bypass_chunks & 0xFF));
  output.push_back(static_cast<uint8_t>((n_bypass_chunks >> 8) & 0xFF));
  output.push_back(static_cast<uint8_t>((n_bypass_chunks >> 16) & 0xFF));
  output.push_back(static_cast<uint8_t>((n_bypass_chunks >> 24) & 0xFF));

  // Pack bypass bits
  uint64_t bit_accumulator = 0;
  int bits_in_acc = 0;

  auto flush_bits = [&]() {
    while (bits_in_acc >= 8) {
      output.push_back(static_cast<uint8_t>(bit_accumulator & 0xFF));
      bit_accumulator >>= 8;
      bits_in_acc -= 8;
    }
  };

  for (auto &chunk : bypass_buffer) {
    bit_accumulator |= static_cast<uint64_t>(chunk.bits) << bits_in_acc;
    bits_in_acc += chunk.nbits;
    flush_bits();
  }

  // Pack main bits (in reverse order — bit_buffer was written during
  // reverse-order encoding, so reading it back-to-front gives forward order)
  for (int i = static_cast<int>(bit_buffer.size()) - 1; i >= 0; --i) {
    bit_accumulator |=
        static_cast<uint64_t>(bit_buffer[i].bits) << bits_in_acc;
    bits_in_acc += bit_buffer[i].nbits;
    flush_bits();
  }

  // Flush remaining bits
  if (bits_in_acc > 0) {
    output.push_back(static_cast<uint8_t>(bit_accumulator & 0xFF));
    bit_accumulator >>= 8;
    bits_in_acc -= 8;
    if (bits_in_acc > 0) {
      output.push_back(static_cast<uint8_t>(bit_accumulator & 0xFF));
    }
  }

  return py::bytes(reinterpret_cast<const char *>(output.data()),
                   output.size());
}

/* ── Decoder ───────────────────────────────────────────────────────────── */

/*
 * Decoding algorithm:
 *   - Read header: R, final_state, bypass_len
 *   - Read bypass chunks
 *   - Decode symbols in FORWARD order:
 *       1. DTable[state - L] → (symbol, nb_bits, new_state_base)
 *       2. Read nb_bits from bitstream
 *       3. state = new_state_base | read_bits
 *       4. If symbol == max_value: read bypass value
 */

std::vector<int32_t> TansDecoder::decode_with_indexes(
    const std::string &encoded,
    const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets,
    int R_hint) {

  assert(!encoded.empty());

  const uint8_t *data = reinterpret_cast<const uint8_t *>(encoded.data());
  size_t pos = 0;

  // Read header
  int R = data[pos++];
  (void)R_hint; // R is stored in the stream

  ensure_tables(cdfs, cdfs_sizes, offsets, R);

  const int L = 1 << R;

  // Read final state (little-endian)
  uint32_t state = 0;
  state |= static_cast<uint32_t>(data[pos++]);
  state |= static_cast<uint32_t>(data[pos++]) << 8;
  state |= static_cast<uint32_t>(data[pos++]) << 16;
  state |= static_cast<uint32_t>(data[pos++]) << 24;

  // Read bypass chunk count
  uint32_t n_bypass_chunks = 0;
  n_bypass_chunks |= static_cast<uint32_t>(data[pos++]);
  n_bypass_chunks |= static_cast<uint32_t>(data[pos++]) << 8;
  n_bypass_chunks |= static_cast<uint32_t>(data[pos++]) << 16;
  n_bypass_chunks |= static_cast<uint32_t>(data[pos++]) << 24;

  // Set up bit reader for the rest of the data
  const uint8_t *bit_data = data + pos;
  size_t bit_data_len = encoded.size() - pos;

  // Bit reader state
  uint64_t bit_acc = 0;
  int bits_avail = 0;
  size_t byte_pos = 0;

  auto refill = [&]() {
    while (bits_avail <= 56 && byte_pos < bit_data_len) {
      bit_acc |= static_cast<uint64_t>(bit_data[byte_pos++]) << bits_avail;
      bits_avail += 8;
    }
  };

  auto read_bits = [&](int n) -> uint32_t {
    refill();
    uint32_t val = static_cast<uint32_t>(bit_acc) & ((1u << n) - 1);
    bit_acc >>= n;
    bits_avail -= n;
    return val;
  };

  // Read bypass chunks
  std::vector<uint32_t> bypass_values;
  bypass_values.reserve(n_bypass_chunks);
  for (uint32_t i = 0; i < n_bypass_chunks; ++i) {
    bypass_values.push_back(read_bits(bypass_precision));
  }

  // Decode symbols in forward order
  std::vector<int32_t> output(indexes.size());
  int bypass_idx = 0;

  for (size_t i = 0; i < indexes.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0 && cdf_idx < static_cast<int>(cdfs.size()));

    const TansTable &tbl = _table_cache[cdf_idx];
    const int max_value = tbl.num_symbols - 1;

    assert(state >= static_cast<uint32_t>(L) &&
           state < static_cast<uint32_t>(2 * L));

    const DTableEntry &entry = tbl.dtable[state - L];

    int32_t value = static_cast<int32_t>(entry.symbol);

    // Read bits and advance state
    uint32_t new_state = entry.new_state;
    if (entry.nb_bits > 0) {
      uint32_t bits = read_bits(entry.nb_bits);
      new_state |= bits;
    }
    state = new_state;

    // Handle bypass
    if (value == max_value) {
      // Read number of bypass chunks
      assert(bypass_idx < static_cast<int>(bypass_values.size()));
      int32_t val = bypass_values[bypass_idx++];
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        assert(bypass_idx < static_cast<int>(bypass_values.size()));
        val = bypass_values[bypass_idx++];
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        assert(bypass_idx < static_cast<int>(bypass_values.size()));
        val = bypass_values[bypass_idx++];
        raw_val |= val << (j * bypass_precision);
      }

      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + tbl.offset;
  }

  return output;
}

/* ── Pybind11 module ───────────────────────────────────────────────────── */

PYBIND11_MODULE(tans, m) {
  m.attr("__name__") = "compressai.tans";
  m.doc() = "table-based Asymmetric Numeral System (tANS) entropy coder";

  py::class_<TansEncoder>(m, "TansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes", &TansEncoder::encode_with_indexes,
           py::arg("symbols"), py::arg("indexes"), py::arg("cdfs"),
           py::arg("cdfs_sizes"), py::arg("offsets"), py::arg("R") = 10,
           "Encode symbols to a byte string using tANS lookup tables");

  py::class_<TansDecoder>(m, "TansDecoder")
      .def(py::init<>())
      .def("decode_with_indexes", &TansDecoder::decode_with_indexes,
           py::arg("encoded"), py::arg("indexes"), py::arg("cdfs"),
           py::arg("cdfs_sizes"), py::arg("offsets"), py::arg("R") = 10,
           "Decode a byte string to symbols using tANS lookup tables");
}
