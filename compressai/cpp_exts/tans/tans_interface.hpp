/*
 * tANS (table-based Asymmetric Numeral Systems) — header.
 *
 * Provides cache-friendly, lookup-table-driven entropy coding as a
 * drop-in replacement for rANS in CompressAI.
 *
 * Key structures:
 *   CTableEntry  — encoding table entry  (new_state, nbBits, bits_to_output)
 *   DTableEntry  — decoding table entry  (symbol, nbBits, new_state_base)
 *
 * The tables are built from quantized CDFs (same format CompressAI uses)
 * and indexed by (symbol, state) for encoding or (state) for decoding.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

/* ── Table entries ─────────────────────────────────────────────────────── */

struct CTableEntry {
  uint32_t new_state;  // state after encoding
  uint8_t nb_bits;     // number of bits to output before transition
};

struct DTableEntry {
  uint16_t symbol;     // decoded symbol
  uint8_t nb_bits;     // bits to read
  uint32_t new_state;  // base of next state (add read bits)
};

/* ── Per-CDF tANS tables ──────────────────────────────────────────────── */

struct TansTable {
  int R;                      // log2(table_size)
  int L;                      // table size = 1 << R
  int num_symbols;            // alphabet size (max_value + 1, including escape)
  int offset;                 // CDF offset for this channel

  // CTable[symbol * L + (state - L)] → CTableEntry
  std::vector<CTableEntry> ctable;

  // DTable[state - L] → DTableEntry
  std::vector<DTableEntry> dtable;
};

/* ── Table builder ─────────────────────────────────────────────────────── */

TansTable build_tans_table(const std::vector<int32_t> &cdf,
                           int cdf_size, int offset, int R);

/* ── Encoder ───────────────────────────────────────────────────────────── */

class TansEncoder {
public:
  TansEncoder() = default;

  TansEncoder(const TansEncoder &) = delete;
  TansEncoder &operator=(const TansEncoder &) = delete;

  py::bytes encode_with_indexes(
      const std::vector<int32_t> &symbols,
      const std::vector<int32_t> &indexes,
      const std::vector<std::vector<int32_t>> &cdfs,
      const std::vector<int32_t> &cdfs_sizes,
      const std::vector<int32_t> &offsets,
      int R = 10);

private:
  // Cache of built tables, keyed by cdf index
  std::vector<TansTable> _table_cache;
  int _cached_R = -1;

  void ensure_tables(const std::vector<std::vector<int32_t>> &cdfs,
                     const std::vector<int32_t> &cdfs_sizes,
                     const std::vector<int32_t> &offsets,
                     int R);
};

/* ── Decoder ───────────────────────────────────────────────────────────── */

class TansDecoder {
public:
  TansDecoder() = default;

  TansDecoder(const TansDecoder &) = delete;
  TansDecoder &operator=(const TansDecoder &) = delete;

  std::vector<int32_t> decode_with_indexes(
      const std::string &encoded,
      const std::vector<int32_t> &indexes,
      const std::vector<std::vector<int32_t>> &cdfs,
      const std::vector<int32_t> &cdfs_sizes,
      const std::vector<int32_t> &offsets,
      int R = 10);

private:
  std::vector<TansTable> _table_cache;
  int _cached_R = -1;

  void ensure_tables(const std::vector<std::vector<int32_t>> &cdfs,
                     const std::vector<int32_t> &cdfs_sizes,
                     const std::vector<int32_t> &offsets,
                     int R);
};
