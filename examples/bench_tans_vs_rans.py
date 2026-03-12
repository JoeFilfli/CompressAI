"""Benchmark pure C++ rANS vs tANS encode/decode times.

Isolates the actual coder performance from Python/wrapper overhead.
Writes results to bench_results.txt to avoid stdout pollution.
"""

import os
import sys
import time

import torch

from compressai.zoo import models as zoo
from compressai import set_entropy_coder, available_entropy_coders
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.tans import TansEncoder, TansDecoder

set_entropy_coder(available_entropy_coders()[0])
net = zoo["bmshj2018-factorized"](quality=1, pretrained=True).eval()
net.update()

torch.manual_seed(42)
x = torch.rand(1, 3, 512, 768)
with torch.no_grad():
    y = net.g_a(x)

eb = net.entropy_bottleneck
indexes = eb._build_indexes(y.size())
medians = eb._get_medians().detach()
medians = eb._extend_ndims(medians, 2)
medians = medians.expand(y.size(0), -1, -1, -1)
symbols = eb.quantize(y, "symbols", medians)

# Pre-convert to lists (same cost for both coders)
sym_list = symbols[0].reshape(-1).int().tolist()
idx_list = indexes[0].reshape(-1).int().tolist()
cdf = eb._quantized_cdf.tolist()
cdf_len = eb._cdf_length.reshape(-1).int().tolist()
offsets = eb._offset.reshape(-1).int().tolist()

# Write results to file
out_path = os.path.join(os.path.dirname(__file__), "bench_results.txt")
out = open(out_path, "w")

n_sym = len(sym_list)

N = 20

# ── rANS ──────────────────────────────────────────────────────
encoder = BufferedRansEncoder()
# Warm up
encoder.encode_with_indexes(sym_list, idx_list, cdf, cdf_len, offsets)
encoded_rans = encoder.flush()

t0 = time.perf_counter()
for _ in range(N):
    encoder.encode_with_indexes(sym_list, idx_list, cdf, cdf_len, offsets)
    encoded_rans = encoder.flush()
t_rans_enc = (time.perf_counter() - t0) / N

decoder = RansDecoder()
decoder.decode_with_indexes(encoded_rans, idx_list, cdf, cdf_len, offsets)

t0 = time.perf_counter()
for _ in range(N):
    decoder.decode_with_indexes(encoded_rans, idx_list, cdf, cdf_len, offsets)
t_rans_dec = (time.perf_counter() - t0) / N

# ── tANS ──────────────────────────────────────────────────────
tans_enc = TansEncoder()
tans_dec = TansDecoder()

# Cold run (includes table building)
t0 = time.perf_counter()
encoded_tans = tans_enc.encode_with_indexes(sym_list, idx_list, cdf, cdf_len, offsets, 10)
t_cold_enc = time.perf_counter() - t0

# Warm encode
t0 = time.perf_counter()
for _ in range(N):
    encoded_tans = tans_enc.encode_with_indexes(sym_list, idx_list, cdf, cdf_len, offsets, 10)
t_tans_enc = (time.perf_counter() - t0) / N

# Cold decode
t0 = time.perf_counter()
tans_dec.decode_with_indexes(encoded_tans, idx_list, cdf, cdf_len, offsets, 10)
t_cold_dec = time.perf_counter() - t0

# Warm decode
t0 = time.perf_counter()
for _ in range(N):
    tans_dec.decode_with_indexes(encoded_tans, idx_list, cdf, cdf_len, offsets, 10)
t_tans_dec = (time.perf_counter() - t0) / N

# ── Write all results at once ─────────────────────────────────
L = 1 << 10
total_ct = sum((cl - 1) * L * 5 for cl in cdf_len)
total_dt = len(cdf) * L * 7

lines = []
lines.append(f"Symbols: {n_sym:,}  |  CDFs: {len(cdf)}  |  Max CDF len: {max(cdf_len)}")
lines.append("")
lines.append(f"=== Pure C++ (avg of {N} warm runs) ===")
lines.append(f"{'':30s} {'rANS':>10s} {'tANS':>10s} {'Speedup':>10s}")
lines.append("-" * 62)
lines.append(f"{'Encode (ms)':<30s} {t_rans_enc*1000:>10.2f} {t_tans_enc*1000:>10.2f} {t_rans_enc/t_tans_enc:>9.2f}x")
lines.append(f"{'Decode (ms)':<30s} {t_rans_dec*1000:>10.2f} {t_tans_dec*1000:>10.2f} {t_rans_dec/t_tans_dec:>9.2f}x")
lines.append(f"{'Bitstream (bytes)':<30s} {len(encoded_rans):>10d} {len(encoded_tans):>10d} {len(encoded_rans)/len(encoded_tans):>9.2f}x")
lines.append("")
lines.append(f"tANS cold encode (table build + encode): {t_cold_enc*1000:.2f} ms")
lines.append(f"tANS cold decode (table build + decode): {t_cold_dec*1000:.2f} ms")
lines.append("")
lines.append(f"=== Table memory (R=10, L={L}) ===")
lines.append(f"Encoding tables (CTable): {total_ct / 1024:.0f} KB  ({total_ct / 1024 / 1024:.1f} MB)")
lines.append(f"Decoding tables (DTable): {total_dt / 1024:.0f} KB  ({total_dt / 1024 / 1024:.1f} MB)")
lines.append(f"Total: {(total_ct + total_dt) / 1024 / 1024:.1f} MB")
lines.append(f"Typical CPU cache: L1=32 KB | L2=256 KB | L3=6-16 MB")

result = "\n".join(lines)
with open(out_path, "w") as f:
    f.write(result)
print("DONE")
