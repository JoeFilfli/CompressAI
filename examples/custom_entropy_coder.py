"""Custom entropy coder — tANS (table-based ANS) replacement for rANS.

This module replaces CompressAI's rANS entropy coder with a CPU-cache-
friendly tANS implementation that uses precomputed lookup tables instead
of arithmetic operations.

The C++ backend is in ``compressai/cpp_exts/tans/tans_interface.cpp``.
"""

from typing import List, Optional

import torch
from torch import Tensor

from compressai.tans import TansEncoder, TansDecoder

# ── Configurable table size parameter ────────────────────────────────────────
# R controls the tradeoff: larger R → better compression, slower table build
# Paper recommends R=9 for ~77% speed improvement with ~12% compression loss.
# R=10 is a good default for balanced performance.
TANS_R = 10

# Module-level encoder/decoder (reuse for table caching)
_encoder = TansEncoder()
_decoder = TansDecoder()


# ── Low-level encode / decode wrappers ───────────────────────────────────────

def encode(symbols, indexes, cdf, cdf_lengths, offsets):
    """Encode integer symbols to a byte string using tANS lookup tables."""
    return _encoder.encode_with_indexes(
        symbols, indexes, cdf, cdf_lengths, offsets, TANS_R
    )


def decode(bitstream, indexes, cdf, cdf_lengths, offsets):
    """Decode a byte string back to integer symbols using tANS."""
    return _decoder.decode_with_indexes(
        bitstream, indexes, cdf, cdf_lengths, offsets, TANS_R
    )


# ── High-level helpers (mirror EntropyModel.compress / decompress) ───────────

def entropy_bottleneck_compress(eb, x):
    """Replicate EntropyBottleneck.compress() using the custom coder."""
    indexes = eb._build_indexes(x.size())
    medians = eb._get_medians().detach()
    spatial_dims = len(x.size()) - 2
    medians = eb._extend_ndims(medians, spatial_dims)
    medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    symbols = eb.quantize(x, "symbols", medians)

    cdf = eb._quantized_cdf.tolist()
    cdf_lengths = eb._cdf_length.reshape(-1).int().tolist()
    offsets = eb._offset.reshape(-1).int().tolist()

    strings = []
    for i in range(symbols.size(0)):
        rv = encode(
            symbols[i].reshape(-1).int().tolist(),
            indexes[i].reshape(-1).int().tolist(),
            cdf, cdf_lengths, offsets,
        )
        strings.append(rv)
    return strings


def entropy_bottleneck_decompress(eb, strings, size):
    """Replicate EntropyBottleneck.decompress() using the custom coder."""
    output_size = (len(strings), eb._quantized_cdf.size(0), *size)
    indexes = eb._build_indexes(output_size).to(eb._quantized_cdf.device)
    medians = eb._extend_ndims(eb._get_medians().detach(), len(size))
    medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))

    cdf = eb._quantized_cdf.tolist()
    cdf_lengths = eb._cdf_length.reshape(-1).int().tolist()
    offsets = eb._offset.reshape(-1).int().tolist()

    outputs = eb._quantized_cdf.new_empty(indexes.size())
    for i, s in enumerate(strings):
        values = decode(
            s,
            indexes[i].reshape(-1).int().tolist(),
            cdf, cdf_lengths, offsets,
        )
        outputs[i] = torch.tensor(
            values, device=outputs.device, dtype=outputs.dtype
        ).reshape(outputs[i].size())

    outputs = eb.dequantize(outputs, medians, medians.dtype)
    return outputs


def gaussian_compress(gc, y, indexes, means=None):
    """Replicate GaussianConditional compress via the custom coder."""
    symbols = gc.quantize(y, "symbols", means)

    cdf = gc._quantized_cdf.tolist()
    cdf_lengths = gc._cdf_length.reshape(-1).int().tolist()
    offsets = gc._offset.reshape(-1).int().tolist()

    strings = []
    for i in range(symbols.size(0)):
        rv = encode(
            symbols[i].reshape(-1).int().tolist(),
            indexes[i].reshape(-1).int().tolist(),
            cdf, cdf_lengths, offsets,
        )
        strings.append(rv)
    return strings


def gaussian_decompress(gc, strings, indexes, dtype=torch.float, means=None):
    """Replicate GaussianConditional decompress via the custom coder."""
    cdf = gc._quantized_cdf.tolist()
    cdf_lengths = gc._cdf_length.reshape(-1).int().tolist()
    offsets = gc._offset.reshape(-1).int().tolist()

    outputs = gc._quantized_cdf.new_empty(indexes.size())
    for i, s in enumerate(strings):
        values = decode(
            s,
            indexes[i].reshape(-1).int().tolist(),
            cdf, cdf_lengths, offsets,
        )
        outputs[i] = torch.tensor(
            values, device=outputs.device, dtype=outputs.dtype
        ).reshape(outputs[i].size())

    outputs = gc.dequantize(outputs, means, dtype)
    return outputs
