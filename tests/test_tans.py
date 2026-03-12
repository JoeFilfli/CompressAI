"""Tests for the tANS (table-based ANS) entropy coder.

Covers:
  1. Basic roundtrip encoding/decoding (small alphabet)
  2. Roundtrip with larger symbol sequences
  3. Bypass coding for out-of-range symbols
  4. Multiple CDF tables (different indexes)
  5. Integration with CompressAI models (FactorizedPrior, ScaleHyperprior,
     MeanScaleHyperprior) — compress → decompress → PSNR check
"""

import pytest
import torch

from compressai.tans import TansEncoder, TansDecoder


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def encoder():
    return TansEncoder()


@pytest.fixture
def decoder():
    return TansDecoder()


def _make_uniform_cdf(num_regular_symbols, escape_prob=1024):
    """Build a CompressAI-style quantized CDF.

    Creates *num_regular_symbols* equiprobable symbols plus one escape
    symbol (with probability *escape_prob* / 2^16).  The escape symbol
    is required for bypass coding of out-of-range values.

    Returns a list of length ``num_regular_symbols + 2``
    (= ``num_regular_symbols`` CDF steps + escape end + total).
    """
    total = 1 << 16
    regular_prob = (total - escape_prob) // num_regular_symbols
    cdf = [0]
    for _ in range(num_regular_symbols):
        cdf.append(cdf[-1] + regular_prob)
    cdf.append(total)  # end-of-escape range
    return cdf


# ── 1. Basic roundtrip ──────────────────────────────────────────────────────

class TestBasicRoundtrip:
    @pytest.mark.parametrize("R", [8, 10, 12])
    def test_small_alphabet(self, encoder, decoder, R):
        """Encode / decode 10 symbols from a 3-symbol alphabet."""
        cdf = _make_uniform_cdf(3)
        cdfs = [cdf]
        cdf_sizes = [len(cdf)]
        offsets = [0]
        symbols = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        indexes = [0] * len(symbols)

        encoded = encoder.encode_with_indexes(
            symbols, indexes, cdfs, cdf_sizes, offsets, R
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, cdfs, cdf_sizes, offsets, R
        )
        assert list(decoded) == symbols

    def test_single_symbol(self, encoder, decoder):
        """One symbol should roundtrip correctly."""
        cdf = _make_uniform_cdf(5)
        encoded = encoder.encode_with_indexes(
            [3], [0], [cdf], [len(cdf)], [0], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, [0], [cdf], [len(cdf)], [0], 10
        )
        assert list(decoded) == [3]

    def test_empty_sequence(self, encoder, decoder):
        """Empty symbol list should produce a decodable (empty) bitstream."""
        cdf = _make_uniform_cdf(4)
        encoded = encoder.encode_with_indexes(
            [], [], [cdf], [len(cdf)], [0], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, [], [cdf], [len(cdf)], [0], 10
        )
        assert list(decoded) == []


# ── 2. Larger sequences ─────────────────────────────────────────────────────

class TestLargerSequences:
    @pytest.mark.parametrize("n", [100, 1000, 10_000])
    def test_random_symbols(self, encoder, decoder, n):
        """Roundtrip *n* random symbols from a 16-symbol alphabet."""
        num_sym = 16
        cdf = _make_uniform_cdf(num_sym)
        torch.manual_seed(42)
        symbols = torch.randint(0, num_sym, (n,)).tolist()
        indexes = [0] * n

        encoded = encoder.encode_with_indexes(
            symbols, indexes, [cdf], [len(cdf)], [0], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, [cdf], [len(cdf)], [0], 10
        )
        assert list(decoded) == symbols

    def test_skewed_distribution(self, encoder, decoder):
        """Non-uniform CDF (heavily skewed) roundtrips correctly."""
        # 4 symbols with PMF ~ [90%, 5%, 3%, 2%]
        total = 1 << 16
        cdf = [0, int(0.9 * total), int(0.95 * total), int(0.98 * total), total, 0]
        symbols = [0] * 500 + [1] * 30 + [2] * 15 + [3] * 5
        indexes = [0] * len(symbols)

        encoded = encoder.encode_with_indexes(
            symbols, indexes, [cdf], [len(cdf)], [0], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, [cdf], [len(cdf)], [0], 10
        )
        assert list(decoded) == symbols


# ── 3. Bypass coding ────────────────────────────────────────────────────────

class TestBypassCoding:
    def test_out_of_range_symbols(self, encoder, decoder):
        """Symbols outside the CDF range should be handled via bypass."""
        cdf = _make_uniform_cdf(4)  # regular symbols: [0..3], escape: 4
        offset = -2
        # With offset=-2, internal value = symbol - (-2) = symbol + 2
        # Valid range: [0, 4)  →  valid symbols: [-2, -1, 0, 1]
        # Out-of-range → bypass
        symbols = [-10, -1, 0, 50, 1, -5]  # mix of in-range and bypass
        indexes = [0] * len(symbols)

        encoded = encoder.encode_with_indexes(
            symbols, indexes, [cdf], [len(cdf)], [offset], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, [cdf], [len(cdf)], [offset], 10
        )
        assert list(decoded) == symbols

    def test_all_bypass(self, encoder, decoder):
        """All symbols out of range → everything is bypass-coded."""
        cdf = _make_uniform_cdf(2)  # regular: [0, 1], escape: 2
        symbols = [100, 200, 300]
        indexes = [0] * 3

        encoded = encoder.encode_with_indexes(
            symbols, indexes, [cdf], [len(cdf)], [0], 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, [cdf], [len(cdf)], [0], 10
        )
        assert list(decoded) == symbols


# ── 4. Multiple CDF tables ──────────────────────────────────────────────────

class TestMultipleCDFs:
    def test_two_tables(self, encoder, decoder):
        """Symbols referencing different CDF tables roundtrip correctly."""
        cdf0 = _make_uniform_cdf(4)   # 4-symbol uniform
        cdf1 = _make_uniform_cdf(8)   # 8-symbol uniform
        cdfs = [cdf0, cdf1]
        cdf_sizes = [len(cdf0), len(cdf1)]
        offsets = [0, 0]

        # Even positions use table 0 (max sym 3), odd use table 1 (max sym 7)
        symbols = [0, 7, 3, 5, 1, 2, 2, 6]
        indexes = [0, 1, 0, 1, 0, 1, 0, 1]

        encoded = encoder.encode_with_indexes(
            symbols, indexes, cdfs, cdf_sizes, offsets, 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, cdfs, cdf_sizes, offsets, 10
        )
        assert list(decoded) == symbols

    def test_many_tables(self, encoder, decoder):
        """16 different CDF tables, symbols randomly assigned."""
        num_tables = 16
        cdfs = [_make_uniform_cdf(4 + i) for i in range(num_tables)]
        cdf_sizes = [len(c) for c in cdfs]
        offsets = [0] * num_tables

        torch.manual_seed(123)
        n = 500
        indexes = torch.randint(0, num_tables, (n,)).tolist()
        # Each symbol must be in range for its table
        symbols = [
            torch.randint(0, 4 + idx, (1,)).item()
            for idx in indexes
        ]

        encoded = encoder.encode_with_indexes(
            symbols, indexes, cdfs, cdf_sizes, offsets, 10
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, cdfs, cdf_sizes, offsets, 10
        )
        assert list(decoded) == symbols


# ── 5. Different R values ───────────────────────────────────────────────────

class TestRValues:
    @pytest.mark.parametrize("R", [6, 8, 10, 12, 14])
    def test_roundtrip_at_different_R(self, encoder, decoder, R):
        """Roundtrip works for various table size parameters R."""
        cdf = _make_uniform_cdf(8)
        torch.manual_seed(7)
        symbols = torch.randint(0, 8, (200,)).tolist()
        indexes = [0] * 200

        encoded = encoder.encode_with_indexes(
            symbols, indexes, [cdf], [len(cdf)], [0], R
        )
        decoded = decoder.decode_with_indexes(
            encoded, indexes, [cdf], [len(cdf)], [0], R
        )
        assert list(decoded) == symbols


# ── 6. Integration with CompressAI models ────────────────────────────────────

class TestModelIntegration:
    """Full compress → decompress using tANS through CompressAI models.

    These tests verify that swapping rANS for tANS produces identical
    reconstructions (same PSNR).
    """

    @staticmethod
    def _get_model_output(model_name, quality):
        from compressai.zoo import models as zoo_models
        net = zoo_models[model_name](quality=quality, pretrained=True)
        net.eval()
        net.update()
        return net

    @staticmethod
    def _random_image(seed=0):
        torch.manual_seed(seed)
        return torch.rand(1, 3, 256, 256)

    @staticmethod
    def _psnr(a, b):
        mse = ((a - b) ** 2).mean()
        if mse == 0:
            return float("inf")
        return -10 * torch.log10(mse).item()

    def _roundtrip_rans(self, net, x):
        """Compress and decompress using the original rANS coder."""
        with torch.no_grad():
            compressed = net.compress(x)
            out = net.decompress(compressed["strings"], compressed["shape"])
        return out["x_hat"], compressed["strings"]

    def _roundtrip_tans(self, net, x, R=10):
        """Compress and decompress using the tANS coder."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from custom_entropy_coder import (
            entropy_bottleneck_compress,
            entropy_bottleneck_decompress,
            gaussian_compress,
            gaussian_decompress,
        )

        model_type = type(net).__name__
        with torch.no_grad():
            if model_type == "FactorizedPrior":
                y = net.g_a(x)
                strings = entropy_bottleneck_compress(net.entropy_bottleneck, y)
                y_hat = entropy_bottleneck_decompress(
                    net.entropy_bottleneck, strings, y.size()[-2:]
                )
                x_hat = net.g_s(y_hat)
            elif model_type in ("ScaleHyperprior", "MeanScaleHyperprior"):
                y = net.g_a(x)
                z = net.h_a(torch.abs(y) if model_type == "ScaleHyperprior" else y)
                z_strings = entropy_bottleneck_compress(net.entropy_bottleneck, z)
                z_hat = entropy_bottleneck_decompress(
                    net.entropy_bottleneck, z_strings, z.size()[-2:]
                )
                params = net.h_s(z_hat)
                if model_type == "ScaleHyperprior":
                    scales = params
                    indexes = net.gaussian_conditional.build_indexes(scales)
                    y_strings = gaussian_compress(
                        net.gaussian_conditional, y, indexes
                    )
                    y_hat = gaussian_decompress(
                        net.gaussian_conditional, y_strings, indexes, z_hat.dtype
                    )
                else:
                    scales, means = params.chunk(2, 1)
                    indexes = net.gaussian_conditional.build_indexes(scales)
                    y_strings = gaussian_compress(
                        net.gaussian_conditional, y, indexes, means=means
                    )
                    y_hat = gaussian_decompress(
                        net.gaussian_conditional, y_strings, indexes,
                        z_hat.dtype, means=means
                    )
                x_hat = net.g_s(y_hat)
            else:
                pytest.skip(f"Unsupported model type: {model_type}")
        return x_hat

    @pytest.mark.slow
    def test_factorized_prior_roundtrip(self):
        """FactorizedPrior: tANS produces same PSNR as rANS."""
        net = self._get_model_output("bmshj2018-factorized", quality=1)
        x = self._random_image()

        x_hat_rans, _ = self._roundtrip_rans(net, x)
        x_hat_tans = self._roundtrip_tans(net, x)

        psnr_rans = self._psnr(x, x_hat_rans)
        psnr_tans = self._psnr(x, x_hat_tans)

        # Must be within 0.01 dB (effectively identical — difference is
        # only rounding of the internal CDF discretisation)
        assert abs(psnr_rans - psnr_tans) < 0.01, (
            f"PSNR mismatch: rANS={psnr_rans:.4f} vs tANS={psnr_tans:.4f}"
        )

    @pytest.mark.slow
    def test_scale_hyperprior_roundtrip(self):
        """ScaleHyperprior: tANS and rANS produce same PSNR."""
        net = self._get_model_output("bmshj2018-hyperprior", quality=1)
        x = self._random_image()

        x_hat_rans, _ = self._roundtrip_rans(net, x)
        x_hat_tans = self._roundtrip_tans(net, x)

        psnr_rans = self._psnr(x, x_hat_rans)
        psnr_tans = self._psnr(x, x_hat_tans)

        assert abs(psnr_rans - psnr_tans) < 0.01, (
            f"PSNR mismatch: rANS={psnr_rans:.4f} vs tANS={psnr_tans:.4f}"
        )

    @pytest.mark.slow
    def test_mean_scale_hyperprior_roundtrip(self):
        """MeanScaleHyperprior: tANS and rANS produce same PSNR."""
        net = self._get_model_output("mbt2018-mean", quality=1)
        x = self._random_image()

        x_hat_rans, _ = self._roundtrip_rans(net, x)
        x_hat_tans = self._roundtrip_tans(net, x)

        psnr_rans = self._psnr(x, x_hat_rans)
        psnr_tans = self._psnr(x, x_hat_tans)

        assert abs(psnr_rans - psnr_tans) < 0.01, (
            f"PSNR mismatch: rANS={psnr_rans:.4f} vs tANS={psnr_tans:.4f}"
        )
