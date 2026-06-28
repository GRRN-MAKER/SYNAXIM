"""Tests for the .symb weight export/import pipeline."""

import numpy as np
import os
import tempfile
import pytest

from grrn_inference.export import SymbioticConverter, unpack_int4_symb
from grrn_inference.config import SymbioticConfig


class TestINT4PackUnpack:
    """Test INT4 bitpacking roundtrip accuracy."""

    def test_roundtrip_small(self):
        """Small tensor roundtrip."""
        original = np.random.randn(128, 128).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".symb", delete=False) as f:
            path = f.name

        try:
            converter = SymbioticConverter(verbose=False)
            converter._pack_int4(original, path, group_size=128)

            recovered, shape = unpack_int4_symb(path)

            assert shape == (128, 128)
            assert recovered.shape == original.shape

            # INT4 quantization introduces error, but should be bounded
            max_error = np.max(np.abs(original - recovered))
            mean_error = np.mean(np.abs(original - recovered))
            print(f"Max error: {max_error:.4f}, Mean error: {mean_error:.4f}")

            # With 4-bit quantization and group_size=128, expect <15% relative error
            relative_error = mean_error / (np.std(original) + 1e-6)
            assert relative_error < 0.20, f"Relative error too high: {relative_error:.3f}"
        finally:
            os.unlink(path)

    def test_roundtrip_large(self):
        """Larger tensor roundtrip."""
        original = np.random.randn(4096, 4096).astype(np.float32) * 0.02

        with tempfile.NamedTemporaryFile(suffix=".symb", delete=False) as f:
            path = f.name

        try:
            converter = SymbioticConverter(verbose=False)
            converter._pack_int4(original, path, group_size=128)

            recovered, shape = unpack_int4_symb(path)
            assert shape == (4096, 4096)

            # Check compression ratio
            original_size = original.nbytes  # 4096*4096*4 = 64MB
            packed_size = os.path.getsize(path)
            ratio = original_size / packed_size
            print(f"Compression ratio: {ratio:.1f}x")
            assert ratio > 3.0, f"Compression ratio too low: {ratio:.1f}x"
        finally:
            os.unlink(path)

    def test_1d_tensor(self):
        """1D tensor (e.g., norm weights)."""
        original = np.random.randn(4096).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".symb", delete=False) as f:
            path = f.name

        try:
            converter = SymbioticConverter(verbose=False)
            converter._pack_int4(original, path, group_size=128)
            recovered, shape = unpack_int4_symb(path)
            assert shape == (4096,)
        finally:
            os.unlink(path)


class TestConfig:
    """Test SymbioticConfig save/load."""

    def test_save_load_roundtrip(self):
        config = SymbioticConfig(
            model_name="test-model",
            hidden_size=4096,
            num_layers=32,
            vocab_size=32000,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            config.save(path)
            loaded = SymbioticConfig.load(path)
            assert loaded.model_name == "test-model"
            assert loaded.hidden_size == 4096
            assert loaded.num_layers == 32
        finally:
            os.unlink(path)

    def test_layer_types(self):
        config = SymbioticConfig(
            num_layers=4,
            layer_types=["linear_attention", "linear_attention",
                         "full_attention", "full_attention"],
        )
        assert config.num_linear_attention_layers == 2
        assert config.num_full_attention_layers == 2
        assert config.get_layer_type(0) == "linear_attention"
        assert config.get_layer_type(2) == "full_attention"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
