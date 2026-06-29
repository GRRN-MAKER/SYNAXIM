"""
Tests for the SYNAXIM core engine — Symbiotic State Machine execution.
Validates the full pipeline: config → convert → load → forward → generate.
"""

import numpy as np
import os
import tempfile
import json
import pytest

from grrn_inference.config import SymbioticConfig
from grrn_inference.engine import SymbioticStateEngine, LayerWeights
from grrn_inference.sampling import SamplingParams, TokenSampler


class TestSymbioticGate:
    """Test the core Symbiotic Gate: M matrix update and retrieval."""

    def _make_minimal_model(self, tmpdir, D=64, V=256, num_layers=2):
        """Create a minimal .symb model for testing."""
        os.makedirs(os.path.join(tmpdir, "tokenizer"), exist_ok=True)
        for i in range(num_layers):
            os.makedirs(os.path.join(tmpdir, f"layers/layer_{i:02d}"), exist_ok=True)

        # Config
        config = SymbioticConfig(
            model_name="test-mini",
            hidden_size=D,
            intermediate_size=D * 4,
            num_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=D // 4,
            vocab_size=V,
            layer_types=["linear_attention"] * num_layers,
            quantization={"method": "fp16", "group_size": 128, "symmetric": True},
        )
        config.save(os.path.join(tmpdir, "config.symb.json"))

        # Embeddings (V, D) as FP16
        emb = np.random.randn(V, D).astype(np.float16)
        emb.tofile(os.path.join(tmpdir, "embeddings.symb"))

        # LM head (D, V) as FP16
        lm = np.random.randn(D, V).astype(np.float16)
        lm.tofile(os.path.join(tmpdir, "lm_head.symb"))

        # Final norm (D,) as FP16
        norm = np.ones(D, dtype=np.float16)
        norm.tofile(os.path.join(tmpdir, "final_norm.symb"))

        # Per-layer weights
        for i in range(num_layers):
            layer_dir = os.path.join(tmpdir, f"layers/layer_{i:02d}")
            # Norms
            np.ones(D, dtype=np.float16).tofile(os.path.join(layer_dir, "norm_attn.symb"))
            np.ones(D, dtype=np.float16).tofile(os.path.join(layer_dir, "norm_mlp.symb"))
            # Attention weights (FP16)
            for name, shape in [
                ("attn_q", (4 * (D // 4), D)),
                ("attn_k", (4 * (D // 4), D)),
                ("attn_v", (4 * (D // 4), D)),
                ("attn_o", (D, 4 * (D // 4))),
            ]:
                w = (np.random.randn(*shape) * 0.02).astype(np.float16)
                w.tofile(os.path.join(layer_dir, f"{name}.symb"))
            # MLP weights (FP16)
            for name, shape in [
                ("mlp_gate", (D * 4, D)),
                ("mlp_up", (D * 4, D)),
                ("mlp_down", (D, D * 4)),
            ]:
                w = (np.random.randn(*shape) * 0.02).astype(np.float16)
                w.tofile(os.path.join(layer_dir, f"{name}.symb"))

        # Minimal tokenizer
        tok_config = {"model_type": "test", "eos_token": "</s>"}
        with open(os.path.join(tmpdir, "tokenizer", "tokenizer_config.json"), "w") as f:
            json.dump(tok_config, f)

        return config

    def test_forward_produces_logits(self):
        """Engine.forward() should return logits of shape (vocab_size,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_model(tmpdir, D=64, V=256, num_layers=2)
            engine = SymbioticStateEngine(tmpdir)

            logits = engine.forward(0)  # Token ID 0
            assert logits.shape == (256,), f"Expected (256,), got {logits.shape}"
            assert not np.any(np.isnan(logits)), "Logits contain NaN"
            assert not np.any(np.isinf(logits)), "Logits contain Inf"

    def test_m_matrix_updates(self):
        """M matrices should change after processing tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_model(tmpdir, D=64, V=256, num_layers=2)
            engine = SymbioticStateEngine(tmpdir)

            M_before = engine.M[0].copy()
            engine.forward(42)
            M_after = engine.M[0]

            # M should have been updated
            diff = np.max(np.abs(M_after - M_before))
            assert diff > 0, "M matrix was not updated during forward pass"

    def test_reset_clears_state(self):
        """reset() should zero all M matrices and position."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_model(tmpdir, D=64, V=256, num_layers=2)
            engine = SymbioticStateEngine(tmpdir)

            # Run some tokens
            engine.forward(1)
            engine.forward(2)
            assert engine._position == 2

            # Reset
            engine.reset()
            assert engine._position == 0
            for M in engine.M:
                assert np.allclose(M, 0), "M matrix not zeroed after reset"

    def test_different_tokens_different_logits(self):
        """Different input tokens should produce different logits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_model(tmpdir, D=64, V=256, num_layers=2)
            engine = SymbioticStateEngine(tmpdir)

            logits_a = engine.forward(10)
            engine.reset()
            logits_b = engine.forward(20)

            # Logits should differ for different tokens
            assert not np.allclose(logits_a, logits_b), \
                "Different tokens produced identical logits"

    def test_o1_memory(self):
        """Memory footprint should be O(1) — M matrices stay fixed size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            D = 64
            config = self._make_minimal_model(tmpdir, D=D, V=256, num_layers=2)
            engine = SymbioticStateEngine(tmpdir)

            # Process many tokens — M size should not change
            for tok in range(50):
                engine.forward(tok % 256)

            for M in engine.M:
                assert M.shape == (D, D), \
                    f"M matrix shape changed: expected ({D},{D}), got {M.shape}"


class TestSampling:
    """Test token sampling strategies."""

    def test_greedy_sampling(self):
        """Temperature=0 should always pick argmax."""
        params = SamplingParams(temperature=0.0, seed=42)
        sampler = TokenSampler(params)

        logits = np.array([1.0, 5.0, 2.0, 3.0])
        token = sampler.sample(logits)
        assert token == 1, f"Greedy should pick index 1 (max), got {token}"

    def test_top_k_filtering(self):
        """Top-K should zero out all but K highest logits."""
        logits = np.array([1.0, 5.0, 2.0, 3.0, 4.0])
        filtered = TokenSampler._top_k_filter(logits.copy(), k=2)
        finite_count = np.sum(filtered > -np.inf)
        assert finite_count == 2, f"Top-K=2 should leave 2 tokens, got {finite_count}"

    def test_top_p_filtering(self):
        """Top-P should keep smallest set with cumulative prob >= p."""
        logits = np.array([1.0, 10.0, 2.0, 3.0])
        filtered = TokenSampler._top_p_filter(logits.copy(), p=0.9)
        # Token with logit=10 has >90% of probability mass
        assert filtered[1] > -np.inf, "Highest logit token should survive top-p"

    def test_repetition_penalty(self):
        """Repetition penalty should reduce scores of repeated tokens."""
        params = SamplingParams(temperature=1.0, repetition_penalty=2.0, seed=42)
        sampler = TokenSampler(params)

        logits = np.array([5.0, 5.0, 5.0])
        penalized = sampler._apply_repetition_penalty(logits.copy(), [0, 0, 1])
        # Token 0 appeared twice, should be penalized most
        assert penalized[0] < logits[0], "Repeated token should be penalized"
        assert penalized[2] == logits[2], "Non-repeated token should be unchanged"

    def test_seed_reproducibility(self):
        """Same seed should produce same samples."""
        logits = np.random.randn(1000).astype(np.float64)

        params = SamplingParams(temperature=1.0, seed=12345)
        s1 = TokenSampler(params)
        t1 = s1.sample(logits.copy())

        params = SamplingParams(temperature=1.0, seed=12345)
        s2 = TokenSampler(params)
        t2 = s2.sample(logits.copy())

        assert t1 == t2, f"Same seed should give same result: {t1} vs {t2}"


class TestSanitizer:
    """Test _sanitize_for_chat_template input sanitization."""

    def test_normal_text_unchanged(self):
        """Normal text should pass through unchanged."""
        from grrn_inference import _sanitize_for_chat_template
        text = "Hello, how are you doing today?"
        assert _sanitize_for_chat_template(text) == text

    def test_strips_html(self):
        """HTML tags should be stripped."""
        from grrn_inference import _sanitize_for_chat_template
        text = "<script>alert('xss')</script>Hello"
        result = _sanitize_for_chat_template(text)
        assert "<script>" not in result
        assert "Hello" in result

    def test_neutralizes_jinja2(self):
        """Jinja2 template injection should be neutralized."""
        from grrn_inference import _sanitize_for_chat_template
        text = "{{ config.__class__.__init__.__globals__ }}"
        result = _sanitize_for_chat_template(text)
        assert "{{" not in result
        assert "}}" not in result

    def test_truncates_word_repetition(self):
        """Repeated words (4+) should be truncated."""
        from grrn_inference import _sanitize_for_chat_template
        text = "the the the the the the the end"
        result = _sanitize_for_chat_template(text)
        count = result.lower().split().count("the")
        assert count <= 4, f"Expected <= 4 'the', got {count}"

    def test_collapses_whitespace(self):
        """Multiple whitespace should be collapsed."""
        from grrn_inference import _sanitize_for_chat_template
        text = "hello    world\n\n\nfoo"
        result = _sanitize_for_chat_template(text)
        assert "    " not in result
        assert "\n\n" not in result

    def test_length_cap(self):
        """Text longer than 32000 chars should be capped."""
        from grrn_inference import _sanitize_for_chat_template
        text = "a" * 50000
        result = _sanitize_for_chat_template(text)
        assert len(result) <= 32000

    def test_non_string_input(self):
        """Non-string input should be converted."""
        from grrn_inference import _sanitize_for_chat_template
        result = _sanitize_for_chat_template(12345)
        assert result == "12345"


class TestConfig:
    """Test SymbioticConfig extended functionality."""

    def test_full_attention_default(self):
        """Default layer type should be full_attention."""
        config = SymbioticConfig(num_layers=4)
        assert config.get_layer_type(0) == "full_attention"
        assert config.get_layer_type(99) == "full_attention"

    def test_mixed_layer_types(self):
        """Mixed layer types should be correctly reported."""
        config = SymbioticConfig(
            num_layers=4,
            layer_types=["linear_attention", "linear_attention",
                         "full_attention", "full_attention"],
        )
        assert config.num_linear_attention_layers == 2
        assert config.num_full_attention_layers == 2

    def test_preset_qwen(self):
        """Qwen preset should have correct dimensions."""
        from grrn_inference.config import preset_qwen
        config = preset_qwen("27b")
        assert config.hidden_size == 5120
        assert config.num_layers == 64
        assert config.num_linear_attention_layers == 48
        assert config.num_full_attention_layers == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
