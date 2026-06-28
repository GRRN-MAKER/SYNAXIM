"""
SYNAXIM: CPU-Accelerated Symbiotic State Engine
=================================================
Powered by SYNAXIM — Symbiotic Native Axiom Inference Machine.
Drop-in replacement for the NumPy engine that uses Numba-compiled
LLVM kernels for hardware-accelerated inference on CPU.

All math operations are compiled to native AVX-512/AVX2/NEON
instructions. The M matrix is mutated in-place with zero memory
allocations during generation.

Same .symb weight format, same API — just faster.

Usage:
    from grrn_inference.cpu_engine import CPUSymbioticEngine
    engine = CPUSymbioticEngine(model_dir)
    logits = engine.forward(token_id)
"""

from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional

import numpy as np

from .config import SymbioticConfig
from .engine import LayerWeights
from .cpu_kernels import (
    CPUAcceleratedOps,
    HAS_NUMBA,
)


class CPUSymbioticEngine:
    """
    CPU-accelerated 64-layer inference engine.

    Uses Numba JIT-compiled kernels for all math operations.
    Falls back to NumPy if Numba is not available.

    Key differences from the base NumPy engine:
    - All matmuls use parallel prange across CPU cores
    - SwiGLU MLP is fused into a single kernel (no intermediate arrays)
    - Symbiotic Gate does SmoothQuant + sigmoid + M mutation + projection in one pass
    - RoPE rotation happens in-register
    - M matrices are mutated in-place (zero allocations per token)
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.config = SymbioticConfig.from_model_dir(model_dir)
        self._ops = CPUAcceleratedOps()

        print(f"[SYNAXIM] Initializing CPU-Accelerated Engine (Numba={'ON' if HAS_NUMBA else 'OFF'})")
        print(f"[SYNAXIM] Model: {self.config.model_name} | Layers: {self.config.num_layers} | D: {self.config.hidden_size}")

        cfg = self.config
        D = cfg.hidden_size
        V = cfg.vocab_size

        # ── Load Global Weights ──
        self.embeddings = self._load_fp16(
            os.path.join(model_dir, cfg.embedding_file), (V, D)
        )

        lm_head_path = os.path.join(model_dir, cfg.lm_head_file)
        if os.path.exists(lm_head_path):
            self.lm_head = self._load_fp16(lm_head_path, (D, V))
        elif cfg.tie_word_embeddings:
            self.lm_head = self.embeddings.T.copy()
        else:
            raise FileNotFoundError("lm_head.symb not found and embeddings not tied")

        final_norm_path = os.path.join(model_dir, "final_norm.symb")
        if os.path.exists(final_norm_path):
            self.final_norm_weight = self._load_fp16(final_norm_path, (D,))
        else:
            self.final_norm_weight = np.ones(D, dtype=np.float32)

        # ── Load Per-Layer Weights ──
        self.layers: List[LayerWeights] = []
        for i in range(cfg.num_layers):
            layer_dir = os.path.join(model_dir, cfg.get_layer_dir(i))
            layer_type = cfg.get_layer_type(i)
            layer = LayerWeights(layer_dir, cfg, i, layer_type)
            self.layers.append(layer)

        # ── Persistent State Matrices (mutated in-place by CPU kernels) ──
        self.M: List[np.ndarray] = [
            np.zeros((D, D), dtype=np.float32) for _ in range(cfg.num_layers)
        ]

        # ── RoPE Precomputation ──
        self._rope_cos, self._rope_sin = self._precompute_rope(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )
        self._position = 0

        # ── Pre-cache all dequantized weights for cache locality ──
        # Loading all weights eagerly keeps them hot in CPU cache
        self._preload_weights()

        # ── Warm up Numba JIT (compile all kernels) ──
        if HAS_NUMBA:
            self._ops.warmup()

    def _load_fp16(self, path: str, shape: tuple) -> np.ndarray:
        data = np.fromfile(path, dtype=np.float16)
        return data.reshape(shape).astype(np.float32)

    @staticmethod
    def _precompute_rope(
        head_dim: int, max_seq: int, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        positions = np.arange(max_seq, dtype=np.float32)
        angles = np.outer(positions, freqs)
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def _preload_weights(self):
        """
        Eagerly load and cache all dequantized layer weights.
        This ensures sequential memory layout for optimal CPU cache behavior.
        """
        self._cached_weights: List[dict] = []
        for i, layer in enumerate(self.layers):
            w = {}
            layer_type = self.config.get_layer_type(i)

            # Attention weights
            w["attn_q"] = layer.get_weight("attn_q")
            w["attn_k"] = layer.get_weight("attn_k")
            w["attn_v"] = layer.get_weight("attn_v")
            w["attn_o"] = layer.get_weight("attn_o")

            # MLP weights
            w["mlp_gate"] = layer.get_weight("mlp_gate")
            w["mlp_up"] = layer.get_weight("mlp_up")
            w["mlp_down"] = layer.get_weight("mlp_down")

            # Norms
            w["norm_attn"] = layer.norm_attn
            w["norm_mlp"] = layer.norm_mlp

            # MoE router (if applicable)
            if layer.is_moe:
                try:
                    w["router"] = layer.get_weight("router")
                except FileNotFoundError:
                    pass

            self._cached_weights.append(w)

            # Free the lazy cache in LayerWeights since we own the data now
            layer.clear_cache()

    # ══════════════════════════════════════════════════════════
    # Layer Execution — Symbiotic Gate (Linear Attention)
    # ══════════════════════════════════════════════════════════

    def _symbiotic_gate(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Execute Continuous Symbiotic Gate using CPU-accelerated kernel.
        M[layer_idx] is modified IN-PLACE — zero allocations.
        """
        W_gate = self._cached_weights[layer_idx]["attn_q"]  # Use Q proj as gate

        if HAS_NUMBA:
            return self._ops.symbiotic_gate(
                x, self.M[layer_idx], W_gate, self.config.gate_alpha
            )
        else:
            # NumPy fallback
            return self._symbiotic_gate_numpy(x, layer_idx, W_gate)

    def _symbiotic_gate_numpy(self, x, layer_idx, W_gate):
        """NumPy fallback for symbiotic gate."""
        D = self.config.hidden_size
        M = self.M[layer_idx]
        alpha = self.config.gate_alpha

        max_x = np.max(np.abs(x)) + 1e-6
        max_m = np.max(np.abs(M), axis=0) + 1e-6
        s = np.power(max_x, alpha) * np.power(max_m, 1.0 - alpha)
        x_smooth = x / s

        gate_scores = x_smooth @ W_gate.T
        gate = 1.0 / (1.0 + np.exp(-np.clip(gate_scores, -60, 60)))
        gate_mean = np.mean(gate)

        imprint = np.outer(x_smooth, x_smooth)
        self.M[layer_idx] = gate_mean * M + (1.0 - gate_mean) * imprint

        return x_smooth @ self.M[layer_idx]

    # ══════════════════════════════════════════════════════════
    # Layer Execution — Full Attention
    # ══════════════════════════════════════════════════════════

    def _full_attention(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Full dot-product attention using CPU-accelerated kernels."""
        cfg = self.config
        w = self._cached_weights[layer_idx]
        hd = cfg.head_dim
        half = hd // 2

        cos = self._rope_cos[self._position, :half].copy()
        sin = self._rope_sin[self._position, :half].copy()

        if HAS_NUMBA:
            return self._ops.full_attention(
                x, w["attn_q"], w["attn_k"], w["attn_v"], w["attn_o"],
                cos, sin,
                cfg.num_attention_heads, cfg.num_key_value_heads, hd,
            )
        else:
            return self._full_attention_numpy(x, layer_idx, cos, sin)

    def _full_attention_numpy(self, x, layer_idx, cos, sin):
        """NumPy fallback for full attention."""
        cfg = self.config
        w = self._cached_weights[layer_idx]
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        half = hd // 2

        q = (x @ w["attn_q"].T).reshape(n_heads, hd)
        k = (x @ w["attn_k"].T).reshape(n_kv, hd)
        v = (x @ w["attn_v"].T).reshape(n_kv, hd)

        # RoPE
        for h in range(n_heads):
            x1, x2 = q[h, :half].copy(), q[h, half:].copy()
            q[h, :half] = x1 * cos - x2 * sin
            q[h, half:] = x1 * sin + x2 * cos
        for h in range(n_kv):
            x1, x2 = k[h, :half].copy(), k[h, half:].copy()
            k[h, :half] = x1 * cos - x2 * sin
            k[h, half:] = x1 * sin + x2 * cos

        # GQA repeat
        if n_kv < n_heads:
            k = np.repeat(k, n_heads // n_kv, axis=0)
            v = np.repeat(v, n_heads // n_kv, axis=0)

        # Single-token attention → output = V (attention weight is always 1.0)
        out = v.reshape(-1)
        return out @ w["attn_o"].T

    # ══════════════════════════════════════════════════════════
    # MLP Execution
    # ══════════════════════════════════════════════════════════

    def _mlp(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """SwiGLU MLP using fused CPU kernel."""
        w = self._cached_weights[layer_idx]

        if HAS_NUMBA:
            intermediate = self._ops.swiglu_fused(x, w["mlp_gate"], w["mlp_up"])
            return self._ops.matvec(w["mlp_down"].T, intermediate)
        else:
            gate_out = x @ w["mlp_gate"].T
            gate_out = gate_out * (1.0 / (1.0 + np.exp(-np.clip(gate_out, -60, 60))))
            up_out = x @ w["mlp_up"].T
            hidden = gate_out * up_out
            return hidden @ w["mlp_down"].T

    # ══════════════════════════════════════════════════════════
    # MoE MLP
    # ══════════════════════════════════════════════════════════

    def _moe_mlp(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Mixture-of-Experts MLP with top-K routing."""
        cfg = self.config
        w = self._cached_weights[layer_idx]
        layer = self.layers[layer_idx]

        # Router
        W_router = w.get("router")
        if W_router is None:
            return self._mlp(x, layer_idx)

        router_logits = x @ W_router.T
        router_logits_max = np.max(router_logits)
        router_probs = np.exp(router_logits - router_logits_max)
        router_probs = router_probs / np.sum(router_probs)

        top_k = cfg.num_experts_per_tok
        top_indices = np.argsort(router_probs)[-top_k:][::-1]
        top_weights = router_probs[top_indices]
        top_weights = top_weights / np.sum(top_weights)

        result = np.zeros_like(x)
        for i, exp_idx in enumerate(top_indices):
            W_gate = layer.get_expert_weight(int(exp_idx), "gate")
            W_up = layer.get_expert_weight(int(exp_idx), "up")
            W_down = layer.get_expert_weight(int(exp_idx), "down")

            if HAS_NUMBA:
                intermediate = self._ops.swiglu_fused(x, W_gate, W_up)
                expert_out = self._ops.matvec(W_down.T, intermediate)
            else:
                gate_out = x @ W_gate.T
                gate_out = gate_out * (1.0 / (1.0 + np.exp(-np.clip(gate_out, -60, 60))))
                up_out = x @ W_up.T
                expert_out = (gate_out * up_out) @ W_down.T

            result += top_weights[i] * expert_out

        return result

    # ══════════════════════════════════════════════════════════
    # Full Forward Pass
    # ══════════════════════════════════════════════════════════

    def forward(self, token_id: int) -> np.ndarray:
        """
        Full forward pass for a single token through all layers.
        Returns logits: (vocab_size,)
        """
        cfg = self.config
        eps = cfg.rms_norm_eps

        # 1. Token embedding lookup
        h = self.embeddings[token_id].copy()

        # 2. Process through all layers
        for i in range(cfg.num_layers):
            w = self._cached_weights[i]
            layer_type = cfg.get_layer_type(i)

            # Pre-attention norm
            if HAS_NUMBA:
                h_normed = self._ops.rmsnorm(h, w["norm_attn"], eps)
            else:
                variance = np.mean(h * h)
                h_normed = h / np.sqrt(variance + eps) * w["norm_attn"]

            # Attention / Symbiotic Gate
            if layer_type == "linear_attention":
                attn_out = self._symbiotic_gate(h_normed, i)
            else:
                attn_out = self._full_attention(h_normed, i)

            # Residual
            if HAS_NUMBA:
                h = self._ops.residual_add(h, attn_out)
            else:
                h = h + attn_out

            # Pre-MLP norm
            if HAS_NUMBA:
                h_normed = self._ops.rmsnorm(h, w["norm_mlp"], eps)
            else:
                variance = np.mean(h * h)
                h_normed = h / np.sqrt(variance + eps) * w["norm_mlp"]

            # MLP
            if self.layers[i].is_moe:
                mlp_out = self._moe_mlp(h_normed, i)
            else:
                mlp_out = self._mlp(h_normed, i)

            # Residual
            if HAS_NUMBA:
                h = self._ops.residual_add(h, mlp_out)
            else:
                h = h + mlp_out

        # 3. Final norm
        if HAS_NUMBA:
            h = self._ops.rmsnorm(h, self.final_norm_weight, eps)
        else:
            variance = np.mean(h * h)
            h = h / np.sqrt(variance + eps) * self.final_norm_weight

        # 4. Project to vocabulary logits
        if HAS_NUMBA:
            logits = self._ops.vecmat(h, self.lm_head)
        else:
            logits = h @ self.lm_head

        self._position += 1
        return logits

    def generate_token(self, token_id: int) -> int:
        """Generate the next token ID."""
        logits = self.forward(token_id)
        return int(np.argmax(logits))

    def reset(self) -> None:
        """Reset all state matrices and position counter."""
        D = self.config.hidden_size
        for i in range(len(self.M)):
            self.M[i][:] = 0.0  # In-place zero (no allocation)
        self._position = 0
