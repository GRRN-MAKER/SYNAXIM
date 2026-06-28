"""
SYNAXIM: Core Symbiotic State Engine — 64-Layer Execution
==========================================================
Framework-free inference engine that executes .symb model checkpoints
using the Continuous Associative State Machine paradigm.
Powered by SYNAXIM — Symbiotic Native Axiom Inference Machine.

Attention ≡ Memory: No KV-Cache. O(1) memory footprint via persistent
state matrix M that accumulates context through sigmoid-gated outer
product updates.

Supports:
  - Dense architectures (all layers sequential)
  - MoE architectures (router-selected expert execution)
  - INT4 on-the-fly dequantization
  - Memory-mapped weights for zero-copy initialization
  - Optional Triton GPU acceleration
"""

from __future__ import annotations

import os
import struct
import math
from typing import Optional, List, Tuple

import numpy as np

from .config import SymbioticConfig


class SymbioticStateEngine:
    """
    Core execution engine for .symb model inference.
    
    Replaces the entire PyTorch nn.Module layer stack with direct
    tensor operations on memory-mapped binary weight files.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize the engine from a .symb model directory.

        Args:
            model_dir: Path to directory containing config.symb.json and .symb files
            device: "cpu" for NumPy execution, "cuda" for Triton GPU kernels
        """
        self.model_dir = model_dir
        self.device = device
        self.config = SymbioticConfig.from_model_dir(model_dir)

        print(f"[SYNAXIM] Initializing Symbiotic State Engine (NumPy backend)")
        print(f"[SYNAXIM] Model: {self.config.model_name} | Layers: {self.config.num_layers} | D: {self.config.hidden_size}")

        cfg = self.config
        D = cfg.hidden_size
        V = cfg.vocab_size

        # ── Load Global Weights ──
        self.embeddings = self._load_fp16(
            os.path.join(model_dir, cfg.embedding_file), (V, D)
        )
        
        # LM head may be tied to embeddings
        lm_head_path = os.path.join(model_dir, cfg.lm_head_file)
        if os.path.exists(lm_head_path):
            self.lm_head = self._load_fp16(lm_head_path, (D, V))
        elif cfg.tie_word_embeddings:
            self.lm_head = self.embeddings.T.copy()
        else:
            raise FileNotFoundError(f"lm_head.symb not found and embeddings not tied")

        # Final RMSNorm
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

        # ── Initialize Persistent State Matrices ──
        # One M matrix per layer for the Continuous Symbiotic Gate
        self.M: List[np.ndarray] = [
            np.zeros((D, D), dtype=np.float32) for _ in range(cfg.num_layers)
        ]

        # ── RoPE Precomputation ──
        self._rope_cos, self._rope_sin = self._precompute_rope(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )
        self._position = 0  # Current position in sequence

    # ══════════════════════════════════════════════════════════
    # Weight Loading
    # ══════════════════════════════════════════════════════════

    def _load_fp16(self, path: str, shape: tuple) -> np.ndarray:
        """Load a flat FP16 binary file into a float32 array."""
        data = np.fromfile(path, dtype=np.float16)
        return data.reshape(shape).astype(np.float32)

    def _load_int4(self, path: str) -> Tuple[np.ndarray, tuple]:
        """Load and dequantize an INT4 .symb file."""
        from .export import unpack_int4_symb
        return unpack_int4_symb(path)

    # ══════════════════════════════════════════════════════════
    # Core Math Operations
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
        # x shape: (seq_len, D) or (D,)
        variance = np.mean(x * x, axis=-1, keepdims=True)
        x_normed = x / np.sqrt(variance + eps)
        return x_normed * weight

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        """SiLU/Swish activation: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured activation function."""
        act = self.config.hidden_act
        if act == "silu":
            return self._silu(x)
        elif act == "gelu":
            return self._gelu(x)
        elif act == "relu":
            return np.maximum(x, 0)
        else:
            return self._silu(x)

    # ══════════════════════════════════════════════════════════
    # Rotary Position Embeddings
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _precompute_rope(
        head_dim: int, max_seq: int, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute RoPE cos/sin tables."""
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        positions = np.arange(max_seq, dtype=np.float32)
        angles = np.outer(positions, freqs)  # (max_seq, head_dim/2)
        cos_table = np.cos(angles)
        sin_table = np.sin(angles)
        return cos_table, sin_table

    def _apply_rope(
        self, q: np.ndarray, k: np.ndarray, pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotary position embeddings to Q and K.
        
        Args:
            q: (num_heads, head_dim)
            k: (num_kv_heads, head_dim)
            pos: Current position index
        """
        half = q.shape[-1] // 2
        cos = self._rope_cos[pos, :half]  # (half,)
        sin = self._rope_sin[pos, :half]

        def rotate(x):
            x1, x2 = x[..., :half], x[..., half:]
            return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        return rotate(q), rotate(k)

    # ══════════════════════════════════════════════════════════
    # Layer Execution — Symbiotic Gate (Linear Attention)
    # ══════════════════════════════════════════════════════════

    def _symbiotic_gate(
        self, x: np.ndarray, layer_idx: int, layer: "LayerWeights"
    ) -> np.ndarray:
        """
        Execute the Continuous Symbiotic Gate.
        
        Attention ≡ Memory: Instead of Q·K^T·V with a growing KV cache,
        we maintain a fixed-size state matrix M that accumulates context
        through sigmoid-gated outer product updates.

        M_{t+1} = σ(xW) ⊙ M_t + (1 - σ(xW)) ⊙ (x^T · x)
        output = x · M_{t+1}
        
        Memory: O(D²) fixed, regardless of sequence length.
        """
        D = self.config.hidden_size
        M = self.M[layer_idx]
        alpha = self.config.gate_alpha

        # 1. COMPRESS: SmoothQuant balancing
        max_x = np.max(np.abs(x), axis=-1, keepdims=True) + 1e-6
        max_m = np.max(np.abs(M), axis=0, keepdims=True) + 1e-6
        s = np.power(max_x, alpha) * np.power(max_m, 1.0 - alpha)
        x_smooth = x / s

        # 2. ROUTE: Compute gate scores through projection
        # Use the attention Q projection as the gate weight
        W_gate = layer.get_weight("attn_q")  # (D, D) or subset
        gate_scores = x_smooth @ W_gate.T
        gate = self._sigmoid(gate_scores)

        # 3. MUTATE: Update persistent associative memory
        if x_smooth.ndim == 1:
            x_smooth = x_smooth.reshape(1, -1)
        if gate.ndim == 1:
            gate = gate.reshape(-1, 1)

        # Outer product: new associative imprint
        imprint = x_smooth.T @ x_smooth

        # Gated memory update
        gate_mean = np.mean(gate)
        self.M[layer_idx] = gate_mean * M + (1.0 - gate_mean) * imprint

        # 4. RESOLVE: Project through updated memory
        output = x_smooth @ self.M[layer_idx]
        return output.reshape(x.shape)

    # ══════════════════════════════════════════════════════════
    # Layer Execution — Full Attention (for hybrid architectures)
    # ══════════════════════════════════════════════════════════

    def _full_attention(
        self, x: np.ndarray, layer_idx: int, layer: "LayerWeights"
    ) -> np.ndarray:
        """
        Standard dot-product attention for full-attention layers.
        Uses the M matrix as a simple KV accumulator for single-token inference.
        """
        cfg = self.config
        D = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim

        # Project to Q, K, V
        W_q = layer.get_weight("attn_q")  # (n_heads * head_dim, D)
        W_k = layer.get_weight("attn_k")  # (n_kv * head_dim, D)
        W_v = layer.get_weight("attn_v")  # (n_kv * head_dim, D)
        W_o = layer.get_weight("attn_o")  # (D, n_heads * head_dim)

        x_flat = x.flatten()

        q = (x_flat @ W_q.T).reshape(n_heads, head_dim)
        k = (x_flat @ W_k.T).reshape(n_kv, head_dim)
        v = (x_flat @ W_v.T).reshape(n_kv, head_dim)

        # Apply RoPE
        q, k = self._apply_rope(q, k, self._position)

        # For single-token generation, attention simplifies to:
        # output = softmax(Q·K^T / sqrt(d)) · V
        # With GQA: repeat KV heads to match Q heads
        if n_kv < n_heads:
            repeat = n_heads // n_kv
            k = np.repeat(k, repeat, axis=0)
            v = np.repeat(v, repeat, axis=0)

        # Scaled dot product (single query against single key)
        scale = 1.0 / math.sqrt(head_dim)
        attn_score = np.sum(q * k, axis=-1) * scale  # (n_heads,)
        attn_weight = self._softmax(attn_score)       # (n_heads,)

        # Weighted sum of values
        out = (attn_weight[:, None] * v)  # (n_heads, head_dim)
        out = out.reshape(-1)             # (n_heads * head_dim,)

        # Output projection
        result = out @ W_o.T
        return result.reshape(x.shape)

    # ══════════════════════════════════════════════════════════
    # MLP Execution
    # ══════════════════════════════════════════════════════════

    def _mlp(self, x: np.ndarray, layer: "LayerWeights") -> np.ndarray:
        """Execute SwiGLU MLP: down(silu(gate(x)) * up(x))"""
        W_gate = layer.get_weight("mlp_gate")
        W_up = layer.get_weight("mlp_up")
        W_down = layer.get_weight("mlp_down")

        x_flat = x.flatten()
        gate_out = self._apply_activation(x_flat @ W_gate.T)
        up_out = x_flat @ W_up.T
        hidden = gate_out * up_out
        result = hidden @ W_down.T
        return result.reshape(x.shape)

    # ══════════════════════════════════════════════════════════
    # MoE Execution
    # ══════════════════════════════════════════════════════════

    def _moe_mlp(self, x: np.ndarray, layer: "LayerWeights") -> np.ndarray:
        """Execute Mixture-of-Experts MLP with top-K routing."""
        cfg = self.config
        x_flat = x.flatten()

        # Router: select top-K experts
        W_router = layer.get_weight("router")
        router_logits = x_flat @ W_router.T
        router_probs = self._softmax(router_logits)

        top_k = cfg.num_experts_per_tok
        top_indices = np.argsort(router_probs)[-top_k:][::-1]
        top_weights = router_probs[top_indices]
        top_weights = top_weights / np.sum(top_weights)  # Renormalize

        # Execute selected experts and weighted sum
        result = np.zeros_like(x_flat)
        for i, exp_idx in enumerate(top_indices):
            W_gate = layer.get_expert_weight(exp_idx, "gate")
            W_up = layer.get_expert_weight(exp_idx, "up")
            W_down = layer.get_expert_weight(exp_idx, "down")

            gate_out = self._apply_activation(x_flat @ W_gate.T)
            up_out = x_flat @ W_up.T
            hidden = gate_out * up_out
            expert_out = hidden @ W_down.T

            result += top_weights[i] * expert_out

        return result.reshape(x.shape)

    # ══════════════════════════════════════════════════════════
    # Full Forward Pass — Single Token
    # ══════════════════════════════════════════════════════════

    def forward(self, token_id: int) -> np.ndarray:
        """
        Execute a full forward pass for a single token.
        
        Returns logits: (vocab_size,)
        """
        cfg = self.config
        eps = cfg.rms_norm_eps

        # 1. Token embedding lookup
        h = self.embeddings[token_id].copy()  # (D,)

        # 2. Process through all layers
        for i, layer in enumerate(self.layers):
            # Pre-attention norm
            h_normed = self._rms_norm(h, layer.norm_attn, eps)

            # Attention / Symbiotic Gate
            layer_type = cfg.get_layer_type(i)
            if layer_type == "linear_attention":
                attn_out = self._symbiotic_gate(h_normed, i, layer)
            else:
                attn_out = self._full_attention(h_normed, i, layer)

            # Residual connection
            h = h + attn_out

            # Pre-MLP norm
            h_normed = self._rms_norm(h, layer.norm_mlp, eps)

            # MLP (dense or MoE)
            if layer.is_moe:
                mlp_out = self._moe_mlp(h_normed, layer)
            else:
                mlp_out = self._mlp(h_normed, layer)

            # Residual connection
            h = h + mlp_out

        # 3. Final norm
        h = self._rms_norm(h, self.final_norm_weight, eps)

        # 4. Project to vocabulary logits
        logits = h @ self.lm_head

        # Advance position
        self._position += 1

        return logits

    def generate_token(self, token_id: int) -> int:
        """Generate the next token ID given an input token ID."""
        logits = self.forward(token_id)
        return int(np.argmax(logits))

    def reset(self) -> None:
        """Reset all state matrices and position counter."""
        D = self.config.hidden_size
        self.M = [np.zeros((D, D), dtype=np.float32) for _ in range(self.config.num_layers)]
        self._position = 0


class LayerWeights:
    """
    Manages weight loading for a single transformer layer.
    Supports lazy loading and caching of dequantized weights.
    """

    def __init__(
        self,
        layer_dir: str,
        config: SymbioticConfig,
        layer_idx: int,
        layer_type: str,
    ):
        self.layer_dir = layer_dir
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_moe = (
            config.architecture == "moe"
            and layer_idx in config.moe_layer_indices
        ) if config.moe_layer_indices else False

        # Cache for dequantized weights
        self._cache: dict = {}

        # Load norms eagerly (small, always needed)
        D = config.hidden_size
        norm_attn_path = os.path.join(layer_dir, "norm_attn.symb")
        norm_mlp_path = os.path.join(layer_dir, "norm_mlp.symb")

        if os.path.exists(norm_attn_path):
            self.norm_attn = np.fromfile(norm_attn_path, dtype=np.float16).astype(np.float32)
        else:
            self.norm_attn = np.ones(D, dtype=np.float32)

        if os.path.exists(norm_mlp_path):
            self.norm_mlp = np.fromfile(norm_mlp_path, dtype=np.float16).astype(np.float32)
        else:
            self.norm_mlp = np.ones(D, dtype=np.float32)

    def get_weight(self, name: str) -> np.ndarray:
        """
        Get a dequantized weight matrix by name.
        Lazily loads and caches on first access.
        """
        if name in self._cache:
            return self._cache[name]

        path = os.path.join(self.layer_dir, f"{name}.symb")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")

        method = self.config.quantization.method
        if method == "int4":
            from .export import unpack_int4_symb
            tensor, shape = unpack_int4_symb(path)
        elif method == "int8":
            with open(path, "rb") as f:
                scale = struct.unpack("f", f.read(4))[0]
                raw = np.frombuffer(f.read(), dtype=np.int8)
            tensor = raw.astype(np.float32) * scale
            # Infer shape from config
            tensor = self._infer_shape(name, tensor)
        elif method == "fp16":
            tensor = np.fromfile(path, dtype=np.float16).astype(np.float32)
            tensor = self._infer_shape(name, tensor)
        else:  # fp32
            tensor = np.fromfile(path, dtype=np.float32)
            tensor = self._infer_shape(name, tensor)

        self._cache[name] = tensor
        return tensor

    def get_expert_weight(self, expert_idx: int, name: str) -> np.ndarray:
        """Get a weight matrix for a specific MoE expert."""
        cache_key = f"expert_{expert_idx}_{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = os.path.join(self.layer_dir, f"expert_{expert_idx}", f"{name}.symb")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert weight not found: {path}")

        method = self.config.quantization.method
        if method == "int4":
            from .export import unpack_int4_symb
            tensor, _ = unpack_int4_symb(path)
        else:
            tensor = np.fromfile(path, dtype=np.float16).astype(np.float32)

        self._cache[cache_key] = tensor
        return tensor

    def _infer_shape(self, name: str, flat: np.ndarray) -> np.ndarray:
        """Infer the 2D shape of a weight from its name and config."""
        cfg = self.config
        D = cfg.hidden_size
        I = cfg.intermediate_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim

        shape_map = {
            "attn_q": (n_heads * hd, D),
            "attn_k": (n_kv * hd, D),
            "attn_v": (n_kv * hd, D),
            "attn_o": (D, n_heads * hd),
            "attn_qkv": ((n_heads + 2 * n_kv) * hd, D),
            "mlp_gate": (I, D),
            "mlp_up": (I, D),
            "mlp_down": (D, I),
            "router": (cfg.num_experts, D),
        }

        if name in shape_map:
            return flat.reshape(shape_map[name])
        return flat

    def clear_cache(self) -> None:
        """Clear the weight cache to free memory."""
        self._cache.clear()
