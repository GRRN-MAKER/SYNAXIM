"""
SYNAXIM: Model Configuration & .symb.json Specification
========================================================
Defines the SymbioticConfig that describes a model's architecture,
dimensions, layer types, and quantization parameters.

The .symb.json file is the single source of truth for any model
converted to the GRRN .symb format.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal


@dataclass
class QuantizationConfig:
    """Quantization parameters for .symb weight packing."""
    method: Literal["int4", "int8", "fp16", "fp32"] = "int4"
    group_size: int = 128
    symmetric: bool = True
    # Per-layer scale factors are stored alongside the packed weights
    # in <layer_dir>/scales.symb (FP32)


@dataclass
class LayerConfig:
    """Configuration for a single transformer layer."""
    index: int = 0
    layer_type: Literal["linear_attention", "full_attention"] = "linear_attention"
    # MoE fields (only used when architecture == "moe")
    is_moe: bool = False
    num_experts: int = 1
    top_k: int = 1


@dataclass
class SymbioticConfig:
    """
    Master configuration for a .symb model checkpoint.
    
    Stored as `config.symb.json` in the model directory.
    This is the only file needed to understand how to load
    and execute a .symb model.
    """
    # ── Identity ──
    model_name: str = "unknown"
    model_type: str = "dense"                # "dense" or "moe"
    source_model: str = ""                   # Original HF model ID
    grrn_version: str = "0.1.0"
    engine_name: str = "SYNAXIM"

    # ── Architecture ──
    architecture: Literal["dense", "moe"] = "dense"
    vocab_size: int = 32000
    hidden_size: int = 4096                  # d_model
    intermediate_size: int = 11008           # MLP intermediate dim
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32            # For GQA (grouped query attention)
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    # ── Layer Types ──
    # Maps layer index to type. If empty, all layers are "linear_attention"
    layer_types: List[str] = field(default_factory=list)
    
    # ── MoE Configuration (only when architecture == "moe") ──
    num_experts: int = 1
    num_experts_per_tok: int = 1
    moe_layer_indices: List[int] = field(default_factory=list)

    # ── Quantization ──
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # ── Activation ──
    hidden_act: str = "silu"                 # silu / gelu / relu

    # ── Symbiotic Engine Parameters ──
    # Controls for the Continuous Associative Memory gate
    gate_alpha: float = 0.5                  # SmoothQuant balancing factor
    gate_scale: float = 0.0039               # INT4 dequantization multiplier
    memory_lambda: float = 0.995             # EMA decay for M matrix

    # ── File Layout ──
    # These are relative paths within the model directory
    embedding_file: str = "embeddings.symb"
    lm_head_file: str = "lm_head.symb"
    layer_dir_pattern: str = "layers/layer_{:02d}"
    tokenizer_dir: str = "tokenizer"

    def save(self, path: str) -> None:
        """Save config to a .symb.json file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SymbioticConfig":
        """Load config from a .symb.json file."""
        with open(path) as f:
            data = json.load(f)
        # Handle nested QuantizationConfig
        if "quantization" in data and isinstance(data["quantization"], dict):
            data["quantization"] = QuantizationConfig(**data["quantization"])
        return cls(**data)

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "SymbioticConfig":
        """Load config from a model directory containing config.symb.json."""
        config_path = os.path.join(model_dir, "config.symb.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config.symb.json found in {model_dir}. "
                f"Convert a model first with: grrn-convert <model_id> {model_dir}"
            )
        return cls.load(config_path)

    @property
    def num_linear_attention_layers(self) -> int:
        if not self.layer_types:
            return self.num_layers
        return sum(1 for lt in self.layer_types if lt == "linear_attention")

    @property
    def num_full_attention_layers(self) -> int:
        if not self.layer_types:
            return 0
        return sum(1 for lt in self.layer_types if lt == "full_attention")

    def get_layer_type(self, layer_idx: int) -> str:
        """Get the attention type for a specific layer."""
        if self.layer_types and layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx]
        return "full_attention"

    def get_layer_dir(self, layer_idx: int) -> str:
        """Get the directory name for a specific layer's weights."""
        return self.layer_dir_pattern.format(layer_idx)


# ── Architecture Presets ──

def preset_llama(size: str = "8b") -> SymbioticConfig:
    """Preset config for LLaMA-style dense models."""
    presets = {
        "7b": dict(hidden_size=4096, intermediate_size=11008, num_layers=32,
                    num_attention_heads=32, num_key_value_heads=32, vocab_size=32000),
        "8b": dict(hidden_size=4096, intermediate_size=14336, num_layers=32,
                    num_attention_heads=32, num_key_value_heads=8, vocab_size=128256),
        "13b": dict(hidden_size=5120, intermediate_size=13824, num_layers=40,
                     num_attention_heads=40, num_key_value_heads=40, vocab_size=32000),
        "70b": dict(hidden_size=8192, intermediate_size=28672, num_layers=80,
                     num_attention_heads=64, num_key_value_heads=8, vocab_size=128256),
    }
    cfg = presets.get(size, presets["8b"])
    cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
    return SymbioticConfig(model_name=f"llama-{size}", model_type="dense", **cfg)


def preset_qwen(size: str = "27b") -> SymbioticConfig:
    """Preset config for Qwen-style models (including Odyssey)."""
    presets = {
        "7b": dict(hidden_size=4096, intermediate_size=11008, num_layers=32,
                    num_attention_heads=32, num_key_value_heads=32, vocab_size=151936),
        "14b": dict(hidden_size=5120, intermediate_size=13696, num_layers=40,
                     num_attention_heads=40, num_key_value_heads=40, vocab_size=151936),
        "27b": dict(hidden_size=5120, intermediate_size=13696, num_layers=64,
                     num_attention_heads=40, num_key_value_heads=8, vocab_size=151936,
                     layer_types=(["linear_attention"] * 48 + ["full_attention"] * 16)),
    }
    cfg = presets.get(size, presets["27b"])
    if isinstance(cfg.get("layer_types"), tuple):
        cfg["layer_types"] = list(cfg["layer_types"])
    cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
    return SymbioticConfig(model_name=f"qwen-{size}", model_type="dense",
                           rope_theta=1000000.0, **cfg)


def preset_mixtral() -> SymbioticConfig:
    """Preset config for Mixtral-style MoE models."""
    return SymbioticConfig(
        model_name="mixtral-8x7b",
        model_type="moe",
        architecture="moe",
        hidden_size=4096,
        intermediate_size=14336,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=32000,
        num_experts=8,
        num_experts_per_tok=2,
    )
