"""
SYNAXIM: Dense Model Architecture Handler
==========================================
Handles inference for standard dense transformer models where
every token passes through every layer sequentially.

Supported model families:
  - LLaMA / LLaMA-2 / LLaMA-3
  - Qwen / Qwen2 / Qwen3.5 (including mixed attention types)
  - Mistral
  - Phi / Phi-3
  - Gemma / Gemma-2
  - Any model with sequential dense layers
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from ..config import SymbioticConfig


class DenseModelHandler:
    """
    Architecture handler for dense (non-MoE) models.
    
    Provides architecture-specific weight mapping and configuration
    for the core SymbioticStateEngine.
    """

    # Model family → weight name patterns
    WEIGHT_PATTERNS: Dict[str, Dict[str, str]] = {
        "llama": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "mlp_gate": "mlp.gate_proj.weight",
            "mlp_up": "mlp.up_proj.weight",
            "mlp_down": "mlp.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "qwen": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "mlp_gate": "mlp.gate_proj.weight",
            "mlp_up": "mlp.up_proj.weight",
            "mlp_down": "mlp.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "mistral": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "mlp_gate": "mlp.gate_proj.weight",
            "mlp_up": "mlp.up_proj.weight",
            "mlp_down": "mlp.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "phi": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.dense.weight",
            "mlp_gate": "mlp.gate_up_proj.weight",
            "mlp_down": "mlp.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "gemma": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "model.embed_tokens.weight",  # Gemma ties embeddings
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "mlp_gate": "mlp.gate_proj.weight",
            "mlp_up": "mlp.up_proj.weight",
            "mlp_down": "mlp.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
    }

    def __init__(self, config: SymbioticConfig):
        self.config = config
        self.family = self._detect_family()

    def _detect_family(self) -> str:
        """Detect model family from config."""
        name = self.config.source_model.lower()
        if "qwen" in name:
            return "qwen"
        elif "mistral" in name:
            return "mistral"
        elif "phi" in name:
            return "phi"
        elif "gemma" in name:
            return "gemma"
        else:
            return "llama"  # Default fallback

    def get_weight_pattern(self) -> Dict[str, str]:
        """Get the weight name mapping for this model family."""
        return self.WEIGHT_PATTERNS.get(self.family, self.WEIGHT_PATTERNS["llama"])

    def get_layer_execution_order(self) -> List[Dict[str, Any]]:
        """
        Get the execution order for all layers.
        
        Returns a list of dicts describing each layer's configuration:
        [{"index": 0, "type": "linear_attention", "has_moe": False}, ...]
        """
        layers = []
        for i in range(self.config.num_layers):
            layers.append({
                "index": i,
                "type": self.config.get_layer_type(i),
                "has_moe": False,
            })
        return layers

    def describe(self) -> str:
        """Human-readable model description."""
        cfg = self.config
        desc = f"Dense {self.family.upper()} model: {cfg.model_name}\n"
        desc += f"  Layers: {cfg.num_layers}\n"
        desc += f"  Hidden: {cfg.hidden_size}\n"
        desc += f"  Heads: {cfg.num_attention_heads} Q, {cfg.num_key_value_heads} KV\n"
        desc += f"  Vocab: {cfg.vocab_size}\n"
        if cfg.layer_types:
            n_lin = cfg.num_linear_attention_layers
            n_full = cfg.num_full_attention_layers
            desc += f"  Layer types: {n_lin} linear + {n_full} full attention\n"
        desc += f"  Quantization: {cfg.quantization.method}\n"
        return desc
