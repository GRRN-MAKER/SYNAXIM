"""
SYNAXIM: Mixture-of-Experts (MoE/MoW) Architecture Handler
===========================================================
Handles inference for sparse MoE models where a router selects
top-K experts per token.

Supported model families:
  - Mixtral (8x7B, 8x22B)
  - DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1
  - DBRX
  - Qwen-MoE
  - Any model with router + expert MLP blocks
"""

from __future__ import annotations

from typing import Dict, List, Any

from ..config import SymbioticConfig


class MoEModelHandler:
    """
    Architecture handler for Mixture-of-Experts models.
    
    Key differences from dense models:
    1. Some/all MLP blocks are replaced with router + N expert MLPs
    2. Router selects top-K experts per token
    3. Expert outputs are weighted-summed by router probabilities
    4. Attention layers remain the same as dense models
    """

    WEIGHT_PATTERNS: Dict[str, Dict[str, str]] = {
        "mixtral": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "router": "block_sparse_moe.gate.weight",
            "expert_gate": "block_sparse_moe.experts.{}.w1.weight",
            "expert_up": "block_sparse_moe.experts.{}.w3.weight",
            "expert_down": "block_sparse_moe.experts.{}.w2.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "deepseek": {
            "embed": "model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.norm.weight",
            "layer_prefix": "model.layers.{}.{}",
            "attn_q": "self_attn.q_proj.weight",
            "attn_k": "self_attn.k_proj.weight",
            "attn_v": "self_attn.v_proj.weight",
            "attn_o": "self_attn.o_proj.weight",
            "router": "mlp.gate.weight",
            "expert_gate": "mlp.experts.{}.gate_proj.weight",
            "expert_up": "mlp.experts.{}.up_proj.weight",
            "expert_down": "mlp.experts.{}.down_proj.weight",
            "shared_expert_gate": "mlp.shared_experts.gate_proj.weight",
            "shared_expert_up": "mlp.shared_experts.up_proj.weight",
            "shared_expert_down": "mlp.shared_experts.down_proj.weight",
            "norm_attn": "input_layernorm.weight",
            "norm_mlp": "post_attention_layernorm.weight",
        },
        "dbrx": {
            "embed": "transformer.wte.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "transformer.norm_f.weight",
            "layer_prefix": "transformer.blocks.{}.{}",
            "attn_q": "norm_attn_norm.attn.Wqkv.weight",
            "router": "ffn.router.layer.weight",
            "expert_gate": "ffn.experts.mlp.w1",
            "expert_up": "ffn.experts.mlp.v1",
            "expert_down": "ffn.experts.mlp.w2",
            "norm_attn": "norm_attn_norm.norm_1.weight",
            "norm_mlp": "norm_attn_norm.norm_2.weight",
        },
    }

    def __init__(self, config: SymbioticConfig):
        self.config = config
        self.family = self._detect_family()

    def _detect_family(self) -> str:
        """Detect MoE model family from config."""
        name = self.config.source_model.lower()
        if "deepseek" in name:
            return "deepseek"
        elif "dbrx" in name:
            return "dbrx"
        elif "qwen" in name and "moe" in name:
            return "mixtral"  # Qwen-MoE uses similar patterns
        else:
            return "mixtral"  # Default MoE pattern

    def get_weight_pattern(self) -> Dict[str, str]:
        """Get the weight name mapping for this MoE family."""
        return self.WEIGHT_PATTERNS.get(self.family, self.WEIGHT_PATTERNS["mixtral"])

    def get_layer_execution_order(self) -> List[Dict[str, Any]]:
        """
        Get the execution order for all layers.
        
        MoE layers have `has_moe=True` and include expert routing info.
        """
        cfg = self.config
        layers = []
        for i in range(cfg.num_layers):
            is_moe = True  # In most MoE models, all layers are MoE
            if cfg.moe_layer_indices:
                is_moe = i in cfg.moe_layer_indices

            layers.append({
                "index": i,
                "type": cfg.get_layer_type(i),
                "has_moe": is_moe,
                "num_experts": cfg.num_experts if is_moe else 1,
                "top_k": cfg.num_experts_per_tok if is_moe else 1,
            })
        return layers

    def get_expert_count(self, layer_idx: int) -> int:
        """Get number of experts for a specific layer."""
        if self.config.moe_layer_indices and layer_idx not in self.config.moe_layer_indices:
            return 1
        return self.config.num_experts

    def has_shared_experts(self) -> bool:
        """Check if this MoE model has shared experts (DeepSeek style)."""
        return self.family == "deepseek"

    def describe(self) -> str:
        """Human-readable model description."""
        cfg = self.config
        desc = f"MoE {self.family.upper()} model: {cfg.model_name}\n"
        desc += f"  Layers: {cfg.num_layers}\n"
        desc += f"  Hidden: {cfg.hidden_size}\n"
        desc += f"  Experts: {cfg.num_experts} total, top-{cfg.num_experts_per_tok} active\n"
        desc += f"  Heads: {cfg.num_attention_heads} Q, {cfg.num_key_value_heads} KV\n"
        desc += f"  Vocab: {cfg.vocab_size}\n"
        desc += f"  Quantization: {cfg.quantization.method}\n"
        if self.has_shared_experts():
            desc += f"  Shared experts: Yes (DeepSeek style)\n"
        return desc
