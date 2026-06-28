"""
SYNAXIM: Weight Converter — HuggingFace/Safetensors → .symb Format
===================================================================
Converts any standard model checkpoint into the GRRN proprietary .symb
binary format with INT4 bitpacking.

Supports:
  - HuggingFace model IDs (auto-downloads)
  - Local safetensors directories
  - Dense architectures (LLaMA, Qwen, Mistral, Phi, Gemma, etc.)
  - MoE architectures (Mixtral, DeepSeek-V2/V3, DBRX, etc.)

Usage:
    from grrn_inference import SymbioticConverter
    converter = SymbioticConverter()
    converter.convert("meta-llama/Llama-3.1-8B", "./llama-symb", quantize="int4")
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, Optional, Literal, Any

import numpy as np
from tqdm import tqdm

from .config import SymbioticConfig, QuantizationConfig


# ── Weight Name Mapping ──
# Maps common HuggingFace weight name patterns to our .symb file names
WEIGHT_MAP = {
    # Embeddings
    "model.embed_tokens.weight": "embeddings",
    "embed_tokens.weight": "embeddings",
    # LM Head
    "lm_head.weight": "lm_head",
    # Per-layer attention
    "self_attn.q_proj.weight": "attn_q",
    "self_attn.k_proj.weight": "attn_k",
    "self_attn.v_proj.weight": "attn_v",
    "self_attn.o_proj.weight": "attn_o",
    # Attention biases (Qwen, etc.)
    "self_attn.q_proj.bias": "attn_q_bias",
    "self_attn.k_proj.bias": "attn_k_bias",
    "self_attn.v_proj.bias": "attn_v_bias",
    # Fused QKV (some models)
    "self_attn.qkv_proj.weight": "attn_qkv",
    # Per-layer MLP (dense)
    "mlp.gate_proj.weight": "mlp_gate",
    "mlp.up_proj.weight": "mlp_up",
    "mlp.down_proj.weight": "mlp_down",
    # Per-layer norms
    "input_layernorm.weight": "norm_attn",
    "post_attention_layernorm.weight": "norm_mlp",
    # Final norm
    "model.norm.weight": "final_norm",
    "norm.weight": "final_norm",
    # MoE router
    "block_sparse_moe.gate.weight": "router",
    "mlp.gate.weight": "router",
    # MoE expert MLPs
    "block_sparse_moe.experts": "expert",
}


class SymbioticConverter:
    """
    Converts standard model checkpoints to GRRN .symb format.
    
    Example:
        converter = SymbioticConverter()
        config = converter.convert(
            source="meta-llama/Llama-3.1-8B",
            output_dir="./llama-symb",
            quantize="int4"
        )
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def convert(
        self,
        source: str,
        output_dir: str,
        quantize: Literal["int4", "int8", "fp16", "fp32"] = "int4",
        architecture: Literal["auto", "dense", "moe"] = "auto",
        group_size: int = 128,
    ) -> SymbioticConfig:
        """
        Convert a model to .symb format.

        Args:
            source: HuggingFace model ID or local path to safetensors
            output_dir: Directory to write .symb files
            quantize: Quantization level for weight packing
            architecture: Model architecture type (auto-detects if "auto")
            group_size: Quantization group size for INT4/INT8

        Returns:
            SymbioticConfig describing the converted model
        """
        self._log(f"Converting: {source}")
        self._log(f"Output: {output_dir}")
        self._log(f"Quantization: {quantize}, group_size={group_size}")

        # Step 1: Resolve source to a local directory with safetensors
        model_dir = self._resolve_source(source)

        # Step 2: Read the original config.json
        hf_config = self._read_hf_config(model_dir)

        # Step 3: Build SymbioticConfig from HF config
        config = self._build_config(hf_config, source, quantize, group_size, architecture)

        # Step 4: Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        for i in range(config.num_layers):
            layer_dir = os.path.join(output_dir, config.get_layer_dir(i))
            os.makedirs(layer_dir, exist_ok=True)

        # Step 5: Copy tokenizer files
        self._copy_tokenizer(model_dir, output_dir, config)

        # Step 6: Convert and pack all weights
        self._convert_weights(model_dir, output_dir, config)

        # Step 7: Save config
        config_path = os.path.join(output_dir, "config.symb.json")
        config.save(config_path)
        self._log(f"Config saved: {config_path}")

        self._log("Conversion complete!")
        return config

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SYNAXIM-Convert] {msg}")

    @staticmethod
    def _load_safetensors(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load a safetensors file into a dict of NumPy float32 arrays.
        Handles bfloat16 (unsupported by NumPy) by falling back to PyTorch.
        """
        from safetensors import safe_open
        try:
            result = {}
            with safe_open(filepath, framework="numpy") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    result[key] = t.astype(np.float32) if t.dtype != np.float32 else t
            return result
        except TypeError:
            # bfloat16 → use PyTorch to load and convert
            try:
                import torch
                from safetensors.torch import load_file as torch_load_file
                torch_dict = torch_load_file(filepath)
                return {k: v.float().numpy() for k, v in torch_dict.items()}
            except ImportError:
                raise RuntimeError(
                    "Model uses bfloat16 weights which NumPy cannot read directly. "
                    "Install PyTorch to enable conversion: pip install torch"
                )

    @staticmethod
    def _iter_safetensors(filepath: str):
        """
        Iterate over tensors in a safetensors file ONE AT A TIME.
        Yields (key, tensor_as_float32) pairs.
        
        This is memory-efficient: only one tensor is in RAM at a time.
        For bfloat16 models, uses PyTorch to convert each tensor individually.
        """
        from safetensors import safe_open
        try:
            with safe_open(filepath, framework="numpy") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    yield key, t.astype(np.float32) if t.dtype != np.float32 else t
        except TypeError:
            # bfloat16 — stream via PyTorch one tensor at a time
            import torch
            with safe_open(filepath, framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key).float().numpy()
                    yield key, t

    def _resolve_source(self, source: str) -> str:
        """Resolve a HF model ID or local path to a directory."""
        if os.path.isdir(source):
            return source

        # Try to download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
            self._log(f"Downloading from HuggingFace: {source}")
            local_dir = snapshot_download(
                repo_id=source,
                allow_patterns=["*.safetensors", "*.json", "*.model",
                                "*.txt", "tokenizer*", "*.tiktoken"],
            )
            return local_dir
        except ImportError:
            raise RuntimeError(
                f"Source '{source}' is not a local directory and huggingface_hub "
                "is not installed. Install it with: pip install huggingface_hub"
            )

    def _read_hf_config(self, model_dir: str) -> Dict[str, Any]:
        """Read config.json from a HF model directory."""
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json found in {model_dir}")
        with open(config_path) as f:
            return json.load(f)

    def _build_config(
        self,
        hf_config: Dict[str, Any],
        source: str,
        quantize: str,
        group_size: int,
        architecture: str,
    ) -> SymbioticConfig:
        """Build a SymbioticConfig from a HuggingFace config.json."""
        # Auto-detect architecture
        model_type = hf_config.get("model_type", "")
        is_moe = architecture == "moe" or (
            architecture == "auto" and any(k in hf_config for k in [
                "num_local_experts", "num_experts", "n_routed_experts"
            ])
        )

        hidden = hf_config.get("hidden_size", hf_config.get("d_model", 4096))
        n_heads = hf_config.get("num_attention_heads", hf_config.get("n_head", 32))
        n_kv = hf_config.get("num_key_value_heads", n_heads)
        n_layers = hf_config.get("num_hidden_layers", hf_config.get("n_layer", 32))

        # Detect layer types (e.g., Qwen3.5 has mixed linear + full attention)
        layer_types = []
        if "layer_types" in hf_config:
            # Direct layer_types list in config (Qwen3.5 style)
            raw = hf_config["layer_types"]
            for lt in raw:
                if lt in (0, "linear", "linear_attention"):
                    layer_types.append("linear_attention")
                else:
                    layer_types.append("full_attention")
        else:
            # Standard transformer — all layers are full attention
            layer_types = ["full_attention"] * n_layers
        
        config = SymbioticConfig(
            model_name=source.split("/")[-1] if "/" in source else source,
            model_type="moe" if is_moe else "dense",
            source_model=source,
            architecture="moe" if is_moe else "dense",
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hidden,
            intermediate_size=hf_config.get("intermediate_size", hidden * 4),
            num_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=hf_config.get("head_dim", hidden // n_heads),
            max_position_embeddings=hf_config.get("max_position_embeddings", 131072),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rms_norm_eps=hf_config.get("rms_norm_eps", hf_config.get("layer_norm_epsilon", 1e-6)),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
            layer_types=layer_types,
            hidden_act=hf_config.get("hidden_act", "silu"),
            quantization=QuantizationConfig(
                method=quantize,
                group_size=group_size,
                symmetric=True,
            ),
        )

        if is_moe:
            config.num_experts = hf_config.get(
                "num_local_experts", hf_config.get("num_experts",
                hf_config.get("n_routed_experts", 8))
            )
            config.num_experts_per_tok = hf_config.get(
                "num_experts_per_tok", hf_config.get("top_k", 2)
            )

        self._log(f"Architecture: {config.architecture}")
        self._log(f"Hidden: {config.hidden_size}, Layers: {config.num_layers}, "
                   f"Heads: {config.num_attention_heads}/{config.num_key_value_heads}")
        self._log(f"Vocab: {config.vocab_size}, Intermediate: {config.intermediate_size}")
        if layer_types:
            n_lin = sum(1 for lt in layer_types if lt == "linear_attention")
            n_full = sum(1 for lt in layer_types if lt == "full_attention")
            self._log(f"Layer types: {n_lin} linear + {n_full} full attention")
        if is_moe:
            self._log(f"MoE: {config.num_experts} experts, top-{config.num_experts_per_tok}")

        return config

    def _copy_tokenizer(self, model_dir: str, output_dir: str, config: SymbioticConfig) -> None:
        """Copy tokenizer files to the output directory."""
        tok_dir = os.path.join(output_dir, config.tokenizer_dir)
        os.makedirs(tok_dir, exist_ok=True)

        tokenizer_files = [
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json",
            "vocab.json", "merges.txt",
        ]
        copied = 0
        for fname in tokenizer_files:
            src = os.path.join(model_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tok_dir, fname))
                copied += 1
        
        # Also copy any .tiktoken files
        for f in Path(model_dir).glob("*.tiktoken"):
            shutil.copy2(str(f), os.path.join(tok_dir, f.name))
            copied += 1

        self._log(f"Tokenizer: {copied} files copied")

    def _convert_weights(
        self, model_dir: str, output_dir: str, config: SymbioticConfig
    ) -> None:
        """Convert all weights from safetensors to .symb format."""
        # Find all safetensors files
        st_files = sorted(Path(model_dir).glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        self._log(f"Found {len(st_files)} safetensors files")

        quant_method = config.quantization.method
        group_size = config.quantization.group_size
        total_params = 0
        total_packed = 0

        for st_file in tqdm(st_files, desc="Converting shards", disable=not self.verbose):
            # Stream one tensor at a time to minimize RAM usage
            for key, tensor in self._iter_safetensors(str(st_file)):
                    # Ensure float32 for quantization
                    if tensor.dtype != np.float32:
                        tensor = tensor.astype(np.float32)
                    total_params += tensor.size

                    # Determine output path
                    out_path = self._map_weight_path(key, output_dir, config)
                    if out_path is None:
                        self._log(f"  [SKIP] Unknown weight: {key}")
                        continue

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    # Determine packing method:
                    # - 1D tensors (biases, norms): always FP16 (small)
                    # - final_norm: always FP16 (tiny)
                    # - Embeddings/lm_head: use selected quant (can be large)
                    is_1d = (tensor.ndim == 1)
                    basename = os.path.basename(out_path).replace(".symb", "")
                    is_norm = basename in ("final_norm",)
                    pack_method = "fp16" if (is_1d or is_norm) else quant_method

                    # Pack the weight based on quantization method
                    packed_size = self._pack_weight(
                        tensor, out_path, pack_method, group_size
                    )
                    total_packed += packed_size

        ratio = total_params * 2 / max(total_packed, 1)  # vs FP16 baseline
        self._log(f"Total parameters: {total_params:,}")
        self._log(f"Packed size: {total_packed:,} bytes")
        self._log(f"Compression ratio: {ratio:.1f}x vs FP16")

    def _map_weight_path(
        self, key: str, output_dir: str, config: SymbioticConfig
    ) -> Optional[str]:
        """Map a HF weight name to a .symb file path."""
        # Global weights (embeddings, lm_head, final_norm)
        # Only match if the key is NOT a per-layer weight (no ".layers." in path)
        if ".layers." not in key:
            for pattern, name in WEIGHT_MAP.items():
                if key == pattern or key.endswith(pattern):
                    if name in ("embeddings", "lm_head", "final_norm"):
                        return os.path.join(output_dir, f"{name}.symb")

        # Per-layer weights: extract layer index
        # Pattern: model.layers.{N}.{submodule}.weight
        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_idx is None:
            return None

        layer_dir = os.path.join(output_dir, config.get_layer_dir(layer_idx))

        # Match the sub-weight name
        suffix = ".".join(parts[parts.index(str(layer_idx)) + 1:])
        for pattern, name in WEIGHT_MAP.items():
            if suffix == pattern or suffix.endswith(pattern):
                if name == "expert":
                    # MoE expert: extract expert index
                    # Pattern: ...experts.{E}.{gate/up/down}_proj.weight
                    exp_idx = None
                    for j, pp in enumerate(parts):
                        if pp == "experts" and j + 1 < len(parts):
                            try:
                                exp_idx = int(parts[j + 1])
                                break
                            except ValueError:
                                continue
                    if exp_idx is not None:
                        exp_dir = os.path.join(layer_dir, f"expert_{exp_idx}")
                        os.makedirs(exp_dir, exist_ok=True)
                        # Figure out which expert sub-weight
                        if "gate_proj" in key:
                            return os.path.join(exp_dir, "gate.symb")
                        elif "up_proj" in key:
                            return os.path.join(exp_dir, "up.symb")
                        elif "down_proj" in key:
                            return os.path.join(exp_dir, "down.symb")
                    return None
                return os.path.join(layer_dir, f"{name}.symb")

        return None

    def _pack_weight(
        self,
        tensor: np.ndarray,
        out_path: str,
        method: str,
        group_size: int,
    ) -> int:
        """Pack a tensor into .symb binary format and return bytes written."""
        if method == "fp32":
            data = tensor.astype(np.float32)
            data.tofile(out_path)
            return data.nbytes

        elif method == "fp16":
            data = tensor.astype(np.float16)
            data.tofile(out_path)
            return data.nbytes

        elif method == "int8":
            flat = tensor.astype(np.float32).flatten()
            scale = np.max(np.abs(flat)) / 127.0 if np.max(np.abs(flat)) > 0 else 1.0
            quantized = np.clip(np.round(flat / scale), -128, 127).astype(np.int8)
            # Write scale + quantized data
            with open(out_path, "wb") as f:
                f.write(struct.pack("f", scale))
                quantized.tofile(f)
            return 4 + quantized.nbytes

        elif method == "int4":
            return self._pack_int4(tensor, out_path, group_size)

        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def _pack_int4(
        self, tensor: np.ndarray, out_path: str, group_size: int
    ) -> int:
        """
        Pack a tensor into INT4 .symb format with per-group scale factors.
        
        Binary layout:
            [4 bytes: num_groups (uint32)]
            [4 bytes: group_size (uint32)]
            [4 bytes: original_numel (uint32)]
            [8 bytes: original shape dim0 (uint32) + dim1 (uint32)]
            [num_groups * 4 bytes: per-group FP32 scale factors]
            [ceil(numel/2) bytes: packed INT4 pairs]
        """
        flat = tensor.astype(np.float32).flatten()
        numel = flat.size

        # Pad to multiple of group_size
        padded_size = ((numel + group_size - 1) // group_size) * group_size
        if padded_size > numel:
            flat = np.concatenate([flat, np.zeros(padded_size - numel, dtype=np.float32)])

        num_groups = padded_size // group_size
        groups = flat.reshape(num_groups, group_size)

        # Compute per-group scales
        scales = np.max(np.abs(groups), axis=1)
        scales = np.where(scales > 0, scales / 7.0, 1.0).astype(np.float32)

        # Quantize to signed INT4 [-8, 7]
        quantized = np.clip(
            np.round(groups / scales[:, None]), -8, 7
        ).astype(np.int8)

        # Map to unsigned [0, 15] for bitpacking
        unsigned = (quantized + 8).astype(np.uint8).flatten()

        # Pack pairs of 4-bit values into bytes: upper | lower
        if unsigned.size % 2 != 0:
            unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.uint8)])
        packed = (unsigned[0::2] << 4) | unsigned[1::2]

        # Write to file
        shape = tensor.shape
        with open(out_path, "wb") as f:
            f.write(struct.pack("III", num_groups, group_size, numel))
            # Write original shape (up to 2 dims)
            if len(shape) >= 2:
                f.write(struct.pack("II", shape[0], shape[1]))
            elif len(shape) == 1:
                f.write(struct.pack("II", shape[0], 1))
            else:
                f.write(struct.pack("II", 1, 1))
            # Write scales
            scales.tofile(f)
            # Write packed data
            packed.tofile(f)

        return 20 + scales.nbytes + packed.nbytes  # header + scales + data


def unpack_int4_symb(path: str) -> tuple:
    """
    Unpack an INT4 .symb file back to float32.
    
    Returns:
        (tensor: np.ndarray, shape: tuple)
    """
    with open(path, "rb") as f:
        num_groups, group_size, numel = struct.unpack("III", f.read(12))
        dim0, dim1 = struct.unpack("II", f.read(8))
        scales = np.frombuffer(f.read(num_groups * 4), dtype=np.float32)
        packed = np.frombuffer(f.read(), dtype=np.uint8)

    # Unpack pairs
    upper = (packed >> 4).astype(np.int8)
    lower = (packed & 0x0F).astype(np.int8)
    unpacked = np.empty(packed.size * 2, dtype=np.int8)
    unpacked[0::2] = upper
    unpacked[1::2] = lower

    # Re-center from unsigned [0,15] to signed [-8,7]
    unpacked = unpacked - 8

    # Trim to actual size
    unpacked = unpacked[:num_groups * group_size]

    # Dequantize per group
    groups = unpacked.reshape(num_groups, group_size).astype(np.float32)
    dequantized = groups * scales[:, None]

    # Flatten and trim to original numel
    result = dequantized.flatten()[:numel]

    shape = (dim0, dim1) if dim1 > 1 else (dim0,)
    return result.reshape(shape), shape
