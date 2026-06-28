"""
SYNAXIM: Convert Any HuggingFace Model
=======================================
Supports Dense models (LLaMA, Qwen, Mistral, Phi, Gemma)
and MoE models (Mixtral, DeepSeek, DBRX).
"""

from grrn_inference import SymbioticConverter

converter = SymbioticConverter()

# ── Dense Models ──

# LLaMA 3.1 8B
converter.convert("meta-llama/Llama-3.1-8B", "./llama-8b-symb", quantize="int4")

# Qwen 2.5 7B
converter.convert("Qwen/Qwen2.5-7B", "./qwen-7b-symb", quantize="int4")

# Mistral 7B
converter.convert("mistralai/Mistral-7B-v0.3", "./mistral-7b-symb", quantize="int4")

# ── MoE Models ──

# Mixtral 8x7B
converter.convert("mistralai/Mixtral-8x7B-v0.1", "./mixtral-symb", quantize="int4")

# ── From local directory ──

# If you already downloaded a model:
converter.convert("/path/to/local/model", "./local-symb", quantize="int4")

# ── Different quantization levels ──

# FP16 (no quantization, largest but most accurate)
converter.convert("meta-llama/Llama-3.1-8B", "./llama-fp16", quantize="fp16")

# INT8 (2x smaller than FP16)
converter.convert("meta-llama/Llama-3.1-8B", "./llama-int8", quantize="int8")
