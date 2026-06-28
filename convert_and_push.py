#!/usr/bin/env python3
"""
SYNAXIM: Convert Llama-3.2-1B to .symb format and push to HuggingFace
======================================================================
Run this on Google Colab (free tier) or any machine with 8+ GB RAM and 5+ GB disk.

Usage (Colab):
    !pip install grrn-inference safetensors huggingface-hub torch tqdm numpy
    !python convert_and_push.py

Usage (local):
    pip install grrn-inference safetensors huggingface-hub torch tqdm numpy
    python convert_and_push.py
"""

import os
import sys

# ── Configuration ──
SOURCE_MODEL = "unsloth/Llama-3.2-1B"       # Source model (no gating)
OUTPUT_DIR = "./llama-1b-symb"               # Local conversion output
HF_REPO = "GRRNNOB/SYNAXIM"                 # Target HF repo
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Set via: export HF_TOKEN=hf_xxx
QUANTIZE = "int4"
GROUP_SIZE = 128


def main():
    print("=" * 60)
    print("  SYNAXIM — Convert Llama-3.2-1B to .symb format")
    print("=" * 60)
    print(f"  Source:  {SOURCE_MODEL}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Target:  {HF_REPO}")
    print(f"  Quant:   {QUANTIZE}, group_size={GROUP_SIZE}")
    print("=" * 60)
    print()

    # Step 1: Install grrn-inference if not available
    try:
        import grrn_inference
    except ImportError:
        print("[1/4] Installing grrn-inference...")
        os.system(f"{sys.executable} -m pip install -q git+https://github.com/GRRN-MAKER/SYNAXIM.git")
        import grrn_inference

    from grrn_inference import SymbioticConverter

    # Step 2: Convert
    print("[2/4] Converting model to .symb format...")
    converter = SymbioticConverter()
    config = converter.convert(
        source=SOURCE_MODEL,
        output_dir=OUTPUT_DIR,
        quantize=QUANTIZE,
        group_size=GROUP_SIZE,
    )
    print(f"  ✓ Converted: {config.num_layers} layers, D={config.hidden_size}, V={config.vocab_size}")
    print(f"  ✓ GQA: {config.num_attention_heads}Q / {config.num_key_value_heads}KV")
    print(f"  ✓ Tied embeddings: {config.tie_word_embeddings}")
    print()

    # Step 3: Calculate total size
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
            file_count += 1
    print(f"[3/4] Converted model: {file_count} files, {total_size / 1e6:.1f} MB total")
    print()

    # Step 4: Push to HuggingFace
    print(f"[4/4] Pushing to {HF_REPO}...")
    from huggingface_hub import HfApi, login

    login(token=HF_TOKEN)
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"  Repo creation note: {e}")

    # Upload the entire output directory
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO,
        commit_message=f"SYNAXIM: Llama-3.2-1B converted to .symb (INT4, group_size={GROUP_SIZE})",
    )

    # Also upload a README
    readme_content = f"""---
license: llama3.2
tags:
  - synaxim
  - grrn-inference
  - int4
  - symbiotic-gate
  - framework-free
---

# Llama-3.2-1B — SYNAXIM .symb Format

Converted from [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B) 
to the SYNAXIM proprietary `.symb` binary format with INT4 quantization.

## Model Details

| Property | Value |
|----------|-------|
| **Source** | `unsloth/Llama-3.2-1B` (Meta Llama 3.2) |
| **Architecture** | LlamaForCausalLM (Dense, GQA) |
| **Parameters** | 1.24B |
| **Hidden Size** | {config.hidden_size} |
| **Layers** | {config.num_layers} |
| **Attention** | {config.num_attention_heads}Q / {config.num_key_value_heads}KV (GQA 4:1) |
| **Head Dim** | {config.head_dim} |
| **Vocab** | {config.vocab_size:,} |
| **Quantization** | INT4 (per-group, group_size={GROUP_SIZE}) |
| **Tied Embeddings** | {config.tie_word_embeddings} |
| **Engine** | SYNAXIM v0.1.0 |
| **Format** | `.symb` (proprietary binary) |

## Usage

```python
pip install grrn-inference
```

```python
from grrn_inference import GRRNModel

model = GRRNModel.from_pretrained("./llama-1b-symb", device="cpu")
result = model.generate("The meaning of life is", max_tokens=50, temperature=0.7)
print(result.text)
```

## What is SYNAXIM?

SYNAXIM replaces the standard Transformer inference paradigm with a 
**Continuous Associative State Machine** — a persistent memory matrix `M` 
that maintains O(1) fixed-size state regardless of sequence length.

- **No KV Cache** — O(1) memory footprint
- **No PyTorch required** — pure NumPy inference  
- **Proprietary `.symb` format** — INT4 bitpacked weights

GitHub: [GRRN-MAKER/SYNAXIM](https://github.com/GRRN-MAKER/SYNAXIM)
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        commit_message="Add README",
    )

    print(f"  ✓ Pushed to https://huggingface.co/{HF_REPO}")
    print()
    print("=" * 60)
    print("  DONE! Download with:")
    print(f"  huggingface-cli download {HF_REPO} --local-dir ./llama-1b-symb")
    print("=" * 60)


if __name__ == "__main__":
    main()
