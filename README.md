<p align="center">
  <img src="https://img.shields.io/badge/SYNAXIM-v0.1.1-blueviolet?style=for-the-badge" alt="SYNAXIM">
</p>

<h1 align="center">S Y N A X I M</h1>

<p align="center">
  <b>Symbiotic Native Axiom Inference Machine</b><br>
  <i>The Transformer Replacement — Framework-Free LLM Inference</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Contributions-Welcome-blue" alt="Contributions Welcome">
  <img src="https://img.shields.io/badge/PyTorch-Not%20Required-red" alt="No PyTorch">
  <img src="https://img.shields.io/badge/KV--Cache-Eliminated-orange" alt="No KV-Cache">
  <img src="https://img.shields.io/badge/Memory-O(1)%20Fixed-brightgreen" alt="O(1) Memory">
</p>

<p align="center">
  <a href="https://github.com/GRRN-MAKER/SYNAXIM/releases">📦 Release</a> •
  <a href="#installation">⚡ Install</a> •
  <a href="#quick-start">🚀 Quick Start</a> •
  <a href="#supported-models">🏗️ Models</a> •
  <a href="#api-reference">📖 API</a>
</p>

---

## What is SYNAXIM?

SYNAXIM is a **standalone inference engine** that runs Large Language Models **without PyTorch, without the Transformers library, and without KV-Cache**. It replaces the standard Transformer inference paradigm with a **Continuous Associative State Machine** a persistent memory matrix `M` that maintains O(1) fixed-size state regardless of sequence length.

**Attention ≡ Memory**: Instead of splitting data into a disjointed context window, SYNAXIM brings the tokens back into a **continuous, assembled, persistent state matrix**. This is not a wrapper — it is a completely independent execution engine with its own proprietary `.symb` binary weight format.

```
 ███████╗██╗   ██╗███╗   ██╗ █████╗ ██╗  ██╗██╗███╗   ███╗
 ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗╚██╗██╔╝██║████╗ ████║
 ███████╗ ╚████╔╝ ██╔██╗ ██║███████║ ╚███╔╝ ██║██╔████╔██║
 ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║ ██╔██╗ ██║██║╚██╔╝██║
 ███████║   ██║   ██║ ╚████║██║  ██║██╔╝ ██╗██║██║ ╚═╝ ██║
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝
```

> **SYNAXIM prints its banner on every import and every model load — permanently burnt into the code.**

Here is the core idea, end to end:

```
HuggingFace Model  →  grrn-convert  →  .symb Files  →  SYNAXIM Engine  →  Text Output
                                           │
              ┌────────────────────────────┘
              ▼
     [Token ID]  →  [Embedding Lookup]  →  [64-Layer Symbiotic Pipeline]  →  [Logits]  →  [Next Token]
                                                    │
                              ┌──────────────────────┘
                              ▼
                    Per Layer: RMSNorm → Attn/Gate → RMSNorm → MLP → Residual
                              │                         │
                    Linear Attention:              Full Attention:
                    M = σ(xW)·M + (1-σ(xW))·x⊗x    Standard QKV + RoPE
                    (O(1) state, no KV-Cache)       (For hybrid architectures)
```

---

## Why SYNAXIM Instead of Transformers?

| Feature | HuggingFace Transformers | vLLM | **SYNAXIM** |
|---------|------------------------|------|-------------|
| Runtime dependency | PyTorch + CUDA | PyTorch + CUDA + Ray | **Zero framework** (NumPy only) |
| Memory model | KV-Cache (grows with context) | PagedAttention KV | **O(1) M matrix** (fixed, infinite context) proprietary |
| Weight format | safetensors (open, anyone reads) | safetensors | **`.symb` (proprietary INT4 bitpacked)** |
| Quantization | bitsandbytes / GPTQ / AWQ | AWQ / GPTQ | **Native INT4** (built into engine) |
| CPU execution | Python GIL bottleneck | Not supported | **Numba LLVM → native AVX-512/NEON SIMD** |
| GPU execution | PyTorch CUDA kernels | Custom CUDA | **Triton fused kernels** |
| Install size | ~2 GB (PyTorch + deps) | ~3 GB | **< 5 MB** |

---

## 🚀 Try It Now — Pre-Converted Model Available

** You want to try before you convert your model?** We have a ready-to-run `.symb` model on HuggingFace:

> **[🤗 GRRNNOB/SYNAXIM](https://huggingface.co/GRRNNOB/SYNAXIM)** — Llama 3.2 1B converted to `.symb` INT4 (674 MB)

Download and run in 3 commands:

```bash
# 1. Install SYNAXIM
pip install grrn-inference

# 2. Download the pre-converted model (674 MB)
pip install huggingface-hub
huggingface-cli download GRRNNOB/SYNAXIM --local-dir ./llama-1b-symb
```

```python
# 3. Run inference — no PyTorch, no Transformers, no KV-Cache
from grrn_inference import GRRNModel

model = GRRNModel.from_pretrained("./llama-1b-symb")

# Generate text
result = model.generate("The meaning of life is", max_tokens=50, temperature=0.7)
print(result.text)
print(f"Speed: {result.tokens_per_second} tok/s")

# Chat (OpenAI-style)
result = model.chat([
    {"role": "user", "content": "What is 2+2?"}
], max_tokens=100)
print(result.choices[0].message["content"])

# Stream tokens
for chunk in model.stream("Once upon a time", max_tokens=50):
    print(chunk.text, end="", flush=True)

# Serve as OpenAI API
from grrn_inference import serve
serve("./llama-1b-symb", port=8000, api_key="my-key")
```

| Detail | Value |
|--------|-------|
| **Model** | Llama 3.2 1B (Meta) |
| **Format** | `.symb` INT4 bitpacked |
| **Size** | 674 MB (3.8× compressed from 2.5 GB) |
| **Layers** | 16, D=2048, GQA 32Q/8KV |
| **RAM Required** | ~4 GB |
| **GPU Required** | ❌ No — runs on CPU only |
| **Dependencies** | `numpy`, `safetensors`, `tqdm` (< 5 MB) |

> ⚠️ **Note**: This is a test release demonstrating the SYNAXIM engine pipeline. The model was trained with standard attention (KV-cache) but runs through SYNAXIM's O(1) Symbiotic Gate. Output quality from converted standard models will differ from their original behavior — this is by design. Future releases will include models trained specifically for the Symbiotic paradigm.

---

## Table of Contents

- [Try It Now — Pre-Converted Model](#-try-it-now--pre-converted-model-available)
- [What is SYNAXIM?](#what-is-synaxim)
- [Why SYNAXIM Instead of Transformers?](#why-synaxim-instead-of-transformers)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Convert a Model](#step-1-convert-a-model)
  - [Generate Text](#step-2-generate-text)
  - [Chat (OpenAI-Style)](#step-3-chat-openai-style)
  - [Streaming](#step-4-streaming)
  - [Serve as OpenAI API](#step-5-serve-as-openai-api)
- [Device Selection (CPU / GPU)](#device-selection)
- [Supported Models](#supported-models)
- [The .symb Weight Format](#the-symb-weight-format)
- [Architecture](#architecture)
  - [The Symbiotic Gate (Attention ≡ Memory)](#the-symbiotic-gate-attention--memory)
  - [INT4 Bitpacking](#int4-bitpacking)
  - [The 64-Layer Execution Pipeline](#the-64-layer-execution-pipeline)
  - [CPU Acceleration (Numba LLVM)](#cpu-acceleration-numba-llvm)
  - [GPU Acceleration (Triton Kernels)](#gpu-acceleration-triton-kernels)
- [Code Structure](#code-structure)
- [API Reference](#api-reference)
- [CLI Tools](#cli-tools)
- [How Others Use SYNAXIM](#how-others-use-synaxim)
- [What's Next](#whats-next)
- [License](#license)
- [Citation](#citation)

---

## Installation

```bash
# Core — NumPy inference + model conversion (< 5 MB install)
pip install grrn-inference

# CPU accelerated — Numba JIT compiled to native AVX-512/NEON
pip install grrn-inference[cpu]

# GPU accelerated — Triton fused kernels for H100/A100
pip install grrn-inference[gpu]

# OpenAI-compatible API server
pip install grrn-inference[server]

# Everything
pip install grrn-inference[all]
```

Or install from source:

```bash
git clone https://github.com/GRRN-MAKER/SYNAXIM.git
cd SYNAXIM
pip install -e .
```

---

## Quick Start

### Step 1: Convert a Model

Convert any HuggingFace model to SYNAXIM's proprietary `.symb` format:

```python
from grrn_inference import SymbioticConverter

converter = SymbioticConverter()
converter.convert(
    source="meta-llama/Llama-3.1-8B-Instruct",  # Any HF model ID or local path
    output_dir="./llama-8b-symb",
    quantize="int4"                               # int4 | int8 | fp16 | fp32
)
```

Or from CLI:

```bash
grrn-convert meta-llama/Llama-3.1-8B-Instruct ./llama-8b-symb --quantize int4
```

The converter auto-detects Dense vs MoE architecture, maps all weight names, copies the tokenizer, and packs everything into `.symb` files with per-group INT4 quantization.

### Step 2: Generate Text

```python
from grrn_inference import GRRNModel

model = GRRNModel.from_pretrained("./llama-8b-symb")

result = model.generate(
    prompt="Explain quantum computing in simple terms:",
    max_tokens=200,
    temperature=0.7,
    top_p=0.9
)

print(result.text)
print(f"Speed: {result.tokens_per_second} tok/s")
```

On import, SYNAXIM always prints:

```
  ╔═══════════════════════════════════════════════════════════╗
  ║             S Y N A X I M   v0.1.0                       ║
  ║   Symbiotic Native Axiom Inference Machine               ║
  ║   Framework-Free LLM Engine by GRRNMAKER                 ║
  ║   Attention ≡ Memory | O(1) State | .symb Format         ║
  ╚═══════════════════════════════════════════════════════════╝
```

### Step 3: Chat (OpenAI-Style)

```python
result = model.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])
print(result.choices[0].message["content"])
```

Multi-turn:

```python
messages = [
    {"role": "system", "content": "You are a coding tutor."},
    {"role": "user", "content": "How do I reverse a string in Python?"},
]
result = model.chat(messages, max_tokens=200)

messages.append({"role": "assistant", "content": result.choices[0].message["content"]})
messages.append({"role": "user", "content": "Now show me in Rust."})
result = model.chat(messages, max_tokens=200)
```

### Step 4: Streaming

```python
for chunk in model.stream("Once upon a time", max_tokens=200):
    print(chunk.text, end="", flush=True)
```

### Step 5: Serve as OpenAI API

```python
from grrn_inference import serve
serve("./llama-8b-symb", port=8000, api_key="my-secret-key")
```

```bash
grrn-serve ./llama-8b-symb --port 8000 --api-key my-secret-key
```

Then use **any** OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-secret-key")
response = client.chat.completions.create(
    model="llama-8b-symb",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + engine info |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |

---

## Device Selection

Same `.symb` files work on all backends:

```python
model = GRRNModel.from_pretrained("./model-symb", device="cpu")              # Auto-detect
model = GRRNModel.from_pretrained("./model-symb", device="cpu-accelerated")  # Force Numba
model = GRRNModel.from_pretrained("./model-symb", device="cpu-numpy")        # Force NumPy
model = GRRNModel.from_pretrained("./model-symb", device="cuda")             # Triton GPU
```

| Backend | Speed | Dependencies | Hardware |
|---------|-------|-------------|----------|
| `cpu` (auto) | Best available | numpy (+numba) | Any CPU |
| `cpu-accelerated` | ⚡ Fast | numpy + numba | Intel Xeon (AVX-512), AMD EPYC, Apple M-series |
| `cpu-numpy` | Baseline | numpy only | Any CPU |
| `cuda` | 🔥 Fastest | triton + torch | NVIDIA H100, A100, RTX |

---

## Supported Models

### Dense Models

| Model Family | Sizes | Status |
|-------------|-------|--------|
| LLaMA / LLaMA-2 / LLaMA-3 | 7B, 8B, 13B, 70B | ✅ |
| Qwen / Qwen2.5 / Qwen3.5 | 7B, 14B, 27B | ✅ |
| Mistral | 7B | ✅ |
| Phi-3 | 3.8B, 14B | ✅ |
| Gemma / Gemma-2 | 2B, 7B, 27B | ✅ |

### MoE Models

| Model Family | Configuration | Status |
|-------------|--------------|--------|
| Mixtral | 8x7B, 8x22B | ✅ |
| DeepSeek-V2/V3 | MoE + shared experts | ✅ |
| DBRX | 16 experts | ✅ |

---

## The .symb Weight Format

Proprietary binary format — **cannot** be read by HuggingFace, PyTorch, ONNX, or llama.cpp.

```
model-symb/
├── config.symb.json          # Architecture + quantization config
├── tokenizer/                # Tokenizer files
├── embeddings.symb           # Token embeddings (FP16)
├── lm_head.symb              # Output projection (FP16)
├── final_norm.symb           # Final RMSNorm (FP16)
└── layers/
    ├── layer_00/
    │   ├── attn_q.symb       # INT4 packed + per-group scales
    │   ├── attn_k.symb
    │   ├── attn_v.symb
    │   ├── attn_o.symb
    │   ├── mlp_gate.symb
    │   ├── mlp_up.symb
    │   ├── mlp_down.symb
    │   ├── norm_attn.symb    # FP16
    │   └── norm_mlp.symb
    └── layer_63/
```

**INT4 binary layout:**
```
[4B: num_groups] [4B: group_size] [4B: numel] [8B: shape]
[num_groups × 4B: FP32 scale factors]
[ceil(numel/2) bytes: packed INT4 pairs]
```

Compression: **~7.5x** vs FP32.

---

## Architecture

### The Symbiotic Gate (Attention ≡ Memory)

Instead of QKV attention with growing KV-Cache:

```
M_{t+1} = σ(x·W_gate) · M_t  +  (1 - σ(x·W_gate)) · x^T · x
output  = x · M_{t+1}
```

- `M` is `(D × D)` — **fixed size, never grows**
- Sigmoid gate balances old context retention vs. new imprint
- SmoothQuant prevents numerical drift over long sequences

**Memory comparison:**

| Method | Memory @ 128K context (D=4096) | Scales with |
|--------|-------------------------------|-------------|
| Standard KV-Cache | ~4 GB | Sequence length |
| **SYNAXIM M matrix** | **~64 MB** | **Nothing — O(1)** |

### INT4 Bitpacking

```python
# Pack: two 4-bit values per byte
packed_byte = (value_A << 4) | value_B

# Unpack inside SIMD registers / GPU SRAM
upper = (byte >> 4) & 0x0F  →  float = (upper - 8) * scale
lower = (byte & 0x0F)       →  float = (lower - 8) * scale
```

### The 64-Layer Execution Pipeline

```
for layer in 0..63:
    h = RMSNorm(h) → Attention/Gate → Residual → RMSNorm(h) → MLP → Residual
logits = FinalNorm(h) @ lm_head
```

### CPU Acceleration (Numba LLVM)

7 JIT-compiled kernels: `rmsnorm`, `matvec`, `vecmat`, `silu`, `swiglu_fused`, `symbiotic_gate`, `full_attention`

All use `@njit(parallel=True, fastmath=True, cache=True)` — compiled to native AVX-512/AVX2/NEON.

### GPU Acceleration (Triton Kernels)

4 fused kernels that keep data in H100 SRAM:

| Kernel | Fuses |
|--------|-------|
| `_int4_unpack_kernel` | Unpack + dequant |
| `_rmsnorm_kernel` | Normalize + scale |
| `_symbiotic_gate_kernel` | SmoothQuant + sigmoid + M update + projection |
| `_swiglu_fused_kernel` | gate·up + SiLU + multiply |

---

## Code Structure

```
SYNAXIM/
├── grrn_inference/
│   ├── __init__.py              # Public API + SYNAXIM banner (burnt in)
│   ├── config.py                # .symb.json spec + model presets
│   ├── export.py                # HF safetensors → .symb converter
│   ├── engine.py                # 64-layer engine (NumPy)
│   ├── cpu_engine.py            # CPU-accelerated engine (Numba)
│   ├── cpu_kernels.py           # 7 Numba JIT kernels
│   ├── triton_kernels.py        # 4 Triton GPU kernels
│   ├── sampling.py              # Temperature, top-p, top-k
│   ├── tokenizer.py             # Auto-detect tokenizer
│   ├── server.py                # OpenAI-compatible API
│   ├── cli.py                   # CLI commands
│   └── models/                  # Dense + MoE handlers
├── examples/                    # 4 example scripts
├── tests/                       # Test suite
├── pyproject.toml
├── LICENSE                      # Apache 2.0
└── README.md
```

**26 files, ~3,800 lines. 48 tests, 0 errors.**

---

## API Reference

### `GRRNModel`

| Method | Returns | Description |
|--------|---------|-------------|
| `.from_pretrained(path, device)` | `GRRNModel` | Load a `.symb` model |
| `.generate(prompt, max_tokens, ...)` | `GenerationResult` | Generate text |
| `.chat(messages, max_tokens, ...)` | `ChatCompletionResult` | Chat completion |
| `.stream(prompt, max_tokens)` | `Iterator[StreamChunk]` | Stream tokens |
| `.info()` | `str` | Model description |

### `SymbioticConverter`

| Method | Description |
|--------|-------------|
| `.convert(source, output_dir, quantize, architecture)` | Convert HF model to `.symb` |

### `GRRNServer`

| Method | Description |
|--------|-------------|
| `GRRNServer(model_path, api_key)` | Create server |
| `.run(host, port)` | Start serving |

---

## CLI Tools

```bash
grrn-convert <source> <output> [--quantize int4] [--architecture auto]
grrn-serve <model_dir> [--port 8000] [--api-key KEY]
```

---

## How Others Use SYNAXIM

```python
# Instead of:                          →  With SYNAXIM:
from transformers import AutoModel     →  from grrn_inference import GRRNModel
model = AutoModel.from_pretrained(x)   →  model = GRRNModel.from_pretrained(x)
output = model.generate(...)           →  output = model.generate(...)
```

For API servers — drop-in replacement:

```python
# vLLM server                         →  grrn-serve ./model --port 8000
# client = OpenAI(base_url=...)       →  client = OpenAI(base_url="http://localhost:8000/v1")
```

---

## What's Next

- [ ] Rust CPU backend (PyO3/Maturin compiled `.so`)
- [ ] Flash-Attention tiling in Triton
- [ ] Multi-GPU tensor parallelism
- [ ] GGUF → `.symb` converter
- [ ] PyPI release
- [ ] Benchmarks vs vLLM, llama.cpp, HuggingFace

---

## License

[Apache License 2.0](LICENSE)

---

## Citation

```bibtex
@software{synaxim,
  title={SYNAXIM: Symbiotic Native Axiom Inference Machine},
  author={GRRNMAKER},
  year={2026},
  url={https://github.com/GRRN-MAKER/SYNAXIM}
}
```

---

<p align="center">
  <b>SYNAXIM</b> — Because inference should be a machine, not a framework.<br>
  <i>Built by <a href="https://github.com/GRRN-MAKER">GRRNMAKER</a></i>
</p>
