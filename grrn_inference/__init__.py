"""
 ███████╗██╗   ██╗███╗   ██╗ █████╗ ██╗  ██╗██╗███╗   ███╗
 ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗╚██╗██╔╝██║████╗ ████║
 ███████╗ ╚████╔╝ ██╔██╗ ██║███████║ ╚███╔╝ ██║██╔████╔██║
 ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║ ██╔██╗ ██║██║╚██╔╝██║
 ███████║   ██║   ██║ ╚████║██║  ██║██╔╝ ██╗██║██║ ╚═╝ ██║
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝

SYNAXIM — Symbiotic Native Axiom Inference Machine
===================================================
Framework-Free LLM Inference Engine by GRRNMAKER

Attention ≡ Memory | No KV-Cache | O(1) State | .symb Format

Quick Start:
    from grrn_inference import SymbioticConverter, GRRNModel
    converter = SymbioticConverter()
    converter.convert("meta-llama/Llama-3.1-8B", "./llama-symb")
    model = GRRNModel.from_pretrained("./llama-symb")
    print(model.generate("Hello, world!", max_tokens=100).text)

https://github.com/GRRN-MAKER/SYNAXIM
"""

__version__ = "0.1.0"
__engine__ = "SYNAXIM"
__tagline__ = "Symbiotic Native Axiom Inference Machine"
__author__ = "GRRNMAKER"

# ── Permanent SYNAXIM Banner — burnt into every import ──
def _print_synaxim_banner():
    banner = (
        "\n"
        "  ╔═══════════════════════════════════════════════════════════╗\n"
        "  ║             S Y N A X I M   v" + __version__ + "                       ║\n"
        "  ║   Symbiotic Native Axiom Inference Machine               ║\n"
        "  ║   Framework-Free LLM Engine by GRRNMAKER                 ║\n"
        "  ║   Attention ≡ Memory | O(1) State | .symb Format         ║\n"
        "  ╚═══════════════════════════════════════════════════════════╝\n"
    )
    print(banner)

_print_synaxim_banner()

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Iterator

from .config import SymbioticConfig


def _sanitize_for_chat_template(text: str) -> str:
    """
    Sanitize input text before passing to the inference engine.
    Prevents Jinja2 template injection, formatting errors, and repetition loops.
    
    Applied automatically to all generate(), chat(), and stream() inputs.
    """
    if not isinstance(text, str):
        text = str(text)
    # Strip HTML/CSS artifacts
    text = re.sub(r'<[^>]+>', '', text)
    # Neutralize Jinja2 template injection
    text = text.replace('{{', '{ {').replace('}}', '} }')
    text = text.replace('{%', '{ %').replace('%}', '% }')
    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Detect and truncate word-level repetition (same word 4+ times in a row)
    words = text.split()
    if len(words) > 3:
        cleaned = [words[0]]
        repeat_count = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                repeat_count += 1
                if repeat_count >= 3:
                    continue
            else:
                repeat_count = 0
            cleaned.append(words[i])
        text = ' '.join(cleaned)
    # Detect phrase-level repetition (same 3+ word phrase repeated 3+ times)
    if len(words) > 9:
        for phrase_len in range(3, min(8, len(words) // 3)):
            for start in range(len(words) - phrase_len * 2):
                phrase = ' '.join(words[start:start + phrase_len]).lower()
                count = 0
                pos = start
                while pos + phrase_len <= len(words):
                    candidate = ' '.join(words[pos:pos + phrase_len]).lower()
                    if candidate == phrase:
                        count += 1
                        pos += phrase_len
                    else:
                        break
                if count >= 3:
                    # Truncate: keep first occurrence only
                    text = ' '.join(words[:start + phrase_len]) + ' ' + ' '.join(words[start + phrase_len * count:])
                    words = text.split()
                    break
    # Hard length cap (prevent memory bombs)
    if len(text) > 32000:
        text = text[:32000]
    return text.strip()
from .export import SymbioticConverter
from .sampling import SamplingParams, TokenSampler


@dataclass
class GenerationResult:
    """Result from a text generation call."""
    text: str
    token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    finish_reason: str  # "stop", "length"
    generation_time_ms: float


@dataclass
class ChatChoice:
    """A single choice in a chat completion."""
    message: dict
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class ChatCompletionResult:
    """Result from a chat completion call."""
    choices: List[ChatChoice]
    usage: dict
    model: str
    generation_time_ms: float


@dataclass
class StreamChunk:
    """A single chunk from a streaming generation."""
    text: str
    token_id: int
    is_finished: bool = False
    finish_reason: Optional[str] = None


class GRRNModel:
    """
    Main public API for SYNAXIM — Symbiotic Native Axiom Inference Machine.
    
    Usage:
        model = GRRNModel.from_pretrained("./my-model-symb")
        
        # Simple text generation
        result = model.generate("Once upon a time", max_tokens=200)
        print(result.text)
        
        # Chat completion (OpenAI-style messages)
        result = model.chat([
            {"role": "user", "content": "What is 2+2?"}
        ])
        print(result.choices[0].message["content"])
        
        # Streaming
        for chunk in model.stream("Tell me a story", max_tokens=500):
            print(chunk.text, end="", flush=True)
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        from .tokenizer import GRRNTokenizer

        self.model_dir = model_dir
        self.device = device
        self.config = SymbioticConfig.from_model_dir(model_dir)

        # Select engine backend based on device
        if device == "cpu-accelerated":
            from .cpu_engine import CPUSymbioticEngine
            self.engine = CPUSymbioticEngine(model_dir)
        elif device == "cpu":
            # Auto-detect: use accelerated if Numba available, else NumPy
            try:
                from .cpu_kernels import HAS_NUMBA
                if HAS_NUMBA:
                    from .cpu_engine import CPUSymbioticEngine
                    self.engine = CPUSymbioticEngine(model_dir)
                    self.device = "cpu-accelerated"
                else:
                    from .engine import SymbioticStateEngine
                    self.engine = SymbioticStateEngine(model_dir, device=device)
            except ImportError:
                from .engine import SymbioticStateEngine
                self.engine = SymbioticStateEngine(model_dir, device=device)
        elif device == "cpu-numpy":
            from .engine import SymbioticStateEngine
            self.engine = SymbioticStateEngine(model_dir, device="cpu")
        else:
            # "cuda" or other — use base engine (Triton path)
            from .engine import SymbioticStateEngine
            self.engine = SymbioticStateEngine(model_dir, device=device)

        tok_dir = os.path.join(model_dir, self.config.tokenizer_dir)
        self.tokenizer = GRRNTokenizer(tok_dir)

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = "cpu") -> "GRRNModel":
        """
        Load a .symb model from a directory.
        
        Args:
            model_dir: Path to directory containing config.symb.json and .symb files
            device: Backend selection:
                - "cpu": Auto-detect (Numba accelerated if available, else NumPy)
                - "cpu-accelerated": Force Numba LLVM-compiled CPU kernels
                - "cpu-numpy": Force pure NumPy (no Numba dependency)
                - "cuda": Triton GPU kernels
            
        Returns:
            GRRNModel ready for inference
        """
        return cls(model_dir, device=device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling (-1 = disabled)
            repetition_penalty: Penalty for repeated tokens
            seed: Random seed for reproducibility
            
        Returns:
            GenerationResult with generated text and metadata
        """
        t0 = time.time()

        # Sanitize input to prevent repetition/injection
        prompt = _sanitize_for_chat_template(prompt)

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)

        # Set up sampling
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )
        sampler = TokenSampler(params)

        # Reset state
        self.engine.reset()

        # Process prompt (build up state)
        for tid in prompt_ids[:-1]:
            self.engine.forward(tid)

        # Generate tokens
        generated_ids = []
        current_id = prompt_ids[-1]
        eos = self.tokenizer.eos_token_id
        finish_reason = "length"

        for _ in range(max_tokens):
            logits = self.engine.forward(current_id)
            next_id = sampler.sample(logits, generated_ids)
            generated_ids.append(next_id)
            current_id = next_id

            if eos is not None and next_id == eos:
                finish_reason = "stop"
                break

        # Decode
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        dt = time.time() - t0
        tps = len(generated_ids) / max(dt, 1e-6)

        return GenerationResult(
            text=output_text,
            token_ids=generated_ids,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated_ids),
            total_tokens=len(prompt_ids) + len(generated_ids),
            tokens_per_second=round(tps, 1),
            finish_reason=finish_reason,
            generation_time_ms=round(dt * 1000),
        )

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> ChatCompletionResult:
        """
        Chat completion with message history (OpenAI-compatible).
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            ChatCompletionResult with assistant response
        """
        # Sanitize all message contents
        messages = [
            {**m, "content": _sanitize_for_chat_template(m.get("content", ""))}
            for m in messages
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Generate
        result = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        return ChatCompletionResult(
            choices=[ChatChoice(
                message={"role": "assistant", "content": result.text},
                finish_reason=result.finish_reason,
            )],
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
            model=self.config.model_name,
            generation_time_ms=result.generation_time_ms,
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream generated tokens one at a time.
        
        Yields StreamChunk objects with incremental text.
        
        Usage:
            for chunk in model.stream("Once upon a time"):
                print(chunk.text, end="", flush=True)
        """
        # Sanitize input
        prompt = _sanitize_for_chat_template(prompt)

        prompt_ids = self.tokenizer.encode(prompt)

        params = SamplingParams(
            temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, seed=seed,
        )
        sampler = TokenSampler(params)

        self.engine.reset()
        for tid in prompt_ids[:-1]:
            self.engine.forward(tid)

        generated_ids = []
        current_id = prompt_ids[-1]
        eos = self.tokenizer.eos_token_id

        for step in range(max_tokens):
            logits = self.engine.forward(current_id)
            next_id = sampler.sample(logits, generated_ids)
            generated_ids.append(next_id)
            current_id = next_id

            # Decode just this token
            token_text = self.tokenizer.decode([next_id], skip_special_tokens=True)

            is_eos = eos is not None and next_id == eos
            is_last = step == max_tokens - 1

            yield StreamChunk(
                text=token_text,
                token_id=next_id,
                is_finished=is_eos or is_last,
                finish_reason="stop" if is_eos else ("length" if is_last else None),
            )

            if is_eos:
                break

    def info(self) -> str:
        """Print model information."""
        from .models import get_model_handler
        handler = get_model_handler(self.config)
        return handler.describe()


# Convenience imports for server
def serve(model_path: str, port: int = 8000, api_key: Optional[str] = None, **kwargs):
    """Quick-start an OpenAI-compatible server."""
    from .server import GRRNServer
    server = GRRNServer(model_path=model_path, api_key=api_key)
    server.run(port=port, **kwargs)


# Re-exports
__all__ = [
    "GRRNModel",
    "SymbioticConverter",
    "SymbioticConfig",
    "SamplingParams",
    "GenerationResult",
    "ChatCompletionResult",
    "StreamChunk",
    "serve",
    "_sanitize_for_chat_template",
]
