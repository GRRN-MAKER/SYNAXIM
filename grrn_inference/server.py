"""
SYNAXIM: OpenAI-Compatible API Server
=======================================
Drop-in replacement for OpenAI's API. Any OpenAI client library
can connect to this server and use .symb models.
Powered by SYNAXIM — Symbiotic Native Axiom Inference Machine.

Usage:
    from grrn_inference import GRRNServer
    server = GRRNServer(model_path="./my-model-symb")
    server.run(host="0.0.0.0", port=8000)

Or from command line:
    grrn-serve --model ./my-model-symb --port 8000
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException, Depends, Header
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_SERVER_DEPS = True
except ImportError:
    HAS_SERVER_DEPS = False


def _check_deps():
    if not HAS_SERVER_DEPS:
        raise RuntimeError(
            "Server dependencies not installed. Install with:\n"
            "  pip install grrn-inference[server]  (SYNAXIM server)"
        )


# ── Request/Response Models (OpenAI-compatible) ──

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    thinking_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "grrn"
    messages: List[ChatMessage] = []
    max_tokens: int = Field(default=256, le=131072)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)
    repetition_penalty: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str = "grrn"
    prompt: str = ""
    max_tokens: int = Field(default=256, le=131072)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None


class GRRNServer:
    """
    OpenAI-compatible inference server for .symb models.
    
    Exposes:
        POST /v1/chat/completions  — Chat completions (GPT-style)
        POST /v1/completions       — Text completions
        GET  /v1/models            — List available models
        GET  /health               — Health check
    """

    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        _check_deps()

        self.model_path = model_path
        self.api_key = api_key
        self.model_name = model_name

        # Lazy-load model on first request
        self._model = None
        self._tokenizer = None
        self._config = None

        self.app = self._create_app()

    def _ensure_loaded(self):
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        import os
        from .config import SymbioticConfig
        from .engine import SymbioticStateEngine
        from .tokenizer import GRRNTokenizer

        print(f"[SYNAXIM] Loading model from {self.model_path}...")
        t0 = time.time()

        self._config = SymbioticConfig.from_model_dir(self.model_path)
        self._model = SymbioticStateEngine(self.model_path)

        tok_dir = os.path.join(self.model_path, self._config.tokenizer_dir)
        self._tokenizer = GRRNTokenizer(tok_dir)

        if not self.model_name:
            self.model_name = self._config.model_name

        dt = time.time() - t0
        print(f"[SYNAXIM] Model loaded in {dt:.1f}s")
        print(f"[SYNAXIM] Architecture: {self._config.architecture}")
        print(f"[SYNAXIM] Layers: {self._config.num_layers}")
        print(f"[SYNAXIM] Hidden: {self._config.hidden_size}")

    def _create_app(self) -> "FastAPI":
        app = FastAPI(
            title="SYNAXIM API",
            description="OpenAI-compatible inference API powered by SYNAXIM — Symbiotic Native Axiom Inference Machine",
            version="0.1.0",
        )

        server = self  # Closure reference

        async def verify_api_key(authorization: Optional[str] = Header(None)):
            if server.api_key:
                if not authorization:
                    raise HTTPException(status_code=401, detail="Missing API key")
                token = authorization.replace("Bearer ", "")
                if token != server.api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "engine": "SYNAXIM",
                "version": "0.1.0",
                "model": server.model_name or "not_loaded",
                "architecture": server._config.architecture if server._config else "not_loaded",
            }

        @app.get("/v1/models")
        async def list_models(auth=Depends(verify_api_key)):
            server._ensure_loaded()
            return {
                "object": "list",
                "data": [{
                    "id": server.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "SYNAXIM",
                    "permission": [],
                }]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest, auth=Depends(verify_api_key)):
            server._ensure_loaded()
            return server._handle_chat(req)

        @app.post("/v1/completions")
        async def completions(req: CompletionRequest, auth=Depends(verify_api_key)):
            server._ensure_loaded()
            return server._handle_completion(req)

        return app

    def _handle_chat(self, req: ChatCompletionRequest) -> dict:
        """Handle a chat completion request."""
        from .sampling import SamplingParams, TokenSampler

        t0 = time.time()
        model = self._model
        tokenizer = self._tokenizer

        # Apply chat template
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text)

        # Set up sampler
        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )
        sampler = TokenSampler(params)

        # Reset engine state for new conversation
        model.reset()

        # Process prompt tokens (build up state)
        for tid in prompt_ids[:-1]:
            model.forward(tid)

        # Generate
        generated_ids = []
        current_id = prompt_ids[-1]
        eos = tokenizer.eos_token_id
        finish_reason = "length"

        for _ in range(req.max_tokens):
            logits = model.forward(current_id)
            next_id = sampler.sample(logits, generated_ids)
            generated_ids.append(next_id)
            current_id = next_id

            if eos is not None and next_id == eos:
                finish_reason = "stop"
                break

        # Decode output
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        dt = time.time() - t0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
            "generation_time_ms": round(dt * 1000),
            "engine": "SYNAXIM",
        }

    def _handle_completion(self, req: CompletionRequest) -> dict:
        """Handle a text completion request."""
        from .sampling import SamplingParams, TokenSampler

        t0 = time.time()
        model = self._model
        tokenizer = self._tokenizer

        prompt_ids = tokenizer.encode(req.prompt)

        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )
        sampler = TokenSampler(params)

        model.reset()
        for tid in prompt_ids[:-1]:
            model.forward(tid)

        generated_ids = []
        current_id = prompt_ids[-1]
        eos = tokenizer.eos_token_id
        finish_reason = "length"

        for _ in range(req.max_tokens):
            logits = model.forward(current_id)
            next_id = sampler.sample(logits, generated_ids)
            generated_ids.append(next_id)
            current_id = next_id

            if eos is not None and next_id == eos:
                finish_reason = "stop"
                break

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        dt = time.time() - t0

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "text": output_text,
                "index": 0,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
            "generation_time_ms": round(dt * 1000),
        }

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start the server."""
        _check_deps()
        print(f"[SYNAXIM] Server starting on {host}:{port}")
        if self.api_key:
            print(f"[SYNAXIM] API key authentication enabled")
        uvicorn.run(self.app, host=host, port=port, **kwargs)
