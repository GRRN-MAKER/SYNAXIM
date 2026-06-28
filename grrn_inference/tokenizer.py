"""
SYNAXIM: Tokenizer Wrapper
============================
Unified tokenizer interface that auto-detects and loads the appropriate
tokenizer from the model's tokenizer directory.

Supports: SentencePiece (.model), Tiktoken (.tiktoken), HF tokenizer.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional


class GRRNTokenizer:
    """
    Universal tokenizer wrapper for .symb models.
    
    Auto-detects the tokenizer format from the model directory and provides
    a unified encode/decode interface.
    """

    def __init__(self, tokenizer_dir: str):
        self.tokenizer_dir = tokenizer_dir
        self._tokenizer = None
        self._backend = None
        self._eos_token_id: Optional[int] = None
        self._bos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None
        self._special_tokens: dict = {}

        self._load(tokenizer_dir)

    def _load(self, tokenizer_dir: str) -> None:
        """Auto-detect and load the tokenizer."""
        # Try HF tokenizers (tokenizer.json) first — fastest
        tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json):
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(tokenizer_json)
                self._backend = "hf_tokenizers"
                self._load_special_tokens(tokenizer_dir)
                return
            except ImportError:
                pass

        # Try SentencePiece (.model)
        sp_model = os.path.join(tokenizer_dir, "tokenizer.model")
        if os.path.exists(sp_model):
            try:
                import sentencepiece as spm
                self._tokenizer = spm.SentencePieceProcessor()
                self._tokenizer.Load(sp_model)
                self._backend = "sentencepiece"
                self._bos_token_id = self._tokenizer.bos_id()
                self._eos_token_id = self._tokenizer.eos_id()
                self._pad_token_id = self._tokenizer.pad_id()
                return
            except ImportError:
                pass

        # Try tiktoken
        tiktoken_files = list(Path(tokenizer_dir).glob("*.tiktoken"))
        if tiktoken_files:
            try:
                import tiktoken
                # Read the tiktoken encoding
                self._tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback
                self._backend = "tiktoken"
                self._load_special_tokens(tokenizer_dir)
                return
            except ImportError:
                pass

        # Fallback: try loading with transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=True
            )
            self._backend = "transformers"
            self._eos_token_id = self._tokenizer.eos_token_id
            self._bos_token_id = self._tokenizer.bos_token_id
            self._pad_token_id = self._tokenizer.pad_token_id
            return
        except (ImportError, Exception):
            pass

        raise RuntimeError(
            f"Could not load tokenizer from {tokenizer_dir}. "
            "Install one of: tokenizers, sentencepiece, tiktoken, or transformers"
        )

    def _load_special_tokens(self, tokenizer_dir: str) -> None:
        """Load special token IDs from tokenizer_config.json."""
        config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract EOS token ID
            eos = config.get("eos_token")
            if isinstance(eos, dict):
                eos = eos.get("content", "")
            if eos and self._tokenizer:
                ids = self.encode(eos)
                if ids:
                    self._eos_token_id = ids[0] if len(ids) == 1 else ids[-1]
            
            bos = config.get("bos_token")
            if isinstance(bos, dict):
                bos = bos.get("content", "")
            if bos and self._tokenizer:
                ids = self.encode(bos)
                if ids:
                    self._bos_token_id = ids[0]

        # Also check special_tokens_map.json
        special_path = os.path.join(tokenizer_dir, "special_tokens_map.json")
        if os.path.exists(special_path):
            with open(special_path) as f:
                self._special_tokens = json.load(f)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if self._backend == "hf_tokenizers":
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids if hasattr(encoding, 'ids') else list(encoding)
        elif self._backend == "sentencepiece":
            return self._tokenizer.Encode(text)
        elif self._backend == "tiktoken":
            return self._tokenizer.encode(text)
        elif self._backend == "transformers":
            return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            raise RuntimeError("No tokenizer backend loaded")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self._backend == "hf_tokenizers":
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        elif self._backend == "sentencepiece":
            return self._tokenizer.Decode(token_ids)
        elif self._backend == "tiktoken":
            return self._tokenizer.decode(token_ids)
        elif self._backend == "transformers":
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            raise RuntimeError("No tokenizer backend loaded")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_token_id

    @property
    def vocab_size(self) -> int:
        if self._backend == "hf_tokenizers":
            return self._tokenizer.get_vocab_size()
        elif self._backend == "sentencepiece":
            return self._tokenizer.GetPieceSize()
        elif self._backend == "tiktoken":
            return self._tokenizer.max_token_value + 1
        elif self._backend == "transformers":
            return len(self._tokenizer)
        return 0

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply a chat template to a list of messages.
        Falls back to a simple ChatML-style template if not available.
        """
        if self._backend == "transformers" and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

        # Fallback: ChatML template
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
