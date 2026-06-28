"""
SYNAXIM: Token Sampling Strategies
====================================
Implements temperature scaling, top-p (nucleus), top-k, and
repetition penalty for next-token selection from logits.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SamplingParams:
    """Parameters controlling token sampling."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1        # -1 = disabled
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 256
    stop_tokens: Optional[List[int]] = None
    seed: Optional[int] = None


class TokenSampler:
    """Samples next tokens from logit distributions."""

    def __init__(self, params: SamplingParams):
        self.params = params
        self.rng = np.random.RandomState(params.seed)

    def sample(
        self,
        logits: np.ndarray,
        generated_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Sample a token from logits.

        Args:
            logits: Raw logits array of shape (vocab_size,)
            generated_ids: Previously generated token IDs (for repetition penalty)

        Returns:
            Selected token ID
        """
        logits = logits.astype(np.float64).copy()

        # 1. Apply repetition penalty
        if generated_ids and self.params.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated_ids)

        # 2. Apply frequency and presence penalties
        if generated_ids and (self.params.frequency_penalty != 0.0
                              or self.params.presence_penalty != 0.0):
            logits = self._apply_frequency_presence_penalty(logits, generated_ids)

        # 3. Temperature scaling
        temp = self.params.temperature
        if temp <= 0 or temp < 1e-6:
            # Greedy decoding
            return int(np.argmax(logits))
        
        logits = logits / temp

        # 4. Top-K filtering
        if self.params.top_k > 0:
            logits = self._top_k_filter(logits, self.params.top_k)

        # 5. Top-P (nucleus) filtering
        if self.params.top_p < 1.0:
            logits = self._top_p_filter(logits, self.params.top_p)

        # 6. Convert to probabilities and sample
        probs = self._softmax(logits)
        token_id = self.rng.choice(len(probs), p=probs)
        return int(token_id)

    def _apply_repetition_penalty(
        self, logits: np.ndarray, token_ids: List[int]
    ) -> np.ndarray:
        """Apply repetition penalty to previously generated tokens."""
        penalty = self.params.repetition_penalty
        unique_ids = set(token_ids)
        for tid in unique_ids:
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty
        return logits

    def _apply_frequency_presence_penalty(
        self, logits: np.ndarray, token_ids: List[int]
    ) -> np.ndarray:
        """Apply frequency and presence penalties (OpenAI style)."""
        from collections import Counter
        counts = Counter(token_ids)
        for tid, count in counts.items():
            if 0 <= tid < len(logits):
                logits[tid] -= (
                    self.params.frequency_penalty * count
                    + self.params.presence_penalty * (1 if count > 0 else 0)
                )
        return logits

    @staticmethod
    def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-K logits, set rest to -inf."""
        if k >= len(logits):
            return logits
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf
        return logits

    @staticmethod
    def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
        """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Convert to probabilities for cumsum
        max_l = np.max(sorted_logits)
        probs = np.exp(sorted_logits - max_l)
        probs = probs / np.sum(probs)
        cumsum = np.cumsum(probs)
        
        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, p) + 1
        cutoff_idx = min(cutoff_idx, len(logits))
        
        # Mask out tokens beyond cutoff
        allowed = set(sorted_indices[:cutoff_idx])
        for i in range(len(logits)):
            if i not in allowed:
                logits[i] = -np.inf
        return logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        # Filter out -inf
        mask = logits > -np.inf
        if not np.any(mask):
            # All filtered — uniform over all tokens (shouldn't happen)
            return np.ones(len(logits)) / len(logits)

        max_l = np.max(logits[mask])
        exp_l = np.where(mask, np.exp(logits - max_l), 0.0)
        return exp_l / np.sum(exp_l)
