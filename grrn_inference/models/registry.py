"""
SYNAXIM: Model Architecture Registry
======================================
Auto-detects model architecture and returns the appropriate handler.
"""

from __future__ import annotations

from ..config import SymbioticConfig
from .dense import DenseModelHandler
from .moe import MoEModelHandler


def get_model_handler(config: SymbioticConfig):
    """
    Auto-detect and return the appropriate model handler.
    
    Args:
        config: SymbioticConfig for the model
        
    Returns:
        DenseModelHandler or MoEModelHandler
    """
    if config.architecture == "moe":
        return MoEModelHandler(config)
    else:
        return DenseModelHandler(config)
