"""Engramveil — Titans test-time neural long-term memory oracle.

Behrouz et al., Google Research (2024): Learning to Memorize at Test Time.
Pure-NumPy reference implementation of surprise-gated LTM + sliding-window STM
with MAG / MAC / MAL memory–attention fusion.
"""

from __future__ import annotations

from engramveil.attention import SlidingWindowAttention, causal_mask, softmax
from engramveil.fusion import TitansBlock, TitansConfig
from engramveil.memory import NeuralLongTermMemory, SurpriseGate, ltm_gradient

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NeuralLongTermMemory",
    "SurpriseGate",
    "ltm_gradient",
    "SlidingWindowAttention",
    "causal_mask",
    "softmax",
    "TitansBlock",
    "TitansConfig",
]
