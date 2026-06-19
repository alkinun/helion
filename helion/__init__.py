from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .checkpoint import checkpoint
from .nn import (
    Attention,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ResidualRMSNorm,
    RMSNorm,
    Softmax,
    SwiGLU,
)
from .optim import SGD, AdamW
from .training import CosineLR, clip_grad_norm

try:
    __version__ = version("helion")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "AdamW",
    "Attention",
    "CosineLR",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Linear",
    "ResidualRMSNorm",
    "RMSNorm",
    "SGD",
    "Softmax",
    "SwiGLU",
    "checkpoint",
    "clip_grad_norm",
]
