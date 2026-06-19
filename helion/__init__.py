from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .nn import (
    Attention,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ResidualRMSNorm,
    RMSNorm,
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
    "SwiGLU",
    "clip_grad_norm",
]
