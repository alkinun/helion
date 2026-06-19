from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .amp import GradScaler, autocast
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
from .state import Checkpoint, load_checkpoint, save_checkpoint
from .training import CosineLR, clip_grad_norm

try:
    __version__ = version("helion")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "AdamW",
    "Attention",
    "Checkpoint",
    "CosineLR",
    "Dropout",
    "Embedding",
    "GradScaler",
    "LayerNorm",
    "Linear",
    "ResidualRMSNorm",
    "RMSNorm",
    "SGD",
    "Softmax",
    "SwiGLU",
    "autocast",
    "checkpoint",
    "clip_grad_norm",
    "load_checkpoint",
    "save_checkpoint",
]
