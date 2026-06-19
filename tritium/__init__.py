from importlib.metadata import PackageNotFoundError, version

from .ops.adamw import adamw_step
from .ops.add import add
from .ops.add_relu import add_relu
from .ops.attention import attention
from .ops.cross_entropy import (
    cross_entropy,
    cross_entropy_backward,
    linear_cross_entropy,
    linear_cross_entropy_backward,
)
from .ops.gelu import add_gelu, gelu, gelu_backward
from .ops.matmul import matmul
from .ops.relu import relu
from .ops.rmsnorm import residual_rmsnorm, rmsnorm, rmsnorm_backward
from .ops.rope import rope, rope_
from .ops.sgd import sgd_step
from .ops.swiglu import swiglu, swiglu_backward

try:
    __version__ = version("tritium")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "add",
    "add_gelu",
    "add_relu",
    "adamw_step",
    "attention",
    "cross_entropy",
    "cross_entropy_backward",
    "gelu",
    "gelu_backward",
    "linear_cross_entropy",
    "linear_cross_entropy_backward",
    "matmul",
    "relu",
    "rope",
    "rope_",
    "residual_rmsnorm",
    "rmsnorm",
    "rmsnorm_backward",
    "sgd_step",
    "swiglu",
    "swiglu_backward",
]
