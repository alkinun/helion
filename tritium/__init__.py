from importlib.metadata import PackageNotFoundError, version

from . import _compat as _compat  # noqa: F401 -- applies Triton build shim
from .ops import (
    adamw_step,
    add,
    add_gelu,
    add_relu,
    attention,
    cross_entropy,
    cross_entropy_backward,
    dropout,
    dropout_backward,
    gelu,
    gelu_backward,
    layernorm,
    layernorm_backward,
    linear_cross_entropy,
    linear_cross_entropy_backward,
    matmul,
    relu,
    residual_rmsnorm,
    rmsnorm,
    rmsnorm_backward,
    rope,
    rope_,
    sgd_step,
    swiglu,
    swiglu_backward,
)

try:
    __version__ = version("helion")
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
    "dropout",
    "dropout_backward",
    "gelu",
    "gelu_backward",
    "layernorm",
    "layernorm_backward",
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
