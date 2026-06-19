"""Compatibility shims applied at import time to keep Triton quiet.

Python 3.14 ships a ``pyconfig.h`` that defines ``_POSIX_C_SOURCE`` while
newer glibc (>= 2.41) headers also define it via ``features.h``.  Triton's
launcher C source includes both, so every cache-cold compilation emits a
harmless but noisy ``_POSIX_C_SOURCE redefined`` gcc warning on stderr.

GCC (unlike Clang) offers no per-warning flag to silence this — only ``-w``
(disable all warnings) works — so we add it to the native-compiler flags.
Real compilation **errors** still produce non-zero exit codes and are fully
reported; only warnings from the auto-generated launcher are hidden.
"""

from __future__ import annotations

import triton.runtime.build as _build_mod

_QUIET_FLAG = "-w"
_orig_build = _build_mod._build


def _build(name, src, srcdir, library_dirs, include_dirs, libraries, ccflags):
    ccflags = list(ccflags) if ccflags else []
    if _QUIET_FLAG not in ccflags:
        ccflags.append(_QUIET_FLAG)
    return _orig_build(
        name, src, srcdir, library_dirs, include_dirs, libraries, ccflags
    )


_build._tritium_patched = True  # type: ignore[attr-defined]

if not getattr(_build_mod._build, "_tritium_patched", False):
    _build_mod._build = _build
