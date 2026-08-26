from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "trie.mojo")
_EXPLICIT_LIB = os.environ.get("MOJO_PYGTRIE_LIB")
LIB = _EXPLICIT_LIB or os.path.join(ROOT, "dist", "libmojo-pygtrie.so")

I = ctypes.c_int64

_SIGNATURES = {
    "mpg_fill_i64": ([I] * 3, None),
    "mpg_grow_i64": ([I] * 5, None),
    "mpg_rehash": ([I] * 8, None),
    "mpg_find": ([I] * 6, I),
    "mpg_insert": ([I] * 14, I),
    "mpg_bulk_insert": ([I] * 16, None),
    "mpg_trace": ([I] * 7, I),
    "mpg_bulk_find": ([I] * 8, None),
    "mpg_bulk_longest": ([I] * 9, None),
    "mpg_collect": ([I] * 8, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if _EXPLICIT_LIB:
        if not os.path.isfile(LIB):
            raise BuildError(f"MOJO_PYGTRIE_LIB does not exist: {LIB}")
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        # PyDLL keeps the GIL held.  The native calls borrow NumPy buffers owned
        # by the trie, so another Python thread must not be able to replace and
        # release those buffers until the call returns.
        _loaded = ctypes.PyDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def addr(array: np.ndarray) -> int:
    """Return an address only for an ABI-compatible, live NumPy buffer."""
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.ndim != 1 or not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native buffers must be aligned, contiguous 1-D arrays")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.uint8)):
        raise TypeError("native buffers must have dtype int64 or uint8")
    pointer = int(array.ctypes.data)
    if not pointer:
        raise ValueError("native buffer has a null data pointer")
    return pointer
