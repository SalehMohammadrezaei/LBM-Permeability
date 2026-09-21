"""Runtime-tested GPU selection. Explicit GPU failures never fall back to CPU."""
from functools import lru_cache
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

@lru_cache(maxsize=1)
def gpu_available():
    if cp is None:
        return False
    try:
        x = cp.arange(4, dtype=cp.float64)
        x = x * 2
        cp.cuda.get_current_stream().synchronize()
        return float(x.sum()) == 12
    except Exception:
        return False


def select(backend):
    if backend not in ('auto','numpy','cupy-array','cuda','cuda-sparse','numba-sparse'):
        raise ValueError('backend must be auto, numpy, cupy-array, cuda, cuda-sparse, or numba-sparse')
    if backend == 'auto':
        backend = 'cuda' if gpu_available() else 'numpy'
    if backend not in ('numpy','numba-sparse') and not gpu_available():
        raise RuntimeError(f'{backend} requires a working CUDA device and CuPy runtime')
    return backend, np if backend in ('numpy','numba-sparse') else cp

HAS_GPU = gpu_available()
