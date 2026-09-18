"""D3Q19 lattice and compatibility entry point."""
from __future__ import annotations

import time
import numpy as np

from .backends import cp, HAS_GPU

# ---------- D3Q19 lattice ----------
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int8)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int8)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
W = np.array([1 / 3,
              1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
              1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
              1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
# opposite-direction pairs (q, -q) for bounce-back
OPP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 10), (8, 9),
             (11, 14), (12, 13), (15, 18), (16, 17)]


def _mom_x(f, F_x):
    return (f[1] - f[2] + f[7] - f[8] + f[9] - f[10]
            + f[11] - f[12] + f[13] - f[14] + 0.5 * F_x)


def _mom_y(f, F_y):
    return (f[3] - f[4] + f[7] + f[8] - f[9] - f[10]
            + f[15] - f[16] + f[17] - f[18] + 0.5 * F_y)


def _mom_z(f, F_z):
    return (f[5] - f[6] + f[11] + f[12] - f[13] - f[14]
            + f[15] + f[16] - f[17] - f[18] + 0.5 * F_z)


def lbm_stokes_3d(blocked, F_x=1e-6, F_y=0., F_z=0., tau=1.,
                  n_steps_max=20000, conv_tol=1e-4, conv_window=500,
                  use_gpu=True, verbose=True, return_fields=False,
                  heartbeat=500, mempool_flush=2000, wall_timeout_s=7200,
                  use_kernel=True, precision="float64", *, backend=None, **controls):
    """Periodic BGK/Guo flow; array axes z,y,x, vector components x,y,z."""
    from .solver import periodic
    if backend is None:
        backend = ("cuda" if use_kernel else "cupy-array") if use_gpu and HAS_GPU else "numpy"
    return periodic(blocked,(F_x,F_y,F_z),tau=tau,n_steps_max=n_steps_max,
                    conv_tol=conv_tol,conv_window=conv_window,backend=backend,
                    precision=precision,return_fields=return_fields,verbose=verbose,
                    heartbeat=heartbeat,mempool_flush=mempool_flush,
                    wall_timeout_s=wall_timeout_s,**controls)
