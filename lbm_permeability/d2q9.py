"""D2Q9 lattice and compatibility entry point; see docs/numerical_limits.md."""
from __future__ import annotations

import time
import numpy as np

from .backends import cp, HAS_GPU

# ---------- D2Q9 lattice ----------
CX = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1], dtype=np.int8)
CY = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=np.int8)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
              1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 7, 8, 5, 6], dtype=np.int8)


def lbm_stokes(blocked, F_x=0., F_y=0., tau=1., n_steps_max=50000,
               conv_tol=1e-5, conv_window=200, use_gpu=True, verbose=True,
               *, backend=None, precision="float64", return_fields=True, **controls):
    """Periodic BGK/Guo flow. Force is force density, True marks solid."""
    from .solver import periodic
    if backend is None:
        backend = "cupy-array" if use_gpu and HAS_GPU else "numpy"
    return periodic(blocked,(F_x,F_y),tau=tau,n_steps_max=n_steps_max,
                    conv_tol=conv_tol,conv_window=conv_window,backend=backend,
                    precision=precision,return_fields=return_fields,verbose=verbose,**controls)
