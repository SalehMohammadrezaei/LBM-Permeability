"""GPU-accelerated lattice-Boltzmann absolute-permeability solver.

Compute the Darcy permeability of a pore-scale binary image by simulating
single-phase Stokes flow (D2Q9 / D3Q19, BGK, Guo body force) and measuring
the steady-state superficial velocity.
"""
from .d2q9 import lbm_stokes, HAS_GPU
from .d3q19 import lbm_stokes_3d
from .d3q19_fast import lbm_stokes_3d_fast
from .units import (
    k_from_run,
    k_lu_to_m2,
    k_m2_to_millidarcy,
    k_millidarcy_to_m2,
)
from . import geometry

__version__ = "0.1.0"

__all__ = [
    "lbm_stokes",
    "lbm_stokes_3d",
    "lbm_stokes_3d_fast",
    "k_from_run",
    "k_lu_to_m2",
    "k_m2_to_millidarcy",
    "k_millidarcy_to_m2",
    "geometry",
    "HAS_GPU",
]
