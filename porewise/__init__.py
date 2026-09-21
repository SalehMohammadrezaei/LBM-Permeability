"""PoreWise: permeability of pore-scale images with the lattice Boltzmann method.

Single-phase creeping flow (D2Q9 / D3Q19, BGK or TRT collision, Guo body force) is solved in
the pore space of a binary image; the steady superficial velocity gives the Darcy permeability
in one direction or the full tensor.  Backends run on NVIDIA GPUs and on CPU cores.
"""
from .d2q9 import lbm_stokes, HAS_GPU
from .d2q9_fast import lbm_stokes_2d_fast
from .d2q9_pressure import lbm_stokes_2d_pressure
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
    "lbm_stokes_2d_fast",
    "lbm_stokes_2d_pressure",
    "lbm_stokes_3d",
    "lbm_stokes_3d_fast",
    "k_from_run",
    "k_lu_to_m2",
    "k_m2_to_millidarcy",
    "k_millidarcy_to_m2",
    "geometry",
    "HAS_GPU",
]

from .tensor import compute_permeability_tensor, assemble_tensor, mirrored
from .validation import decode_mask
__all__ += ['compute_permeability_tensor', 'assemble_tensor', 'mirrored', 'decode_mask']
