"""Validate the solver against an analytical porous-medium benchmark:
transverse Stokes flow through a square array of cylinders.

A square periodic array of solid cylinders (radius ``a``, cell side ``L``,
solid fraction ``c = pi a^2 / L^2``) is the canonical model of a fibrous /
granular porous medium for which numerical drag solutions and asymptotic expansions exist.

Reference (square array, flow transverse to the cylinder axis):

    Sangani & Acrivos (1982), Int. J. Multiphase Flow 8, 193-206
    k / a^2 = (1 / 8c) [ -ln c - 1.476 + 2c - 1.774 c^2 + 4.076 c^3 ]

This legacy truncated correlation is not an exact reference; see docs/reference_scope.md.

We build a single centred cylinder in a periodic cell, drive transverse flow,
measure k with the LBM solver, and compare k/a^2 to the correlation.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from porewise import lbm_stokes, k_from_run, geometry, HAS_GPU


def sangani_acrivos_square(c: float) -> float:
    """Legacy truncated k/a^2 correlation; averaging convention needs primary-source verification."""
    return (1.0 / (8.0 * c)) * (
        -math.log(c) - 1.476 + 2.0 * c - 1.774 * c * c + 4.076 * c ** 3
    )


def centred_cylinder(L: int, radius: float) -> np.ndarray:
    """Periodic square cell of side L with one solid cylinder at the centre."""
    yy, xx = np.mgrid[0:L, 0:L]
    c0 = (L - 1) / 2.0
    return (yy - c0) ** 2 + (xx - c0) ** 2 <= radius * radius


def main():
    from benchmarks.run_cases import cases, run
    from pathlib import Path
    root=Path("results/cylinder-validation")
    for case in cases():
        if case['id'].startswith('cylinder_'): run(case,root,60)

if __name__ == "__main__": main()
