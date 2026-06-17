"""Validate the solver against an analytical porous-medium benchmark:
transverse Stokes flow through a square array of cylinders.

A square periodic array of solid cylinders (radius ``a``, cell side ``L``,
solid fraction ``c = pi a^2 / L^2``) is the canonical model of a fibrous /
granular porous medium for which the permeability is known in closed form.

Reference (square array, flow transverse to the cylinder axis):

    Sangani & Acrivos (1982), Int. J. Multiphase Flow 8, 193-206
    k / a^2 = (1 / 8c) [ -ln c - 1.476 + 2c - 1.774 c^2 + 4.076 c^3 ]

accurate for dilute-to-moderate solid fractions (c <~ 0.4).

We build a single centred cylinder in a periodic cell, drive transverse flow,
measure k with the LBM solver, and compare k/a^2 to the correlation.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import lbm_stokes, k_from_run, geometry, HAS_GPU


def sangani_acrivos_square(c: float) -> float:
    """Analytical k/a^2 for a square array of cylinders, transverse flow."""
    return (1.0 / (8.0 * c)) * (
        -math.log(c) - 1.476 + 2.0 * c - 1.774 * c * c + 4.076 * c ** 3
    )


def centred_cylinder(L: int, radius: float) -> np.ndarray:
    """Periodic square cell of side L with one solid cylinder at the centre."""
    yy, xx = np.mgrid[0:L, 0:L]
    c0 = (L - 1) / 2.0
    return (yy - c0) ** 2 + (xx - c0) ** 2 <= radius * radius


def main():
    # L=100 is small enough to reach true steady state (diffusive relaxation
    # time ~ L^2 / nu) within the step budget; larger domains need far more
    # steps and, if stopped early, under-report k (developing flow).
    L = 100
    solid_fractions = [0.10, 0.15, 0.20, 0.30]
    use_gpu = HAS_GPU

    print(f"Square array of cylinders, L={L}, backend={'GPU' if use_gpu else 'CPU'}")
    print(f"{'c (solid)':>10} {'a (cells)':>10} {'k/a^2 LBM':>12} "
          f"{'k/a^2 S&A':>12} {'rel.err':>9}")
    print("-" * 58)

    for c in solid_fractions:
        a = math.sqrt(c / math.pi) * L
        blocked = centred_cylinder(L, a)
        c_actual = blocked.mean()        # discrete solid fraction
        a_eff = math.sqrt(c_actual / math.pi) * L

        res = lbm_stokes(blocked, F_x=1e-6, tau=1.0, n_steps_max=300000,
                         conv_tol=1e-7, conv_window=1000,
                         use_gpu=use_gpu, verbose=False)
        k_lu = k_from_run(res, "x")          # cells^2
        k_over_a2 = k_lu / (a_eff * a_eff)
        k_ref = sangani_acrivos_square(c_actual)
        rel = abs(k_over_a2 - k_ref) / k_ref
        print(f"{c_actual:>10.4f} {a_eff:>10.1f} {k_over_a2:>12.4e} "
              f"{k_ref:>12.4e} {rel:>8.2%}")


if __name__ == "__main__":
    main()
