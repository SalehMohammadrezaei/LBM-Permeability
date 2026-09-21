"""Validate the 3D solver against the canonical porous-medium benchmark for
pore-scale LBM codes: slow flow through a **simple-cubic array of spheres**.

A single solid sphere (radius ``a``) centred in a periodic cubic cell of side
``L`` is a simple-cubic lattice of spheres at solid fraction ``c = (4/3)pi a^3 / L^3``.
Legacy series comparison, with unverified averaging normalization (see docs/reference_scope.md). Its Stokes permeability is treated semi-analytically (Hasimoto 1959; Sangani &
Acrivos 1982):

    K(c) = F / (6 pi mu a U)                      (normalised drag)
    1/K  = 1 - 1.7601 c^(1/3) + c - 1.5593 c^2 + 3.9799 c^(8/3) - 3.0734 c^(10/3)
    k/a^2 = 2 / (9 c K)                           (permeability, from a cell force balance)

This is the 3D counterpart of ``cylinder_array.py`` (2D, Sangani-Acrivos
cylinders). We build the sphere, drive periodic body-force flow, measure k with
the LBM solver, and compare k/a^2 to the series.

Reference: A.S. Sangani & A. Acrivos, "Slow flow through a periodic array of
spheres", Int. J. Multiphase Flow 8 (1982) 343-360; H. Hasimoto, J. Fluid Mech.
5 (1959) 317.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from porewise import k_from_run, HAS_GPU
from porewise.d3q19 import lbm_stokes_3d


def sangani_acrivos_sc(c: float):
    """Return (k/a^2, K) for a simple-cubic sphere array at solid fraction c."""
    x = c ** (1.0 / 3.0)
    Kinv = (1.0 - 1.7601 * x + c - 1.5593 * c * c
            + 3.9799 * c ** (8.0 / 3.0) - 3.0734 * c ** (10.0 / 3.0))
    K = 1.0 / Kinv
    return 2.0 / (9.0 * c * K), K


def sc_sphere(L: int, a: float) -> np.ndarray:
    """Periodic cubic cell of side L with one centred solid sphere (True=solid)."""
    zz, yy, xx = np.ogrid[0:L, 0:L, 0:L]
    c0 = (L - 1) / 2.0
    return (zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2 <= a * a


def run_case(L, a, n_steps_max=80000, conv_tol=1e-4):
    blocked = sc_sphere(L, a)
    c = float(blocked.mean())                       # discrete solid fraction
    a_eff = (c * L ** 3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)  # radius from volume
    res = lbm_stokes_3d(blocked, F_x=1e-6, tau=1.0, n_steps_max=n_steps_max,
                        conv_tol=conv_tol, conv_window=500, use_gpu=HAS_GPU,
                        use_kernel=HAS_GPU, verbose=False)
    k_lu = k_from_run(res, "x")                     # cells^2
    ka2_lbm = k_lu / (a_eff * a_eff)
    ka2_ref, K = sangani_acrivos_sc(c)
    return dict(L=L, a=a, c=c, a_eff=a_eff, ka2_lbm=ka2_lbm, ka2_ref=ka2_ref,
                K=K, step=res["step_converged"],
                rel=abs(ka2_lbm - ka2_ref) / ka2_ref)


def main():
    from benchmarks.run_cases import cases, run
    from pathlib import Path
    root=Path("results/sphere-validation")
    for case in cases():
        if case['id'].startswith('sphere_'): run(case,root,120)

if __name__ == "__main__": main()
