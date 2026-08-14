"""Validate the pressure-driven D2Q9 solver two ways:

1. Against the exact plane-Poiseuille permeability  k = gap**3 / (12*Ny).
2. Against the body-force/periodic solver on random grain packs — the two
   independent drivers must return the same permeability for the same geometry.

Both are the same checks used for the body-force solver, so passing them means
the pressure boundary conditions (Zou & He) and the pressure-gradient
measurement are correct.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes, k_from_run, geometry, HAS_GPU, lbm_stokes_2d_pressure,
)


def main():
    if not HAS_GPU:
        print("pressure-driven solver needs a GPU (CuPy); skipping.")
        return

    # 1. exact Poiseuille
    gap, Ny, Nx = 20, 40, 80
    blk = geometry.parallel_plates(Ny, Nx, gap)
    r = lbm_stokes_2d_pressure(blk, deltaP=1e-4, tau=1.0, pad=0,
                               n_steps_max=200000, energy_eps=1e-5, verbose=False)
    k_exact = gap ** 3 / (12 * Ny)
    print(f"Poiseuille: pressure k={r['k_lu']:.4f}  exact={k_exact:.4f}  "
          f"err={abs(r['k_lu'] - k_exact) / k_exact * 100:.2f}%")

    # 2. pressure vs body force on porous packs
    print(f"\n{'seed':>5} {'phi':>6} {'k body-force':>13} {'k pressure':>11} {'diff':>7}")
    print("-" * 46)
    for seed in (5, 9, 13):
        blk = geometry.random_disks(200, 200, n_disks=22, radius=20, seed=seed)
        phi = geometry.porosity(blk)
        rb = lbm_stokes(blk, F_x=1e-6, tau=1.0, n_steps_max=200000,
                        conv_tol=1e-8, conv_window=200, use_gpu=True, verbose=False)
        kb = k_from_run(rb, "x")
        rp = lbm_stokes_2d_pressure(blk, deltaP=1e-4, tau=1.0, pad=6,
                                    n_steps_max=200000, energy_eps=1e-6, verbose=False)
        print(f"{seed:>5} {phi:>6.3f} {kb:>13.4f} {rp['k_lu']:>11.4f} "
              f"{abs(kb - rp['k_lu']) / kb * 100:>6.2f}%")


if __name__ == "__main__":
    main()
