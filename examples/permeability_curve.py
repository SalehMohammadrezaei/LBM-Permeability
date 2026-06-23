"""Sweep porosity and plot the computed permeability against the Kozeny–Carman trend.

Runs the D2Q9 solver on random grain packs of increasing solid fraction and shows
that the measured permeability collapses onto the classic k ∝ φ³/(1−φ)² law — the
headline result of pore-scale / digital-rock permeability prediction.

    python examples/permeability_curve.py     ->  docs/permeability_vs_porosity.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes, k_from_run, k_lu_to_m2, k_m2_to_millidarcy, geometry, HAS_GPU,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
DX = 2.0e-6
N = 150


def main():
    os.makedirs(OUT, exist_ok=True)
    phis, ks = [], []
    for nd in (22, 34, 46, 60, 76, 94):                       # more grains -> lower porosity
        blocked = geometry.random_disks(N, N, n_disks=nd, radius=14, seed=100 + nd)
        phi = geometry.porosity(blocked)
        res = lbm_stokes(blocked, F_x=1e-6, tau=1.0, n_steps_max=30000,
                         conv_tol=1e-5, conv_window=300, use_gpu=HAS_GPU, verbose=False)
        k_mD = k_m2_to_millidarcy(k_lu_to_m2(k_from_run(res, "x"), DX))
        phis.append(phi); ks.append(k_mD)
        print(f"φ={phi:.3f}  k={k_mD:9.1f} mD  (step {res['step_converged']})", flush=True)
    phis = np.array(phis); ks = np.array(ks)

    # Kozeny–Carman reference k = C·φ³/(1−φ)², C fit to the measured points (least squares in log)
    kc_shape = phis ** 3 / (1 - phis) ** 2
    C = np.exp(np.mean(np.log(ks) - np.log(kc_shape)))
    pp = np.linspace(phis.min() - 0.02, phis.max() + 0.02, 100)
    kc = C * pp ** 3 / (1 - pp) ** 2

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(pp, kc, "--", color="#888", lw=1.8, label="Kozeny–Carman  $k\\propto\\phi^3/(1-\\phi)^2$")
    ax.scatter(phis, ks, s=90, c=ks, cmap="viridis", edgecolor="#1a1a1a",
               zorder=5, label="LBM (this solver)")
    ax.set_yscale("log")
    ax.set_xlabel("porosity  $\\phi$")
    ax.set_ylabel("permeability  $k$  (mD)")
    ax.set_title("Pore-scale permeability vs porosity\nLBM Stokes flow through random grain packs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "permeability_vs_porosity.png"), dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}/permeability_vs_porosity.png")


if __name__ == "__main__":
    main()
