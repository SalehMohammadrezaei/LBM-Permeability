"""Render the pore geometry and the computed velocity field to PNGs.

Produces the figures shown in the README:
    docs/geometry.png   the binary pore-scale image (solid vs. pore)
    docs/velocity.png   steady-state speed |u| with flow streamlines

    python examples/visualize.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes, k_from_run, k_lu_to_m2, k_m2_to_millidarcy, geometry, HAS_GPU,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
DX = 2.0e-6  # physical cell size (m)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # A synthetic grain pack — reproducible, no external data needed.
    blocked = geometry.random_disks(460, 460, n_disks=120, radius=26, seed=3)
    phi = geometry.porosity(blocked)
    print(f"porosity = {phi:.3f}   backend = {'GPU' if HAS_GPU else 'CPU'}")

    res = lbm_stokes(blocked, F_x=1e-6, tau=1.0, n_steps_max=150000,
                     conv_tol=1e-6, conv_window=500, use_gpu=HAS_GPU, verbose=False)
    ux, uy = res["ux"], res["uy"]
    speed = np.sqrt(ux ** 2 + uy ** 2)
    speed_masked = np.ma.masked_where(blocked, speed)

    k_lu = k_from_run(res, "x")
    k_mD = k_m2_to_millidarcy(k_lu_to_m2(k_lu, DX))
    print(f"k_x = {k_mD:.1f} mD   (converged step {res['step_converged']})")

    # --- Figure 1: geometry ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    grain_cmap = ListedColormap(["#f5f5f5", "#2b2b2b"])  # pore, solid
    ax.imshow(blocked, cmap=grain_cmap, origin="lower", interpolation="nearest")
    ax.set_title(f"Pore-scale geometry\n$\\phi$ = {phi:.2f} (pore fraction)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "geometry.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: velocity magnitude + streamlines ---
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    speed_cmap = matplotlib.colormaps["magma"].copy()
    speed_cmap.set_bad("#c2c2c2")        # solid grains -> neutral gray
    im = ax.imshow(speed_masked, cmap=speed_cmap, origin="lower",
                   interpolation="bilinear")
    ny, nx = speed.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    # mask velocity inside grains so streamlines don't cross them
    ux_p = np.where(blocked, np.nan, ux)
    uy_p = np.where(blocked, np.nan, uy)
    ax.streamplot(X, Y, ux_p, uy_p, color="white", density=1.1,
                  linewidth=0.6, arrowsize=0.7)
    ax.set_title(f"Steady-state flow  |u|\n"
                 f"$k_x$ = {k_mD:.0f} mD  (flow $\\rightarrow$)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, nx - 1); ax.set_ylim(0, ny - 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("velocity magnitude $|u|$ (lattice units)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "velocity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_DIR}/geometry.png and {OUT_DIR}/velocity.png")


if __name__ == "__main__":
    main()
