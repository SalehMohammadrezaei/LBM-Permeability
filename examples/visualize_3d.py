"""Render a 3D pore-scale geometry and its computed flow field to PNGs.

Produces the 3D figures shown in the README:
    docs/geometry_3d.png   the 3D grain pack (sphere pack)
    docs/velocity_3d.png   flow streamlines through the pore space, colored by speed

Requires PyVista for rendering (``pip install pyvista``) in addition to the
solver's own dependencies.  Off-screen rendering is used, so no display is
needed.

    python examples/visualize_3d.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes_3d, k_from_run, k_lu_to_m2, k_m2_to_millidarcy, geometry, HAS_GPU,
)

import pyvista as pv

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
DX = 2.0e-6


def to_grid(field_zyx):
    """Flatten a (Nz,Ny,Nx) array into PyVista point order (x fastest)."""
    return field_zyx.ravel(order="C")


def main():
    pv.OFF_SCREEN = True
    os.makedirs(OUT_DIR, exist_ok=True)

    N = 64
    blocked = geometry.random_spheres(N, N, N, n_spheres=45, radius=11, seed=2)
    phi = geometry.porosity(blocked)
    print(f"{N}^3 sphere pack, porosity = {phi:.3f}, backend = {'GPU' if HAS_GPU else 'CPU'}")

    Nz, Ny, Nx = blocked.shape
    grid = pv.ImageData(dimensions=(Nx, Ny, Nz))
    grid["solid"] = to_grid(blocked.astype(np.float32))

    # --- Figure 1: 3D grain pack ---
    grains = grid.contour([0.5], scalars="solid")
    p = pv.Plotter(off_screen=True, window_size=(900, 800))
    p.add_mesh(grains, color="#b9762e", smooth_shading=True,
               specular=0.3, specular_power=15)
    p.set_background("white")
    p.add_text(f"3D pore-scale geometry  (phi = {phi:.2f})",
               font_size=12, color="black")
    p.camera_position = "iso"
    p.camera.azimuth = 25
    p.camera.elevation = 15
    p.screenshot(os.path.join(OUT_DIR, "geometry_3d.png"))
    p.close()
    print("wrote geometry_3d.png")

    # --- Run the 3D solver and keep the velocity field ---
    res = lbm_stokes_3d(blocked, F_x=1e-6, tau=1.0, n_steps_max=40000,
                        conv_tol=1e-6, conv_window=1000, use_gpu=HAS_GPU,
                        verbose=True, return_fields=True)
    k_lu = k_from_run(res, "x")
    k_mD = k_m2_to_millidarcy(k_lu_to_m2(k_lu, DX))
    print(f"k_x = {k_mD:.1f} mD  (converged step {res['step_converged']})")

    speed = np.sqrt(res["ux"] ** 2 + res["uy"] ** 2 + res["uz"] ** 2)
    grid["speed"] = to_grid(speed)
    # velocity vectors as (Npts, 3): order must be (vx, vy, vz) = (ux, uy, uz)
    grid["vel"] = np.column_stack([to_grid(res["ux"]),
                                   to_grid(res["uy"]),
                                   to_grid(res["uz"])])

    # Seed streamlines across the inlet face (x small), only in pore space.
    zz, yy, xx = np.where(~blocked)
    inlet = xx < 4
    sel = np.where(inlet)[0]
    if sel.size > 350:
        sel = sel[:: max(1, sel.size // 350)]
    seeds = pv.PolyData(np.column_stack([xx[sel], yy[sel], zz[sel]]).astype(float))
    lines = grid.streamlines_from_source(
        seeds, vectors="vel", integration_direction="forward",
        max_length=400.0)

    p = pv.Plotter(off_screen=True, window_size=(960, 800))
    p.add_mesh(grains, color="#cfcfcf", opacity=0.18, smooth_shading=True)
    if lines.n_points > 0:
        p.add_mesh(lines.tube(radius=0.35), scalars="speed", cmap="magma",
                   scalar_bar_args=dict(title="speed (LU)", color="black"))
    p.set_background("white")
    p.add_text(f"3D flow streamlines   k_x = {k_mD:.0f} mD",
               font_size=12, color="black")
    p.camera_position = "iso"
    p.camera.azimuth = 25
    p.camera.elevation = 15
    p.screenshot(os.path.join(OUT_DIR, "velocity_3d.png"))
    p.close()
    print(f"wrote velocity_3d.png  ({lines.n_points} streamline points)")


if __name__ == "__main__":
    main()
