"""Compute the 3D absolute permeability of a pore-scale volume.

    # synthetic demo (no data needed)
    python examples/run_3d.py --demo

    # your own segmented volume (.npy bool array, True = solid)
    python examples/run_3d.py volume.npy --direction z --dx 2e-6

3D runs are heavy: a 200^3 volume takes minutes per 1000 steps on a GPU and
is impractical on CPU.  The demo below uses a small 64^3 volume.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes_3d, k_from_run, k_lu_to_m2, k_m2_to_millidarcy, geometry, HAS_GPU,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume_npy", nargs="?", help=".npy bool volume (True = solid)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--direction", choices=["x", "y", "z"], default="x")
    ap.add_argument("--F", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--dx", type=float, default=2.0e-6)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    if args.demo or args.volume_npy is None:
        blocked = geometry.random_spheres(64, 64, 64, n_spheres=40, radius=10, seed=1)
        print("synthetic random-sphere volume 64x64x64")
    else:
        blocked = np.load(args.volume_npy).astype(bool)
        print(f"loaded {blocked.shape} volume from {args.volume_npy}")

    phi = geometry.porosity(blocked)
    print(f"porosity = {phi:.4f}   ({'GPU' if HAS_GPU and not args.no_gpu else 'CPU'})")

    F = {"x": (args.F, 0, 0), "y": (0, args.F, 0), "z": (0, 0, args.F)}[args.direction]
    res = lbm_stokes_3d(blocked, F_x=F[0], F_y=F[1], F_z=F[2], tau=args.tau,
                        n_steps_max=args.steps, conv_tol=args.tol,
                        use_gpu=not args.no_gpu)

    k_lu = k_from_run(res, args.direction)
    k_m2 = k_lu_to_m2(k_lu, args.dx)
    k_mD = k_m2_to_millidarcy(k_m2)
    print(f"\nk_{args.direction} = {k_lu:.4e} cells^2 "
          f"= {k_m2:.4e} m^2 = {k_mD:.4e} mD")
    print(f"converged at step {res['step_converged']} in {res['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
