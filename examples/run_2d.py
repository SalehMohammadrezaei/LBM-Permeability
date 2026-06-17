"""Compute the 2D absolute permeability of a pore-scale image.

Either point at a binary mask on disk (.npy bool array, ``True`` = solid) or
let the script generate a synthetic random-disk bed so it runs out of the box:

    # synthetic demo (no data needed)
    python examples/run_2d.py --demo

    # your own segmented image
    python examples/run_2d.py mask.npy --direction x --dx 2e-6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import (
    lbm_stokes, k_from_run, k_lu_to_m2, k_m2_to_millidarcy, geometry, HAS_GPU,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mask_npy", nargs="?", help=".npy bool mask (True = solid)")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic random-disk geometry")
    ap.add_argument("--direction", choices=["x", "y"], default="x")
    ap.add_argument("--F", type=float, default=1e-6, help="body force (LBM units)")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--dx", type=float, default=2.0e-6,
                    help="physical cell size in metres")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    if args.demo or args.mask_npy is None:
        blocked = geometry.random_disks(256, 256, n_disks=60, radius=18, seed=1)
        print("synthetic random-disk geometry 256x256")
    else:
        blocked = np.load(args.mask_npy).astype(bool)
        print(f"loaded {blocked.shape} mask from {args.mask_npy}")

    phi = geometry.porosity(blocked)
    print(f"porosity = {phi:.4f}   ({'GPU' if HAS_GPU and not args.no_gpu else 'CPU'})")

    F_x = args.F if args.direction == "x" else 0.0
    F_y = args.F if args.direction == "y" else 0.0
    res = lbm_stokes(blocked, F_x=F_x, F_y=F_y, tau=args.tau,
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
