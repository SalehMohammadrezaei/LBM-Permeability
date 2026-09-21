"""Synthetic pore-scale geometries for testing and demos.

Each generator returns a boolean ``blocked`` array (``True`` = solid) using
the same convention the solvers expect.  Real studies feed in segmented
images (e.g. micro-CT or simulation snapshots) instead.
"""
from __future__ import annotations

import numpy as np
from .validation import mask, integer, positive


def parallel_plates(Ny, Nx, gap):
    """A 2D channel of fluid ``gap`` cells wide, solid walls top and bottom.

    Flow driven along x has the analytical Darcy permeability

        k = (gap**2 / 12) * (gap / Ny)        [cells^2]

    (parabolic Poiseuille profile, then converted to a superficial velocity
    over the whole Ny-tall domain).  Used as the validation benchmark.
    """
    integer("Ny",Ny); integer("Nx",Nx); integer("gap",gap)
    if gap > Ny: raise ValueError("gap cannot exceed Ny")
    blocked = np.ones((Ny, Nx), dtype=bool)
    lo = (Ny - gap) // 2
    blocked[lo:lo + gap, :] = False
    return blocked


def random_disks(Ny, Nx, n_disks, radius, seed=0, periodic=True):
    """A 2D bed of randomly placed solid disks (grains)."""
    integer("Ny",Ny); integer("Nx",Nx); positive("radius",radius)
    rng = np.random.default_rng(seed)
    integer("n_disks",n_disks,0)
    yy, xx = np.ogrid[0:Ny, 0:Nx]
    blocked = np.zeros((Ny, Nx), dtype=bool)
    for _ in range(n_disks):
        cy, cx = rng.integers(0, Ny), rng.integers(0, Nx)
        dy = np.abs(yy - cy)
        dx = np.abs(xx - cx)
        if periodic:
            dy = np.minimum(dy, Ny - dy)
            dx = np.minimum(dx, Nx - dx)
        blocked |= (dy * dy + dx * dx) <= radius * radius
    return blocked


def random_spheres(Nz, Ny, Nx, n_spheres, radius, seed=0, periodic=True):
    """Overlapping randomly placed solid spheres, not a nonoverlapping packing."""
    integer("Ny",Ny); integer("Nx",Nx); positive("radius",radius)
    rng = np.random.default_rng(seed)
    integer("Nz",Nz); integer("n_spheres",n_spheres,0)
    zz, yy, xx = np.ogrid[0:Nz, 0:Ny, 0:Nx]
    blocked = np.zeros((Nz, Ny, Nx), dtype=bool)
    for _ in range(n_spheres):
        cz, cy, cx = rng.integers(0, Nz), rng.integers(0, Ny), rng.integers(0, Nx)
        dz, dy, dx = np.abs(zz - cz), np.abs(yy - cy), np.abs(xx - cx)
        if periodic:
            dz = np.minimum(dz, Nz - dz)
            dy = np.minimum(dy, Ny - dy)
            dx = np.minimum(dx, Nx - dx)
        blocked |= (dz * dz + dy * dy + dx * dx) <= radius * radius
    return blocked


def porosity(blocked):
    """Fluid fraction of the domain."""
    blocked = mask(blocked)
    return float(np.count_nonzero(~blocked) / blocked.size)
