"""Unit conversions for lattice-Boltzmann permeability.

The solver works in lattice units (LU), where the lattice spacing and time
step are both 1.  Permeability comes out in units of *cells squared*; these
helpers convert it to physical units.

Darcy's law (single phase, body-force driven):

    q = (k / mu) * (rho * F)            with rho = 1, mu = nu

so at steady state

    k_LU [cells^2] = <u>_total * nu / F

where ``<u>_total`` is the superficial velocity (averaged over the *whole*
domain, solid cells counted as u = 0).
"""
from __future__ import annotations

# 1 Darcy = 9.869233e-13 m^2  ->  1 mD = 9.869233e-16 m^2
M2_PER_MILLIDARCY = 9.869233e-16


def k_from_run(result: dict, direction: str) -> float:
    """Permeability in lattice units (cells^2) from a solver result dict."""
    nu = result["nu"]
    u = result[f"u_{direction}_mean_total"]
    F = result[f"F_{direction}"]
    if F == 0:
        return float("nan")
    return u * nu / F


def k_lu_to_m2(k_lu: float, dx_phys: float) -> float:
    """Convert permeability from cells^2 to m^2 given the cell size (m)."""
    return k_lu * dx_phys * dx_phys


def k_m2_to_millidarcy(k_m2: float) -> float:
    """Convert m^2 to milliDarcies."""
    return k_m2 / M2_PER_MILLIDARCY


def k_millidarcy_to_m2(k_mD: float) -> float:
    """Convert milliDarcies to m^2."""
    return k_mD * M2_PER_MILLIDARCY
