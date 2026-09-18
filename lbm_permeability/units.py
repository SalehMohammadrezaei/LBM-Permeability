"""Unit conversions for lattice-Boltzmann permeability.

The solver works in lattice units (LU), where the lattice spacing and time
step are both 1.  Permeability comes out in units of *cells squared*; these
helpers convert it to physical units.

Darcy's law (single phase, body-force driven):

    q = (k / mu) * F   (F is force density)            with rho = 1, mu = nu

so at steady state

    k_LU [cells^2] = <u>_total * nu / F

where ``<u>_total`` is the superficial velocity (averaged over the *whole*
domain, solid cells counted as u = 0).
"""
from __future__ import annotations

# 1 Darcy = 9.869233e-13 m^2  ->  1 mD = 9.869233e-16 m^2
M2_PER_MILLIDARCY = 9.869233e-16


def k_from_run(result: dict, direction: str, *, allow_unconverged=False) -> float:
    """Extract an axis-aligned response; explicit estimates never certify validity."""
    import numpy as np
    if result.get('termination_reason') in ('nonfinite','invalid_density','fully_fluid_periodic','zero_forcing'):
        raise ValueError('this termination state has no meaningful permeability estimate')
    if not result.get("valid_for_permeability", False) and not allow_unconverged:
        raise ValueError(f"run is not valid for permeability: {result.get('termination_reason', 'missing status')}")
    if direction not in ('x','y','z') or f'u_{direction}_mean_total' not in result:
        raise ValueError('direction is not present in this result')
    if 'deltaP' in result:
        if direction != 'x': raise ValueError('pressure path supports x extraction only')
        value=result['k_lu_diagnostic_estimate']
    else:
        force=[result.get(f'F_{c}',0) for c in 'xyz']
        if sum(f != 0 for f in force) != 1 or result[f'F_{direction}']==0:
            raise ValueError('scalar extraction requires one nonzero load in the requested direction; use tensor workflow')
        value=result[f'u_{direction}_mean_total']*result['nu']/result[f'F_{direction}']
    if value is None or not np.isfinite(value):
        raise ValueError('permeability estimate is nonfinite or undefined')
    return float(value)


def k_lu_to_m2(k_lu: float, dx_phys: float) -> float:
    """Convert permeability from cells^2 to m^2 given the cell size (m)."""
    from .validation import spacing
    dx_phys = spacing(dx_phys)
    return k_lu * dx_phys * dx_phys


def k_m2_to_millidarcy(k_m2: float) -> float:
    """Convert m^2 to milliDarcies."""
    return k_m2 / M2_PER_MILLIDARCY


def k_millidarcy_to_m2(k_mD: float) -> float:
    """Convert milliDarcies to m^2."""
    return k_mD * M2_PER_MILLIDARCY
