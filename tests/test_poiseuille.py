"""Validate the 2D solver against analytical plane-Poiseuille flow.

Body-force-driven flow between parallel plates has the exact superficial
(Darcy) permeability

    k = (a**2 / 12) * (gap / Ny)        [cells^2]

with effective aperture ``a = gap + 1`` for half-way bounce-back.  The
discrete LBM result converges to this value as the channel is refined, which
is exactly what these tests assert.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm_permeability import lbm_stokes, k_from_run, geometry


def _measure(gap, Ny):
    blocked = geometry.parallel_plates(Ny, 8, gap)
    res = lbm_stokes(blocked, F_x=1e-6, tau=1.0, n_steps_max=80000,
                     conv_tol=1e-8, conv_window=200, use_gpu=False, verbose=False)
    k = k_from_run(res, "x")
    a = gap + 1  # effective aperture, half-way bounce-back
    k_exact = (a * a / 12.0) * (gap / Ny)
    return k, k_exact


def test_poiseuille_accuracy():
    """At a well-resolved aperture the error is a few percent."""
    k, k_exact = _measure(gap=40, Ny=80)
    rel_err = abs(k - k_exact) / k_exact
    assert rel_err < 0.05, f"relative error {rel_err:.3%} too large"


def test_poiseuille_convergence():
    """Error must shrink as the channel is refined (2nd-order LBM)."""
    errs = []
    for gap, Ny in [(10, 50), (20, 60), (40, 80)]:
        k, k_exact = _measure(gap, Ny)
        errs.append(abs(k - k_exact) / k_exact)
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"


if __name__ == "__main__":
    # Runs with or without pytest installed.
    test_poiseuille_accuracy()
    print("test_poiseuille_accuracy: PASS")
    test_poiseuille_convergence()
    print("test_poiseuille_convergence: PASS")
