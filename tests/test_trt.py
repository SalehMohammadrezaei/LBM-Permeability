"""Two-relaxation-time collision: viscosity-independent wall location.

With BGK and bounce-back the effective wall position depends on tau, so the
permeability of a fixed geometry drifts with the relaxation time.  TRT with the
magic parameter 3/16 places the wall exactly half-way for plane Poiseuille flow,
independently of tau: every node then carries the analytical parabola
u(y) = F y (gap - y) / (2 nu) to round-off.  The volume average of that nodal
parabola is a midpoint sum, so the discrete permeability is
(gap**3/12 + gap/24)/Ny for every viscosity; the gap/24 term is quadrature of
an exact profile, not a wall-location error.  These tests assert all of this.
"""
import numpy as np
import pytest

from porewise import lbm_stokes, lbm_stokes_2d_fast, lbm_stokes_3d_fast, k_from_run, geometry
from porewise.backends import HAS_GPU

GAP, NY = 8, 16
K_EXACT = (GAP ** 3 / 12.0 + GAP / 24.0) / NY
TAUS = (0.6, 0.8, 1.0, 1.5, 2.0)


def _channel(tau, **kw):
    m = geometry.parallel_plates(NY, 4, GAP)
    # the force scales with nu so every tau reaches the same (small) velocity
    r = lbm_stokes(m, F_x=1e-6 * (tau - .5) * 2, tau=tau, backend='numpy', verbose=False,
                   n_steps_max=200000, conv_tol=1e-10, conv_window=200, **kw)
    assert r['valid_for_permeability'], r['termination_reason']
    return r


def test_bgk_channel_permeability_drifts_with_tau():
    k = [k_from_run(_channel(t), 'x') for t in TAUS]
    assert (max(k) - min(k)) / K_EXACT > 0.02, k


def test_trt_channel_permeability_is_exact_for_every_tau():
    for t in TAUS:
        r = _channel(t, collision='trt')
        assert r['collision'] == 'trt' and r['magic'] == 3 / 16
        k = k_from_run(r, 'x')
        assert abs(k - K_EXACT) / K_EXACT < 1e-6, (t, k, K_EXACT)
        y = np.arange(GAP) + .5
        exact = r['F_x'] * y * (GAP - y) / (2 * r['nu'])
        u = r['ux'][(NY - GAP) // 2:(NY - GAP) // 2 + GAP, 0]
        assert np.abs(u - exact).max() / exact.max() < 1e-6, (t, u, exact)


def test_trt_with_equal_rates_reproduces_bgk():
    tau = 0.8
    a = _channel(tau)
    b = _channel(tau, collision='trt', magic=(tau - .5) ** 2)
    assert np.allclose(a['ux'], b['ux'], rtol=1e-10, atol=1e-18)


def test_collision_arguments_are_validated():
    m = geometry.parallel_plates(NY, 4, GAP)
    with pytest.raises(ValueError):
        lbm_stokes(m, F_x=1e-6, backend='numpy', verbose=False, collision='mrt')
    with pytest.raises(ValueError):
        lbm_stokes(m, F_x=1e-6, backend='numpy', verbose=False, collision='trt', magic=0.)


def test_bgk_default_is_recorded():
    m = geometry.parallel_plates(NY, 4, GAP)
    r = lbm_stokes(m, F_x=1e-6, backend='numpy', verbose=False, n_steps_max=10, conv_window=5)
    assert r['collision'] == 'bgk' and r['magic'] is None


@pytest.mark.skipif(not HAS_GPU, reason='requires actual CUDA')
def test_trt_cuda_matches_reference_and_is_exact():
    tau = 1.5
    F = 1e-6 * (tau - .5) * 2
    ref = _channel(tau, collision='trt')
    m2 = geometry.parallel_plates(NY, 4, GAP)
    kw = dict(tau=tau, collision='trt', verbose=False, n_steps_max=200000, conv_tol=1e-10, conv_window=200)
    g2 = lbm_stokes_2d_fast(m2, F_x=F, **kw)
    assert g2['valid_for_permeability']
    assert np.allclose(g2['ux'], ref['ux'], rtol=1e-9, atol=1e-18)
    m3 = np.broadcast_to(m2, (3, NY, 4)).copy()
    g3 = lbm_stokes_3d_fast(m3, F_x=F, **kw)
    assert g3['valid_for_permeability']
    assert abs(k_from_run(g3, 'x') - K_EXACT) / K_EXACT < 1e-6
