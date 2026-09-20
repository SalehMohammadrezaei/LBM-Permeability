"""The multi-core pore-only backend: same algorithm as cuda-sparse, exact channel, reference agreement."""
import numpy as np
import pytest

pytest.importorskip('numba')
from lbm_permeability import lbm_stokes_3d, k_from_run, geometry
from lbm_permeability.backends import HAS_GPU

TIGHT = dict(verbose=False, n_steps_max=200000, conv_tol=1e-11, conv_window=200, return_fields=True)


def _spheres(n=16, count=4, seed=5):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    solid = np.zeros((n, n, n), bool)
    for _ in range(count):
        c = rng.uniform(0, n, 3); r = rng.uniform(3, 5)
        d = [np.minimum(abs(a - b), n - abs(a - b)) for a, b in zip((z, y, x), c)]
        solid |= d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= r * r
    return solid


def test_cpu_sparse_trt_channel_is_exact():
    gap, ny, tau = 8, 16, 1.5
    m = np.broadcast_to(geometry.parallel_plates(ny, 4, gap), (3, ny, 4)).copy()
    r = lbm_stokes_3d(m, F_x=1e-6 * (tau - .5) * 2, tau=tau, collision='trt', backend='numba-sparse', **TIGHT)
    assert r['valid_for_permeability'] and r['backend'] == 'numba-sparse'
    exact = (gap ** 3 / 12 + gap / 24) / ny
    assert abs(k_from_run(r, 'x') - exact) / exact < 1e-6


@pytest.mark.parametrize('collision', ['bgk', 'trt'])
def test_cpu_sparse_matches_numpy_reference(collision):
    m = _spheres()
    kw = dict(F_z=1e-6, F_x=0., tau=0.9, collision=collision, **TIGHT)
    ref = lbm_stokes_3d(m, backend='numpy', **kw)
    cpu = lbm_stokes_3d(m, backend='numba-sparse', **kw)
    assert ref['valid_for_permeability'] and cpu['valid_for_permeability']
    assert np.abs(ref['uz'] - cpu['uz']).max() < 1e-6 * np.abs(ref['uz']).max()


def test_cpu_sparse_has_no_alternating_mode():
    m = _spheres()
    kw = dict(F_y=1e-6, F_x=0., tau=0.9, backend='numba-sparse', verbose=False, conv_tol=1e-30,
              conv_atol=1e-300, conv_window=10 ** 6, stability_every=10 ** 6, return_fields=True)
    a, b = (lbm_stokes_3d(m, n_steps_max=n, **kw)['uy'] for n in (3000, 3001))
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()   # an excited mode is about 1e-5


@pytest.mark.skipif(not HAS_GPU, reason='requires actual CUDA')
def test_cpu_and_cuda_sparse_agree_step_for_step():
    m = _spheres()
    kw = dict(F_x=1e-6, tau=0.8, collision='trt', verbose=False, n_steps_max=501, conv_tol=1e-30, conv_atol=1e-300,
              conv_window=10 ** 6, stability_every=10 ** 6, return_fields=True)
    a = lbm_stokes_3d(m, backend='cuda-sparse', **kw)['ux']
    b = lbm_stokes_3d(m, backend='numba-sparse', **kw)['ux']
    # identical algorithm; CUDA contracts multiply-adds, so only round-off separates them
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()
