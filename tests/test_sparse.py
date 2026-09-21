"""Pore-only CUDA storage must reproduce the dense solver's steady state."""
import numpy as np
import pytest

from lbm_permeability import lbm_stokes_3d, k_from_run, geometry, compute_permeability_tensor
from lbm_permeability.backends import HAS_GPU

pytestmark = pytest.mark.skipif(not HAS_GPU, reason='requires actual CUDA')

TIGHT = dict(verbose=False, n_steps_max=200000, conv_tol=1e-11, conv_window=200, return_fields=True)


def _spheres(n=24, count=10, seed=3):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    solid = np.zeros((n, n, n), bool)
    for _ in range(count):
        c = rng.uniform(0, n, 3)
        r = rng.uniform(3, 5)
        d = [np.minimum(abs(a - b), n - abs(a - b)) for a, b in zip((z, y, x), c)]
        solid |= d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= r * r
    return solid


@pytest.mark.parametrize('collision', ['bgk', 'trt'])
def test_sparse_matches_dense_steady_state(collision):
    m = _spheres()
    kw = dict(F_y=1e-6, F_x=0., tau=0.9, collision=collision, **TIGHT)
    dense = lbm_stokes_3d(m, backend='cuda', **kw)
    sparse = lbm_stokes_3d(m, backend='cuda-sparse', **kw)
    assert dense['valid_for_permeability'] and sparse['valid_for_permeability']
    for c in 'xyz':
        scale = abs(dense['u_y_mean_total'])
        assert abs(dense[f'u_{c}_mean_total'] - sparse[f'u_{c}_mean_total']) < 1e-7 * scale
        assert np.abs(dense['u' + c] - sparse['u' + c]).max() < 1e-6 * np.abs(dense['uy']).max()
    assert not sparse['uy'][m].any()
    assert sparse['porosity'] == dense['porosity']


def test_sparse_trt_channel_is_exact_and_smaller():
    gap, ny = 8, 16
    m = np.broadcast_to(geometry.parallel_plates(ny, 4, gap), (3, ny, 4)).copy()
    tau = 1.5
    r = lbm_stokes_3d(m, F_x=1e-6 * (tau - .5) * 2, tau=tau, collision='trt', backend='cuda-sparse', **TIGHT)
    assert r['valid_for_permeability']
    exact = (gap ** 3 / 12 + gap / 24) / ny
    assert abs(k_from_run(r, 'x') - exact) / exact < 1e-6
    stored = r['memory']['sparse_bytes']
    assert stored["populations"] == 19 * 8 * (~m).sum()
    assert stored['neighbour_table'] == 18 * 4 * (~m).sum()


def test_sparse_tensor_workflow_and_float32_storage():
    m = _spheres(n=16, count=4, seed=5)
    t = compute_permeability_tensor(m, backend='cuda-sparse', tau=0.9, collision='trt', verbose=False,
                                    n_steps_max=60000, conv_tol=1e-9, conv_window=200)
    assert t['valid_for_permeability'] and t['reciprocity_error'] < 1e-3
    runs = [lbm_stokes_3d(m, F_x=1e-6, tau=0.9, collision='trt', backend='cuda-sparse', precision=p, verbose=False,
                          n_steps_max=60000, conv_tol=1e-9, conv_window=200) for p in ('float64', 'float32')]
    assert runs[1]['storage_dtype'] == 'float32' and all(r['valid_for_permeability'] for r in runs)
    # populations are stored as f_q-w_q, so single precision keeps the 1e-6 flow signal
    assert abs(runs[0]['k_lu'] - runs[1]['k_lu']) < 1e-6 * runs[0]['k_lu']


def test_sparse_is_three_dimensional_only():
    from lbm_permeability import lbm_stokes
    with pytest.raises(ValueError):
        lbm_stokes(geometry.parallel_plates(16, 4, 8), F_x=1e-6, backend='cuda-sparse', verbose=False)


def test_sparse_start_does_not_excite_the_staggered_momentum_mode():
    """Half-way bounce-back conserves an alternating momentum; the start must sit on its fixed point."""
    m = _spheres()
    kw = dict(F_y=1e-6, F_x=0., tau=0.9, backend='cuda-sparse', verbose=False, conv_tol=1e-30,
              conv_atol=1e-300, conv_window=10 ** 6, stability_every=10 ** 6, return_fields=True)
    a, b = (lbm_stokes_3d(m, n_steps_max=n, **kw)['uy'] for n in (4000, 4001))
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()   # an excited mode is about 1e-5


def test_sparse_float32_resolves_a_force_below_single_precision_epsilon():
    m = _spheres(n=16, count=4, seed=5)
    kw = dict(tau=0.9, collision='trt', backend='cuda-sparse', verbose=False, n_steps_max=60000, conv_tol=1e-9, conv_window=200)
    strong = lbm_stokes_3d(m, F_x=1e-6, precision='float64', **kw)
    weak = lbm_stokes_3d(m, F_x=1e-8, precision='float32', **kw)
    assert weak['valid_for_permeability']
    assert abs(weak['k_lu'] - strong['k_lu']) < 1e-4 * strong['k_lu']
