"""Mirrored domains: sealed-sample directional permeabilities without the wrap mismatch."""
import numpy as np

from porewise import compute_permeability_tensor, mirrored, geometry


def _blobs(n=20, seed=2):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    solid = np.zeros((n, n), bool)
    for _ in range(7):
        cy, cx, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(2, 3.5)
        solid |= (y - cy) ** 2 + (x - cx) ** 2 <= r * r          # not periodic: blobs are cut at the edges
    return solid


def test_mirrored_shape_and_symmetry():
    m = _blobs()
    mm = mirrored(m)
    assert mm.shape == (40, 40) and mm.mean() == m.mean()
    assert (mm == mm[::-1]).all() and (mm == mm[:, ::-1]).all() and (mm[:20, :20] == m).all()


def test_mirror_removes_cross_flow_and_changes_k():
    m = _blobs()
    kw = dict(backend='numpy', collision='trt', tau=0.8, verbose=False, n_steps_max=60000, conv_tol=1e-9, conv_window=200)
    wrap = compute_permeability_tensor(m, **kw)
    seal = compute_permeability_tensor(m, mirror=True, **kw)
    assert wrap['valid_for_permeability'] and seal['valid_for_permeability']
    K = np.array(seal['K_lu'])
    assert abs(K[0, 1]) < 1e-9 * K[0, 0] and abs(K[1, 0]) < 1e-9 * K[1, 1]
    assert abs(np.array(wrap['K_lu'])[0, 1]) > 1e-4 * K[0, 0]          # the wrapped image does have cross-flow
    assert 'mirrored' in seal['boundary'] and 'periodic wrap' in wrap['boundary']


def test_mirror_leaves_a_periodic_symmetric_geometry_unchanged():
    m = geometry.parallel_plates(16, 4, 8)
    kw = dict(backend='numpy', collision='trt', tau=0.8, verbose=False, n_steps_max=60000, conv_tol=1e-10, conv_window=200)
    a = np.array(compute_permeability_tensor(m, **kw)['K_lu'])[0, 0]
    b = np.array(compute_permeability_tensor(m, mirror=True, **kw)['K_lu'])[0, 0]
    assert abs(a - b) < 1e-8 * a
