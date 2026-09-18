import numpy as np
import pytest
from lbm_permeability import lbm_stokes,lbm_stokes_2d_fast,lbm_stokes_3d_fast,k_from_run,geometry
from lbm_permeability.backends import HAS_GPU


def test_channel_profile_and_closed_directions():
    for gap in (12,24,48):
        m=geometry.parallel_plates(2*gap,8,gap)
        r=lbm_stokes(m,F_x=1e-6*(12/gap)**2,backend='numpy',verbose=False,n_steps_max=80000,conv_tol=1e-7,conv_window=100)
        assert r['valid_for_permeability']
        yy=np.arange(gap)+.5
        expected=r['F_x']*yy*(gap-yy)/(2*r['nu'])
        u=r['ux'][gap//2:gap//2+gap,0]
        # Deliberately include the known small tau-dependent offset without fitting it.
        assert np.linalg.norm(u-expected)/np.linalg.norm(expected)<.01
    m=geometry.parallel_plates(12,10,6)
    r=lbm_stokes(m,F_y=1e-6,backend='numpy',verbose=False,n_steps_max=15000,conv_tol=1e-6,conv_window=50)
    assert r['valid_for_permeability']
    assert abs(k_from_run(r,'y'))<1e-4
    cavity=np.ones((12,16),bool);cavity[3:9,4:12]=False
    r=lbm_stokes(cavity,F_x=1e-6,backend='numpy',verbose=False,n_steps_max=15000,conv_tol=1e-6,conv_window=50)
    assert r['valid_for_permeability'] and abs(k_from_run(r,'x'))<1e-4

@pytest.mark.skipif(not HAS_GPU,reason='requires actual CUDA')
def test_direct_fast_wrappers():
    m=geometry.parallel_plates(12,10,6)
    for fun,mask in ((lbm_stokes_2d_fast,m),(lbm_stokes_3d_fast,np.broadcast_to(m,(4,12,10)).copy())):
        r=fun(mask,n_steps_max=100,conv_window=20,verbose=False)
        assert not r['valid_for_permeability'] and r['iterations_completed']==100
        with pytest.raises(ValueError):fun(mask,n_steps_max=0,verbose=False)


def test_backend_requests_and_precision(monkeypatch):
    import lbm_permeability.backends as b
    monkeypatch.setattr(b,'gpu_available',lambda:False)
    assert b.select('auto')[0]=='numpy'
    for backend in ('cuda','cupy-array'):
        with pytest.raises(RuntimeError):b.select(backend)
    with pytest.raises(ValueError):b.select('typo')
