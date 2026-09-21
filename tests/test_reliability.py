import numpy as np
import pytest
from porewise import lbm_stokes, lbm_stokes_3d, lbm_stokes_2d_fast, lbm_stokes_3d_fast, lbm_stokes_2d_pressure, k_from_run, geometry
from porewise.backends import HAS_GPU
from porewise.validation import decode_mask

PATHS=['numpy2','numpy3']+(['array2','array3','cuda2','cuda3'] if HAS_GPU else [])

def solve(path, mask=None, **kw):
    ndim=int(path[-1]); m=geometry.parallel_plates(12,10,6)
    if ndim==3: m=np.broadcast_to(m,(4,)+m.shape).copy()
    if mask is not None: m=mask
    b={'numpy':'numpy','array':'cupy-array','cuda':'cuda'}[path[:-1]]
    fun=lbm_stokes if ndim==2 else lbm_stokes_3d
    settings=dict(F_x=1e-6,n_steps_max=100,conv_window=20,verbose=False,backend=b,return_fields=True)
    settings.update(kw)
    return fun(m,**settings)

@pytest.mark.parametrize('path',PATHS)
@pytest.mark.parametrize('bad',[{'tau':.5},{'tau':float('nan')},{'F_x':float('inf')},{'n_steps_max':0},{'n_steps_max':1.2},{'conv_window':0},{'conv_tol':0},{'wall_timeout_s':-1},{'precision':'bad'},{'heartbeat':0}])
def test_invalid(path,bad):
    with pytest.raises(ValueError): solve(path,**bad)

@pytest.mark.parametrize('path',PATHS)
def test_statuses(path):
    r=solve(path,n_steps_max=1)
    assert r['termination_reason']=='max_steps' and r['iterations_completed']==1
    with pytest.raises(ValueError): k_from_run(r,'x')
    assert np.isfinite(k_from_run(r,'x',allow_unconverged=True))
    r=solve(path,wall_timeout_s=1e-12)
    assert r['termination_reason']=='timeout' and not r['valid_for_permeability']
    r=solve(path,F_x=0)
    assert r['termination_reason']=='zero_forcing' and not r['valid_for_permeability']
    shape=(5,6) if path[-1]=='2' else (4,5,6)
    r=solve(path,np.ones(shape,bool))
    assert r['valid_for_permeability'] and k_from_run(r,'x')==0
    r=solve(path,np.zeros(shape,bool))
    assert r['termination_reason']=='fully_fluid_periodic' and not r['valid_for_permeability']
    with pytest.raises(ValueError): solve(path,np.zeros(shape,np.uint8))

@pytest.mark.parametrize('path',PATHS)
def test_instability(path):
    shape=(12,10) if path[-1]=='2' else (8,12,10)
    m=np.random.default_rng(42).random(shape)<.3
    r=solve(path,m,F_x=1.,n_steps_max=300,stability_every=1)
    assert r['termination_reason'] in ('nonfinite','invalid_density')
    assert not r['valid_for_permeability']

@pytest.mark.parametrize('path',PATHS)
def test_converged(path):
    r=solve(path,n_steps_max=5000,conv_tol=1e-7)
    assert r['valid_for_permeability'],r['termination_reason']
    assert r['diagnostics']['mass_drift']<1e-8
    assert abs(k_from_run(r,'x')/(.5*6**2/12)-1)<.03

@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('axis',[0,1,2])
def test_fixed_step_parity(ndim,axis):
    if axis>=ndim or not HAS_GPU: pytest.skip('dimension or device')
    rng=np.random.default_rng(7)
    shape=(12,16) if ndim==2 else (8,10,12)
    m=rng.random(shape)<.2
    fun=lbm_stokes if ndim==2 else lbm_stokes_3d
    kw=dict(F_x=0,verbose=False,n_steps_max=31,conv_window=10,return_fields=True)
    kw['F_'+'xyz'[axis]]=1e-6
    rs=[fun(m,backend=b,**kw) for b in ('numpy','cupy-array','cuda')]
    for r in rs[1:]:
        for c in 'xyz'[:ndim]: np.testing.assert_allclose(r['u'+c],rs[0]['u'+c],atol=2e-14,rtol=1e-8)


def test_mask_and_units():
    m=decode_mask(np.array([[0,255],[255,0]]),solid_value=255,pore_value=0)
    assert geometry.porosity(m)==.5
    with pytest.raises(ValueError): geometry.porosity(np.array([[0,255]]))
    with pytest.raises(ValueError): decode_mask([[0,128]],solid_value=255,pore_value=0)
    with pytest.raises(ValueError): geometry.parallel_plates(8,8,10)
    with pytest.raises(ValueError): solve('numpy3',precision='float32')


def test_zou_he_algebra():
    from porewise.d2q9_pressure import zou_he
    from porewise.d2q9 import CX,CY
    f=np.random.default_rng(2).uniform(.01,.04,(9,8,10))
    faces=np.ones(8,bool)
    zou_he(f,faces,faces,1.,.99,np)
    for col,density in ((0,1.),(-1,.99)):
        np.testing.assert_allclose(f[:,:,col].sum(axis=0),density,atol=1e-15)
        np.testing.assert_allclose((f[:,:,col]*CY[:,None]).sum(axis=0),0,atol=1e-15)

@pytest.mark.skipif(not HAS_GPU,reason='requires real CUDA')
def test_pressure():
    m=np.zeros((8,24),bool)
    for dp in (1e-5,-1e-5):
        r=lbm_stokes_2d_pressure(m,deltaP=dp,walls_y=True,pad=4,n_steps_max=15000,energy_eps=1e-6,energy_window=200,verbose=False,wall_timeout_s=30)
        assert r['valid_for_permeability'],r['termination_reason']
        assert r['diagnostics']['flux_span_relative']<1e-4
        assert r['k_lu']>0 and r['ux'].shape==m.shape
    for kw in ({'deltaP':0},{'deltaP':1},{'pad':-1},{'sample_every':0}):
        with pytest.raises(ValueError): lbm_stokes_2d_pressure(m,verbose=False,**kw)

@pytest.mark.parametrize('path',PATHS)
def test_force_below_storage_roundoff(path):
    with pytest.raises(ValueError,match='roundoff'):
        solve(path,F_x=1e-30)
    if path.startswith('cuda'):
        with pytest.raises(ValueError,match='roundoff'):
            solve(path,F_x=1e-9,precision='float32')


def test_pressure_degenerate_geometry():
    with pytest.raises(ValueError,match='faces'):
        lbm_stokes_2d_pressure(np.ones((8,24),bool),verbose=False)
    with pytest.raises(ValueError,match='confining walls'):
        lbm_stokes_2d_pressure(np.zeros((8,24),bool),verbose=False)


@pytest.mark.skipif(not HAS_GPU,reason='requires real CUDA')
def test_pressure_failure_statuses():
    m=np.zeros((8,24),bool)
    common=dict(walls_y=True,verbose=False,return_fields=False)
    r=lbm_stokes_2d_pressure(m,n_steps_max=1,**common)
    assert r['termination_reason']=='max_steps' and not r['valid_for_permeability']
    r=lbm_stokes_2d_pressure(m,wall_timeout_s=1e-12,**common)
    assert r['termination_reason']=='timeout' and r['iterations_completed']==0
    r=lbm_stokes_2d_pressure(m,deltaP=.32,n_steps_max=200,stability_every=1,**common)
    assert r['termination_reason'] in ('invalid_density','nonfinite')
    assert not r['valid_for_permeability'] and r['k_lu'] is None
