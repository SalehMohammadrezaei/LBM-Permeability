import numpy as np
import pytest
from porewise.tensor import assemble_tensor,compute_permeability_tensor
from porewise.morphology import local_thickness_distribution


def test_known_tensor():
    K=np.array([[2.,-2.],[-2.,2.]])
    loads=[]
    for j in range(2):
        F=np.zeros(2);F[j]=1e-6
        loads.append(dict(nu=.1,F_x=F[0],F_y=F[1],u_x_mean_total=K[0,j]*1e-5,u_y_mean_total=K[1,j]*1e-5,valid_for_permeability=True))
    r=assemble_tensor(loads,2e-6)
    np.testing.assert_allclose(r['K_lu'],K)
    np.testing.assert_allclose(r['K_m2'],K*4e-12)
    assert r['reciprocity_error']==0 and abs(r['principal_values_lu'][0])<1e-14
    loads[0]['valid_for_permeability']=False
    assert not assemble_tensor(loads)['valid_for_permeability']


def test_oblique_tensor():
    y,x=np.indices((16,16));blocked=((x+y)%16)>=8
    r=compute_permeability_tensor(blocked,backend='numpy',verbose=False,n_steps_max=10000,conv_window=50,conv_tol=1e-7,return_fields=False)
    assert r['valid_for_permeability']
    assert r['K_lu'][0,1]<-.1 and r['reciprocity_error']<.01
    assert abs(r['principal_values_lu'][0])<1e-3
    rr=compute_permeability_tensor(blocked,force_magnitude=-1e-6,backend='numpy',verbose=False,n_steps_max=10000,conv_window=50,conv_tol=1e-7,return_fields=False)
    np.testing.assert_allclose(rr['K_lu'],r['K_lu'],rtol=1e-5,atol=1e-8)


def test_psd_disk_sphere_and_scaling():
    pytest.importorskip('scipy')
    for dim in (2,3):
        coords=np.indices((21,)*dim)-10
        blocked=np.sum(coords**2,axis=0)>=6**2
        r=local_thickness_distribution(blocked,return_map=True)
        # The central maximal open ball contains this exact digital pore.
        np.testing.assert_allclose(r['diameter_map'][~blocked],12)
        assert r['probability_mass'].sum()==pytest.approx(1)
        rr=local_thickness_distribution(blocked,voxel_size=2e-6,return_map=True)
        np.testing.assert_allclose(rr['diameter_map'],r['diameter_map']*2e-6)


def test_psd_slit_and_edges():
    m=np.ones((24,48),bool);m[8:16]=False
    r=local_thickness_distribution(m,return_map=True)
    np.testing.assert_allclose(r['diameter_map'][8:16,8:-8],8)
    assert r['diameter_map'][8,0]<8 # explicit finite-image wall at left edge
    assert local_thickness_distribution(np.ones((8,8),bool))['status']=='no_pores'
    assert local_thickness_distribution(np.zeros((8,8),bool))['pore_voxels']==64
    with pytest.raises(ValueError): local_thickness_distribution(m,boundary='periodic')
    with pytest.raises(ValueError): local_thickness_distribution(m,bins=[100,200])


def test_psd_independent_brute_force():
    from scipy.ndimage import distance_transform_edt
    m=np.random.default_rng(5).random((9,11))<.2
    p=np.pad(~m,1);dt=distance_transform_edt(p);truth=np.zeros(p.shape)
    grid=np.indices(p.shape)
    for c in np.argwhere(p):
        r=dt[tuple(c)];covered=np.sum((grid-c[:,None,None])**2,axis=0)<r*r-1e-12
        truth[covered&p]=np.maximum(truth[covered&p],2*r)
    r=local_thickness_distribution(m,return_map=True)
    np.testing.assert_allclose(r['diameter_map'],truth[1:-1,1:-1])
