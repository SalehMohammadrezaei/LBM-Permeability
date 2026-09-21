import numpy as np
import pytest
from porewise.topology import connectivity_report


def test_periodic_winding_not_face_contact():
    m=np.ones((8,8),bool);m[3,0]=m[3,-1]=False
    r=connectivity_report(m,boundary='periodic')
    assert r['components']==1 and r['winding_porosity_by_array_axis']==[0.,0.]
    m[3,:]=False
    r=connectivity_report(m,boundary='periodic')
    assert r['winding_porosity_by_array_axis']==[0.,1/8]


def test_diagonal_transport_links():
    m=~np.eye(8,dtype=bool)
    face=connectivity_report(m,boundary='periodic',connectivity='face')
    links=connectivity_report(m,boundary='periodic',connectivity='lattice')
    assert face['components']==8
    assert face['winding_porosity_by_array_axis']==[0.,0.]
    assert links['winding_porosity_by_array_axis']==[1/8,1/8]


def test_finite_closed_and_open():
    m=np.ones((12,16),bool);m[4:8,:]=False
    r=connectivity_report(m)
    assert r['spanning_porosity_by_array_axis']==[0.,1/3]
    m[:,0]=m[:,-1]=True
    r=connectivity_report(m)
    assert r['spanning_porosity_by_array_axis']==[0.,0.]
    assert r['interior_component_pore_fraction']==pytest.approx(56/192)
