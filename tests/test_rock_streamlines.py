import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('rock_streamlines',Path(__file__).resolve().parents[1]/'benchmarks/rock_streamlines.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_paths_stop_at_solid_and_display_boundary():
    mask=np.zeros((5,6,12),bool)
    fields=dict(ux=np.ones(mask.shape),uy=np.zeros(mask.shape),uz=np.zeros(mask.shape))
    path,why=module.integrate((2,2,1),fields,mask)
    assert why=='solid_stencil_or_display_boundary'
    assert 10.7<path[-1,2]<11
    np.testing.assert_allclose(path[:,:2],np.broadcast_to([2,2],path[:,:2].shape))
    mask[:,:,6]=True
    path,why=module.integrate((2,2,1),fields,mask)
    assert path[:,2].max()<5
    assert why=='solid_stencil_or_display_boundary'


def test_interpolation_rejects_near_wall_stencil():
    mask=np.zeros((4,4,4),bool);mask[2,2,2]=True
    fields={k:np.ones(mask.shape) for k in ('ux','uy','uz')}
    assert module.velocity_at([1.1,1.1,1.1],fields,mask) is None
