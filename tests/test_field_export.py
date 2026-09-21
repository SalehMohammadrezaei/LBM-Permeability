import numpy as np
import pytest
from porewise.field_export import write_vti


def test_vti_noncubic_roundtrip(tmp_path):
    vtk = pytest.importorskip('vtk')
    from vtk.util.numpy_support import vtk_to_numpy
    z,y,x = np.indices((3,4,5)); mask=(x==2)&(y==1)
    fields=dict(ux=(100*z+10*y+x).astype(float),uy=np.full(mask.shape,-2.),
                uz=np.full(mask.shape,7.),rho=np.ones(mask.shape))
    path=tmp_path/'field.vti'
    metadata=write_vti(path,mask,fields,2e-6,origin=(1e-3,2e-3,3e-3))
    reader=vtk.vtkXMLImageDataReader();reader.SetFileName(str(path));reader.Update()
    image=reader.GetOutput()
    assert image.GetDimensions()==(6,5,4)
    assert image.GetNumberOfCells()==60 and image.GetPointData().GetNumberOfArrays()==0
    np.testing.assert_allclose(image.GetSpacing(),[2e-6]*3)
    np.testing.assert_allclose(image.GetOrigin(),[1e-3,2e-3,3e-3])
    vel=vtk_to_numpy(image.GetCellData().GetArray('velocity_lu')).reshape(3,4,5,3)
    for i,c in enumerate('xyz'):np.testing.assert_array_equal(vel[...,i],fields['u'+c])
    np.testing.assert_array_equal(vtk_to_numpy(image.GetCellData().GetArray('solid')).reshape(mask.shape),mask)
    np.testing.assert_allclose(image.GetCell(0).GetBounds(),[.001,.001002,.002,.002002,.003,.003002])
    assert metadata['association']=='cell'


def test_vti_bad_field_shape(tmp_path):
    with pytest.raises(ValueError,match='shape'):
        write_vti(tmp_path/'bad.vti',np.zeros((2,3,4),bool),{'ux':np.zeros((2,3,5))},1.)
