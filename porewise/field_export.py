"""Uncompressed VTK ImageData cell export; arrays (z,y,x), vectors (x,y,z).

Origin is the lower voxel-face corner in metres, not the first cell centre.
Velocities remain lattice dx/dt. Appended raw UInt64 blocks permit large files
without a second full-volume interleaved vector allocation.
"""
import struct
from pathlib import Path
import numpy as np
from .validation import mask, spacing


def write_vti(path, blocked, fields, voxel_size, origin=(0., 0., 0.)):
    blocked = mask(blocked, 3)
    dx = spacing(voxel_size)
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('origin must contain three finite Cartesian coordinates')
    for name in ('ux', 'uy', 'uz', 'rho'):
        if np.shape(fields[name]) != blocked.shape:
            raise ValueError('all cell fields must match the mask shape')
    nz, ny, nx = blocked.shape
    # Scalar solid mask, vector velocity, scalar density: little-endian binary.
    blocks = [('solid', 'UInt8', 1, blocked.size),
              ('velocity_lu', 'Float64', 3, blocked.size*24),
              ('density_lu', 'Float64', 1, blocked.size*8)]
    offset = 0; descriptions = []
    for name, dtype, components, count in blocks:
        descriptions.append(f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="appended" offset="{offset}"/>')
        offset += 8 + count
    extent = f'0 {nx} 0 {ny} 0 {nz}'
    header = ('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
              f'<ImageData WholeExtent="{extent}" Origin="{" ".join(map(str,origin))}" Spacing="{dx} {dx} {dx}">'
              f'<Piece Extent="{extent}"><PointData/><CellData Scalars="solid" Vectors="velocity_lu">'
              + ''.join(descriptions) + '</CellData></Piece></ImageData><AppendedData encoding="raw">_')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        stream.write(header.encode())
        stream.write(struct.pack('<Q', blocked.size))
        for slab in blocked: stream.write(slab.astype('u1').tobytes(order='C'))
        stream.write(struct.pack('<Q', blocked.size*24))
        for z in range(nz):
            vector = np.stack([fields['u'+c][z] for c in 'xyz'], axis=-1)
            stream.write(vector.astype('<f8', copy=False).tobytes(order='C'))
        stream.write(struct.pack('<Q', blocked.size*8))
        for slab in fields['rho']: stream.write(np.asarray(slab, dtype='<f8').tobytes(order='C'))
        stream.write(b'</AppendedData></VTKFile>\n')
    temporary.replace(path)
    return dict(format='VTK XML ImageData appended raw', association='cell',
                array_axes=['z','y','x'], vector_components=['x','y','z'],
                origin_lower_corner_m=origin.tolist(), spacing_m=dx,
                velocity_units='lattice dx/dt; no physical time scale supplied',
                density_units='lattice reference density one', solid_value=1, pore_value=0)
