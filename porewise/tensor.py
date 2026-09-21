"""Sequential independent loads; rows=response x,y,z, columns=load x,y,z."""
import time
import numpy as np
from .validation import mask, spacing, positive
from .solver import periodic


def assemble_tensor(loads, voxel_size=1.):
    dx=spacing(voxel_size)
    ndim=len(loads)
    if ndim not in (2,3): raise ValueError('two or three independent loads required')
    estimate=np.empty((ndim,ndim));valid=[]
    for j,r in enumerate(loads):
        force=np.array([r[f'F_{c}'] for c in 'xyz'[:ndim]])
        if force[j]==0 or np.count_nonzero(force)!=1:
            raise ValueError('column j must have its only nonzero force in component j')
        estimate[:,j]=r['nu']*np.array([r[f'u_{c}_mean_total'] for c in 'xyz'[:ndim]])/force[j]
        valid.append(bool(r.get('valid_for_permeability',False)) and bool(np.isfinite(estimate[:,j]).all()))
    raw=estimate.copy();raw[:,~np.array(valid)]=np.nan
    out=dict(K_lu=raw,K_m2=raw*dx**2,K_lu_diagnostic_estimate=estimate,
             valid_columns=valid,valid_for_permeability=all(valid),loads=loads,
             voxel_size_m=dx,array_axes=list('yx' if ndim==2 else 'zyx'),components=list('xyz'[:ndim]),
             convention='K_ij=nu*U_i(load j)/F_j, rho_ref=1, total original volume average',
             eigenvectors_are_columns=True,principal_basis='symmetric part (K+K.T)/2',
             reciprocity_error=None,principal_values_lu=None,principal_directions=None)
    if all(valid):
        symmetric=(raw+raw.T)/2
        out['symmetric_part_lu']=symmetric
        out['reciprocity_error']=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw),np.finfo(float).tiny))
        vals,vecs=np.linalg.eigh(symmetric)
        out.update(principal_values_lu=vals,principal_values_m2=vals*dx**2,principal_directions=vecs,
                   principal_warning='ascending eigenvalues; near-degenerate directions are not unique; no eigenvalue clipping')
    return out


def mirrored(blocked):
    """The image followed by its reflection along every axis (2**ndim times the volume).

    An image of a real sample is not periodic: across a periodic wrap most pore voxels face
    solid, which adds resistance.  In the mirrored domain every pore meets itself across each
    wrap and each mirror plane is a symmetry plane, which acts as a sealed, free-slip side wall.
    """
    out=mask(blocked)
    for axis in range(out.ndim):
        out=np.concatenate([out,np.flip(out,axis=axis)],axis=axis)
    return np.ascontiguousarray(out)


def compute_permeability_tensor(blocked, force_magnitude=1e-6, voxel_size=1., *, backend='auto', mirror=False, **settings):
    """Run all loads sequentially; invalid columns stay invalid and are never symmetrized away.

    mirror=True solves on the mirrored image.  That removes the wrap mismatch of a non-periodic
    sample, and by symmetry it also removes every off-diagonal response: the result holds the
    directional permeabilities of a sealed sample.  Use mirror=False for the full tensor.
    """
    blocked=mask(blocked);spacing(voxel_size)
    if mirror: blocked=mirrored(blocked)
    if not np.isfinite(force_magnitude) or force_magnitude==0: raise ValueError('force_magnitude must be finite and nonzero')
    started=time.perf_counter();loads=[]
    for j in range(blocked.ndim):
        force=np.zeros(blocked.ndim);force[j]=force_magnitude
        loads.append(periodic(blocked,force,backend=backend,**settings))
    out=assemble_tensor(loads,voxel_size)
    out['elapsed_s']=time.perf_counter()-started
    out['execution']='sequential independent rest initializations, all directional solves included'
    out['boundary']=('mirrored in every axis: sealed-sample directional permeabilities, off-diagonal terms vanish by symmetry'
                     if mirror else 'periodic wrap of the image as given: full tensor, biased low for non-periodic samples')
    return out
