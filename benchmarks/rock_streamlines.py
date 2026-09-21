"""Conservative visualization paths: arc-length Euler integration in voxel centres.

Trilinear interpolation is allowed only when all eight surrounding cells are pore.
Segments are checked every <=0.05 voxel; stop at mask/interpolation/display edges.
These unweighted paths are qualitative, never a flux-density representation.
"""
import itertools
import numpy as np


def velocity_at(point, fields, blocked):
    point=np.asarray(point);base=np.floor(point).astype(int)
    if np.any(base<0) or np.any(base+1>=np.array(blocked.shape)):return None
    fraction=point-base;velocity=np.zeros(3)
    for corner in itertools.product((0,1),repeat=3):
        index=tuple(base+corner)
        if blocked[index]:return None
        weight=np.prod(np.where(corner,fraction,1-fraction))
        velocity+=weight*np.array([fields['u'+c][index] for c in 'zyx'])
    return velocity


def integrate(seed,fields,blocked,step=.2,max_steps=4000):
    point=np.array(seed,dtype=float);path=[point.copy()];reason='maximum_steps'
    for _ in range(max_steps):
        v=velocity_at(point,fields,blocked)
        if v is None:reason='solid_stencil_or_display_boundary';break
        speed=np.linalg.norm(v)
        if speed<1e-18:reason='stagnation';break
        delta=step*v/speed
        # Check all subsegments, including endpoint, before retaining the step.
        if any(velocity_at(point+delta*a,fields,blocked) is None
               for a in np.linspace(.05/step,1,max(1,int(np.ceil(step/.05))))):
            reason='solid_stencil_or_display_boundary';break
        point=point+delta;path.append(point.copy())
    return np.asarray(path),reason
