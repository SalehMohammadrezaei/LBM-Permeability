"""Explicit geometric connectivity without modifying the mask or averaging volume."""
from collections import deque
import itertools
import numpy as np
from .validation import mask, integer


def connectivity_report(blocked, *, connectivity='face', boundary='finite', max_periodic_voxels=250000):
    """Geometric finite spanning or periodic winding, independent of flow acceptance.

    face: 4/6 neighbours; lattice: D2Q9's 8 or D3Q19's 18 links. Periodic
    winding requires a noncontractible graph cycle; opposite-face contact alone
    is insufficient. This small-image graph diagnostic is intentionally bounded.
    """
    b=mask(blocked);pore=~b;dim=b.ndim
    if connectivity not in ('face','lattice'): raise ValueError('connectivity must be face or lattice')
    if boundary not in ('finite','periodic'): raise ValueError('boundary must be finite or periodic')
    integer('max_periodic_voxels',max_periodic_voxels)
    steps=[d for d in itertools.product((-1,0,1),repeat=dim) if 0<sum(x*x for x in d)<= (1 if connectivity=='face' else 2)]
    if boundary=='finite':
        from scipy.ndimage import label,generate_binary_structure
        labels,n=label(pore,generate_binary_structure(dim,1 if connectivity=='face' else 2))
        sizes=np.bincount(labels.ravel());through=[]
        for axis in range(dim):
            lo=set(np.take(labels,0,axis=axis).ravel())-{0}
            hi=set(np.take(labels,-1,axis=axis).ravel())-{0}
            through.append(float(sum(sizes[i] for i in lo&hi)/b.size))
        touches=set()
        for axis in range(dim):
            touches.update(np.take(labels,0,axis=axis).ravel());touches.update(np.take(labels,-1,axis=axis).ravel())
        interior=sum(sizes[i] for i in range(1,n+1) if i not in touches)
        return dict(porosity=float(pore.mean()),components=n,connectivity=connectivity,boundary=boundary,
                    array_axes=list('yx' if dim==2 else 'zyx'),spanning_porosity_by_array_axis=through,
                    interior_component_pore_fraction=float(interior/b.size),denominator='entire original sample; no pore removal')
    if b.size>max_periodic_voxels: raise ValueError('periodic winding graph exceeds bounded small-image diagnostic limit')
    visited=np.zeros(b.shape,bool);lift=np.zeros(b.shape+(dim,),np.int64);components=[]
    shape=np.array(b.shape)
    for start in np.argwhere(pore):
        st=tuple(start)
        if visited[st]:continue
        visited[st]=True;lift[st]=start;todo=deque([st]);count=0;winding=np.zeros(dim,bool)
        while todo:
            pos=todo.popleft();count+=1
            for d in steps:
                nxt=tuple((np.array(pos)+d)%shape)
                if not pore[nxt]:continue
                unwrapped=lift[pos]+d
                if visited[nxt]:winding |= unwrapped!=lift[nxt]
                else:
                    visited[nxt]=True;lift[nxt]=unwrapped;todo.append(nxt)
        components.append(dict(pore_voxels=count,winding_by_array_axis=winding.tolist()))
    winding_fraction=[sum(c['pore_voxels'] for c in components if c['winding_by_array_axis'][j])/b.size for j in range(dim)]
    return dict(porosity=float(pore.mean()),components=len(components),component_details=components,
                connectivity=connectivity,boundary=boundary,array_axes=list('yx' if dim==2 else 'zyx'),
                winding_porosity_by_array_axis=winding_fraction,denominator='entire original sample; no pore removal')
