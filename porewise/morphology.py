"""Optional SciPy voxel-centred local thickness, with explicit exterior walls.

For every pore centre c, r(c) is its Euclidean distance to a solid centre.
The open ball |x-c| < r(c) lies in the pore phase. At x return twice the
largest r(c) among balls containing x. This is a covering-ball calculation,
not a nearest-wall histogram. Radius sweep uses every distinct EDT radius.
Finite crops have one explicit solid exterior layer; periodic is unsupported.
"""
import time
import numpy as np
from .validation import mask, spacing, integer
from .geometry import porosity


def local_thickness_distribution(blocked, voxel_size=1., bins=30, *, boundary='solid-exterior',
                                 return_map=False, max_voxels=2_100_000):
    started=time.perf_counter();blocked=mask(blocked);dx=spacing(voxel_size)
    integer('max_voxels',max_voxels)
    if boundary!='solid-exterior':
        raise ValueError('only finite solid-exterior morphology is supported; periodic flow does not imply periodic PSD')
    if blocked.size>max_voxels: raise ValueError('morphology ROI exceeds max_voxels; use a documented smaller ROI')
    try:
        import scipy
        from scipy.ndimage import distance_transform_edt
    except ImportError as e: raise ImportError('install porewise[morphology]') from e
    pore=np.pad(~blocked,1,constant_values=False)
    dt=distance_transform_edt(pore)
    radii=np.unique(dt[dt>0])[::-1]
    thickness=np.zeros(pore.shape,np.float64)
    for radius in radii:
        centres=dt>=radius
        # EDT to eligible centres supplies the union of all radius-r open balls.
        covered=distance_transform_edt(~centres)<radius
        thickness[(thickness==0)&covered&pore]=2*radius*dx
    thickness=thickness[tuple(slice(1,-1) for _ in blocked.shape)]
    diameters=thickness[~blocked]
    if diameters.size and np.any(diameters<=0): raise RuntimeError('local thickness failed to cover all pore voxels')
    if np.isscalar(bins): integer('bins',bins)
    else:
        bins=np.asarray(bins,dtype=float)
        if bins.ndim!=1 or len(bins)<2 or not np.isfinite(bins).all() or not np.all(np.diff(bins)>0):
            raise ValueError('bin edges must be finite and strictly increasing')
        if diameters.size and (bins[0]>diameters.min() or bins[-1]<diameters.max()):
            raise ValueError('bin edges must include every pore diameter')
    counts,edges=np.histogram(diameters,bins=bins)
    mass=counts/diameters.size if diameters.size else np.zeros_like(counts,dtype=float)
    out=dict(algorithm='voxel-centred maximal covering open balls; exhaustive distinct EDT-radius sweep',
             dependency='scipy',dependency_version=scipy.__version__,boundary=boundary,
             boundary_note='exterior is solid; crop-conditioned descriptor with artificial exterior walls',
             radius_definition='distance between pore and nearest solid voxel centres; diameter=2*radius',
             weighting='equal pore-voxel volume, including isolated pores',units='m',voxel_size_m=dx,
             porosity=porosity(blocked),pore_voxels=int(diameters.size),shape=list(blocked.shape),
             status='ok' if diameters.size else 'no_pores',number_of_radii=len(radii),
             bin_edges=edges,bin_centres=(edges[1:]+edges[:-1])/2,probability_mass=mass,
             pdf_per_m=mass/np.diff(edges),cumulative_undersize=np.cumsum(mass),
             percentiles_m=dict(zip(('p10','p50','p90'),np.percentile(diameters,[10,50,90]))) if diameters.size else None,
             elapsed_s=time.perf_counter()-started)
    if return_map: out['diameter_map']=thickness
    return out
