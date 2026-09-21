"""Installed CLI: python -m porewise MASK.npy --dx METRES --output DIR."""
import argparse
import sys
import numpy as np
from . import geometry
from .validation import decode_mask,mask,spacing
from .solver import periodic
from .tensor import compute_permeability_tensor
from .io import save_result,provenance,write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mask_npy',nargs='?')
    p.add_argument('--demo',action='store_true');p.add_argument('--dimension',type=int,choices=[2,3],default=2)
    p.add_argument('--direction',choices=['x','y','z'],default='x');p.add_argument('--tensor',action='store_true');p.add_argument('--mirror',action='store_true',help='with --tensor: solve on the mirrored image (sealed-sample directional permeabilities)')
    p.add_argument('--backend',choices=['auto','numpy','cupy-array','cuda','cuda-sparse','numba-sparse'],default='auto')
    p.add_argument('--no-gpu',action='store_true');p.add_argument('--precision',choices=['float64','float32'],default='float64')
    p.add_argument('--F',type=float,default=1e-6);p.add_argument('--tau',type=float,default=1.)
    p.add_argument('--collision',choices=['bgk','trt'],default='bgk');p.add_argument('--magic',type=float)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--tol',type=float,default=1e-6)
    p.add_argument('--check-every',type=int,default=100);p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--dx',type=float,required=True);p.add_argument('--output',default='lbm-output')
    p.add_argument('--solid-value',type=float);p.add_argument('--pore-value',type=float)
    p.add_argument('--fields',action='store_true')
    a=p.parse_args(argv)
    if not a.mask_npy and not a.demo:p.error('provide a mask or explicitly select --demo')
    try:
        spacing(a.dx)
        if a.demo:
            b=geometry.parallel_plates(24,16,12)
            if a.dimension==3:b=np.broadcast_to(b,(8,)+b.shape).copy()
        else:
            image=np.load(a.mask_npy,allow_pickle=False)
            if (a.solid_value is None)!=(a.pore_value is None):raise ValueError('supply both binary labels')
            b=mask(image) if a.solid_value is None else decode_mask(image,solid_value=a.solid_value,pore_value=a.pore_value)
        settings=dict(backend='numpy' if a.no_gpu else a.backend,precision=a.precision,tau=a.tau,collision=a.collision,magic=a.magic,
                      n_steps_max=a.steps,conv_tol=a.tol,conv_window=a.check_every,
                      wall_timeout_s=a.timeout,return_fields=a.fields,verbose=False)
        if a.tensor:
            r=compute_permeability_tensor(b,force_magnitude=a.F,voxel_size=a.dx,mirror=a.mirror,**settings)
        else:
            j='xyz'.index(a.direction)
            if j>=b.ndim:raise ValueError('direction exceeds mask dimension')
            force=np.zeros(b.ndim);force[j]=a.F
            r=periodic(b,force,**settings)
            r['k_m2']=r['k_lu']*a.dx**2 if r['k_lu'] is not None else None
        save_result(a.output,r,dict(voxel_size_m=a.dx,geometry_provenance=provenance(b)))
        valid=r['valid_for_permeability']
        print(('Accepted permeability' if valid else 'No accepted permeability')+f'; diagnostics saved in {a.output}')
        return 0 if valid else 2
    except (ValueError,RuntimeError,MemoryError,OSError) as e:
        write_json(a.output+'/error.json',{'error':str(e),'provenance':provenance()})
        print(str(e),file=sys.stderr);return 2

if __name__=='__main__': sys.exit(main())
