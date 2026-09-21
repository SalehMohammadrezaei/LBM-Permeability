"""Repeated accepted-permeability timings on matching non-cubic geometries."""
import argparse,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.solver import periodic
from porewise import geometry
from porewise.backends import cp
from porewise.io import save_result,write_json,provenance
p=argparse.ArgumentParser();p.add_argument('--output',default='results/2026-09-18-pilot/end_to_end');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if (out/'summary.json').exists():raise RuntimeError('choose a new output directory; do not overwrite timings')
rows=[];charged=0.
settings=dict(tau=1.,precision='float64',n_steps_max=15000,wall_timeout_s=60,
 conv_tol=1e-6,conv_atol=1e-12,conv_window=100,stability_every=50,
 characteristic_length=6.,return_fields=False,verbose=False)
for ndim in (2,3):
 m=(geometry.random_disks(20,24,8,3,seed=42) if ndim==2 else
    geometry.random_spheres(16,20,24,12,3,seed=42))
 f=np.zeros(ndim);f[0]=1e-6
 for b in ('numpy','cupy-array','cuda'):
  reps=[]
  for rep in range(4):
   if b!='numpy':cp.get_default_memory_pool().free_all_blocks();cp.cuda.get_current_stream().synchronize()
   pre=dict(load_average=os.getloadavg(),gpu_free_total_bytes=cp.cuda.runtime.memGetInfo())
   t=time.perf_counter();r=periodic(m,f,backend=b,**settings)
   if b!='numpy':cp.cuda.get_current_stream().synchronize()
   wall=time.perf_counter()-t;charged+=wall
   r['synchronized_public_call_s']=wall
   save_result(out/f'{ndim}d_{b}_{rep}',r,dict(warmup=rep==0,mask=provenance(m),resources_before=pre,
    timing_scope='setup, solve, diagnostics; no fields requested; evidence serialization timed separately'))
   export_s=time.perf_counter()-t-wall
   reps.append(dict(warmup=rep==0,valid=r['valid_for_permeability'],status=r['termination_reason'],
    iterations=r['iterations'],k_lu=r['k_lu'],wall_s=wall,evidence_export_s=export_s,timing=r['timing'],memory=r['memory']))
   print(ndim,b,rep,r['termination_reason'],wall,flush=True)
  times=[x['wall_s'] for x in reps[1:]]
  rows.append(dict(ndim=ndim,shape=list(m.shape),backend=b,storage_dtype='float64',compute_dtype='float64',
   accepted_repetitions=all(x['valid'] for x in reps[1:]),repetitions=reps,
   wall_median_s=float(np.median(times)),wall_min_s=min(times),wall_max_s=max(times)))
write_json(out/'summary.json',dict(cases=rows,settings=settings,charged_seconds=charged,
 note='three measured repetitions after one complete warmup; independent rest initializations; no claim that NumPy uses 96 cores'))
