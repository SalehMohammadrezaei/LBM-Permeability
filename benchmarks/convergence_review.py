"""Recorded before/after overhead and tolerance checks; one GPU process only."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lbm_permeability.solver import periodic
from lbm_permeability.backends import cp
from lbm_permeability.io import save_result,write_json,provenance

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['performance','convergence'],required=True)
p.add_argument('--size',type=int,default=128)
p.add_argument('--output',default='results/2026-09-18-pilot/convergence_review')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
base=Path('results/2026-09-18-pilot')
m=np.load(base/f'dataset/blocked_{a.size}.npy',allow_pickle=False)
s=dict(backend='cuda',precision='float64',tau=1.,conv_tol=1e-6,conv_atol=1e-12,
       conv_window=100,stability_every=50,consecutive=3,characteristic_length=20.,
       return_fields=False,verbose=True,heartbeat=2000,n_steps_max=60000,wall_timeout_s=1200)
meta=dict(provenance=provenance(m),settings=s,shape=list(m.shape),load_average=os.getloadavg())
if a.mode=='performance':
 spec=importlib.util.spec_from_file_location('lbm_permeability._review_baseline',out/'solver_before.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 s.update(n_steps_max=500,verbose=False)
 records=[];reference=None
 for i,label in enumerate(['before','after']*4):
  cp.get_default_memory_pool().free_all_blocks();cp.cuda.get_current_stream().synchronize()
  t=time.perf_counter();r=(old.periodic if label=='before' else periodic)(m,(1e-6,0,0),**s)
  cp.cuda.get_current_stream().synchronize();elapsed=time.perf_counter()-t
  hist=r['convergence_history']
  if reference is None:reference=hist
  identical=hist==reference
  records.append(dict(implementation=label,repetition=i//2,warmup=i<2,elapsed_s=elapsed,
                      history_exactly_equal=identical,timing=r['timing'],memory=r['memory']))
  print(label,elapsed,'identical diagnostics:',identical,flush=True)
 write_json(out/'overhead_comparison.json',dict(metadata=meta,records=records,
  charged_seconds=sum(r['elapsed_s'] for r in records),iterations_per_call=500,
  note='paired interleaved synchronized whole-call timing; first pair is warmup'))
else:
 runs=[]
 for label,tol,atol in [('standard',1e-6,1e-12),('tight',1e-8,1e-14)]:
  cp.get_default_memory_pool().free_all_blocks();s.update(conv_tol=tol,conv_atol=atol)
  r=periodic(m,(1e-6,0,0),**s);runs.append(r)
  save_result(out/f'rock_{a.size}_{label}',r,dict(mask_provenance=provenance(m),voxel_size_m=5e-6))
 valid=all(r['valid_for_permeability'] for r in runs)
 columns=[np.array([r[f'u_{c}_mean_total'] for c in 'xyz'])*r['nu']/1e-6 for r in runs]
 err=float(np.linalg.norm(columns[0]-columns[1])/np.linalg.norm(columns[1])) if valid else None
 write_json(out/f'tolerance_{a.size}.json',dict(valid_pair=valid,column_relative_error=err,
  threshold=.001,passed=valid and err<.001,raw_columns=[x.tolist() for x in columns],
  charged_seconds=sum(r['elapsed_s'] for r in runs),note='independent rest starts; relative and absolute tolerances both tightened 100-fold'))
