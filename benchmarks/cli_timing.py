"""Whole-process input-to-accepted-export timing, including imports and JSON/CSV."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',default='results/2026-09-18-pilot/cli_timing');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
if (root/'summary.json').exists():raise RuntimeError('preserve existing timings')
m=np.ones((16,24),dtype=bool);m[4:12]=False
mask=root/'channel.npy';np.save(mask,m)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
records=[]
for backend in ('numpy','cupy-array','cuda'):
 for rep in range(4):
  directory=root/f'{backend}_{rep}'
  command=[sys.executable,'-m','porewise',str(mask),'--backend',backend,
   '--dx','1','--F','1e-6','--steps','15000','--tol','1e-6','--check-every','100',
   '--timeout','60','--output',str(directory)]
  before=os.getloadavg();t=time.perf_counter()
  completed=subprocess.run(command,env=env,capture_output=True,text=True)
  wall=time.perf_counter()-t
  r=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
  records.append(dict(backend=backend,repetition=rep,warmup=rep==0,command=command,
   process_wall_s=wall,exit_code=completed.returncode,valid=r.get('valid_for_permeability',False),
   iterations=r.get('iterations'),solver_elapsed_s=r.get('elapsed_s'),load_average=before,
   stdout=completed.stdout,stderr=completed.stderr))
  (root/'summary.json').write_text(json.dumps(dict(records=records,charged_seconds=sum(x['process_wall_s'] for x in records),
   timing_scope='new interpreter + imports/runtime probe + input read + setup + accepted solve + JSON/CSV export; no velocity fields requested; first repetition warms external compilation caches'),indent=2)+'\n')
  print(backend,rep,completed.returncode,wall,flush=True)
