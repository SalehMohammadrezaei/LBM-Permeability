"""Post-hoc tolerance/cost sensitivity from a tight run; does not relabel runs."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot/convergence_review')
results=[]
for size in (128,256):
 path=root/f'rock_{size}_tight/result.json'
 if not path.exists():continue
 r=json.loads(path.read_text())
 if not r['valid_for_permeability']:continue
 h=r['convergence_history'];target=np.array(h[-1]['superficial_velocity'])
 for tol in (1e-3,1e-4,1e-5,1e-6):
  streak=0;record=None
  for prev,cur in zip(h,h[1:]):
   u=np.array(cur['superficial_velocity']);v=np.array(prev['superficial_velocity'])
   passed=(cur['field_change_rms']<=1e-12+tol*cur['velocity_rms'] and
           np.all(abs(u-v)<=1e-12+tol*np.maximum(abs(u),abs(v))) and
           cur['mass_drift']<=r['settings']['mass_tol'])
   streak=streak+1 if passed else 0
   if streak>=3 and cur['iterations']>=300:
    error=float(np.linalg.norm(u-target)/np.linalg.norm(target))
    record=dict(shape=size,rtol=tol,atol=1e-12,first_three_pass_step=cur['iterations'],
     response_column_error_vs_tight=error,within_predeclared_point_one_percent=error<.001)
    break
  results.append(record or dict(shape=size,rtol=tol,first_three_pass_step=None))
(root/'replayed_tolerances.json').write_text(json.dumps(dict(
 note='Post-hoc sensitivity on recorded x-load trajectories only. Does not relax main campaign settings, create accepted solves, or qualify other loads/geometries. No timing speedup claimed from step ratio.',
 comparisons=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
