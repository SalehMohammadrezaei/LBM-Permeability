"""Sequential synchronized timings of public solvers, including diagnostics.

No kernel-only claim. Same masks/precision/steps/settings per paired comparison.
"""
import argparse,sys,time,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lbm_permeability.solver import periodic
from lbm_permeability.backends import cp,HAS_GPU
from lbm_permeability.io import write_json,provenance


def mask(shape):
    coordinates=np.ogrid[tuple(slice(0,n) for n in shape)]
    # Deterministic repeating resolved solid spheres/disks; same fraction across sizes.
    return sum(((x%16)-7.5)**2 for x in coordinates)<=4**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/2026-09-18-pilot/performance');p.add_argument('--budget',type=float,default=280);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    trials=[];used=0.
    geometries=[(128,128),(256,256),(512,512),(1024,1024),(32,32,32),(64,64,64),(128,128,128),(256,256,256)]
    for shape in geometries:
        bs=['numpy','cupy-array','cuda'] if np.prod(shape)<=64**3 else ['cuda']
        if len(shape)==3 and shape[0]==64:bs=['cuda']
        m=mask(shape)
        for b in bs:
            for precision in (['float64','float32'] if b=='cuda' else ['float64']):
                cid='x'.join(map(str,shape))+'_'+b+'_'+precision
                if used>=a.budget:
                    trials.append(dict(id=cid,status='deferred_budget'));continue
                if HAS_GPU:cp.get_default_memory_pool().free_all_blocks()
                free=cp.cuda.runtime.memGetInfo()[0] if HAS_GPU else 0
                estimated=np.prod(shape)*(38*np.dtype(precision).itemsize+8*24+4)
                if b!='numpy' and estimated>.8*free:
                    trials.append(dict(id=cid,status='excluded_memory_preflight',estimated_bytes=int(estimated),gpu_free_bytes=free));continue
                steps=20 if np.prod(shape)>=128**3 else 100
                warm=10 if steps==20 else 100
                settings=dict(tau=1.,backend=b,precision=precision,verbose=False,return_fields=False,
                              conv_window=50,stability_every=50,conv_tol=1e-12,conv_atol=1e-15,
                              wall_timeout_s=min(30.,a.budget-used))
                force=np.zeros(m.ndim);force[0]=1e-6
                r=dict(id=cid,shape=list(shape),porosity=float((~m).mean()),backend=b,precision=precision,
                       compute_dtype='float64',timed_steps=steps,warmup_steps=warm,
                       settings=settings,mask_sha256=provenance(m)['mask_sha256_c_order_bool'],
                       load_average=os.getloadavg(),repetitions=[])
                t=time.perf_counter()
                try:
                    cold=periodic(m,force,n_steps_max=warm,**settings)
                    r['cold_warmup_wall_s']=cold['elapsed_s']
                    for repeat in range(5):
                        if used+time.perf_counter()-t>=a.budget:break
                        if b!='numpy':cp.cuda.get_current_stream().synchronize()
                        tt=time.perf_counter();run=periodic(m,force,n_steps_max=steps,**settings)
                        if b!='numpy':cp.cuda.get_current_stream().synchronize()
                        wall=time.perf_counter()-tt
                        count=run['iterations_completed']
                        r['repetitions'].append(dict(index=repeat,wall_s=wall,timing=run['timing'],memory=run['memory'],
                                                     iterations_completed=count,status=run['termination_reason'],
                                                     mlups_total_lattice=count*m.size/run['timing']['solve_and_diagnostics_s']/1e6,
                                                     mlups_fluid_nodes=count*np.count_nonzero(~m)/run['timing']['solve_and_diagnostics_s']/1e6,
                                                     diagnostics=run['diagnostics']))
                    valid=[rr for rr in r['repetitions'] if rr['iterations_completed']==steps and rr['status']=='max_steps']
                    r['status']='measured_fixed_steps' if len(valid)==5 else 'incomplete_repetitions_or_steps'
                    if valid:
                        times=[rr['wall_s'] for rr in valid]
                        r.update(wall_median_s=float(np.median(times)),wall_min_s=min(times),wall_max_s=max(times),wall_std_s=float(np.std(times)))
                except Exception as e:r.update(status='exception',error=repr(e))
                cost=time.perf_counter()-t;used+=cost;r['charged_wall_s']=cost
                r['provenance']=provenance();write_json(root/(cid+'.json'),r);trials.append(r)
                print(cid,r['status'],f'{cost:.2f}s',flush=True)
    write_json(root/'summary.json',dict(cases=trials,charged_wall_s=used,budget_s=a.budget,
                                      note='Fixed-step throughput is not time to accepted permeability; see verification and tensor logs for end-to-end results.'))

if __name__=='__main__':main()
