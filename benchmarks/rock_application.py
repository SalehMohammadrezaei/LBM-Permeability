"""Preflight and execute predetermined Bentheimer periodic crops, sequential loads."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.solver import periodic
from porewise.tensor import assemble_tensor
from porewise.io import write_json,save_result,provenance
from porewise.backends import cp

root=Path('results/2026-09-18-pilot/dataset');out=Path('results/2026-09-18-pilot/rocks');out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'dataset_manifest.json').read_text());inspection=json.loads((root/'inspection.json').read_text())
assert inspection['segmentation_sha256']=='42bf2c7b771333d1a640afb7b5967b25cd04c78504f9648e457b75749ebe1b2a'
settings=dict(backend='cuda',precision='float64',tau=1.,conv_tol=1e-6,conv_atol=1e-12,
              conv_window=100,stability_every=50,n_steps_max=30000,wall_timeout_s=180.,
              return_fields=False,verbose=True,heartbeat=1000,characteristic_length=20.)
write_json(out/'configuration.json',dict(settings=settings,force_magnitude=1e-6,voxel_size_m=5e-6,
                                      boundary='periodic all axes',reference_status='application only; unknown reference BVP and units',
                                      reynolds_length_note='declared diagnostic length 20 cells=100 micrometres; not an independently measured hydraulic diameter'))
cost=0.;memory=[]
for n in (128,256):
    m=np.load(root/f'blocked_{n}.npy',allow_pickle=False)
    cp.get_default_memory_pool().free_all_blocks();free,total=cp.cuda.runtime.memGetInfo()
    estimate=m.size*(38*8+24*8+4)
    write_json(out/f'preflight_{n}.json',dict(shape=list(m.shape),free_bytes=free,total_bytes=total,
                                         conservative_estimate_bytes=estimate,headroom_fraction=.2,
                                         allowed=estimate<.8*free,load_order=['x','y','z']))
    if estimate>=.8*free:continue
    if n==128:
        s=dict(settings,n_steps_max=200,return_fields=True,wall_timeout_s=30.)
        t=time.perf_counter();r=periodic(m,[1e-6,0.,0.],**s);cost+=time.perf_counter()-t
        save_result(out/'smoke_128',r,dict(crop=inspection['crops']['128'],voxel_size_m=5e-6,role='loading and execution smoke only',parent_manifest='../dataset/dataset_manifest.json'))
        print('128 smoke',r['termination_reason'],r['elapsed_s'],flush=True)
    else:
        loads=[];t=time.perf_counter()
        for j in range(3):
            force=np.zeros(3);force[j]=1e-6
            r=periodic(m,force,**settings);loads.append(r)
            memory.append(r.get('memory',{}).get('gpu_pool_reserved_peak_sampled_bytes',0))
            save_result(out/f'256_load_{"xyz"[j]}',r,dict(crop=inspection['crops']['256'],voxel_size_m=5e-6,role='periodic application',geometry_provenance=provenance(m)))
            print('256',j,r['termination_reason'],r['elapsed_s'],r.get('k_lu'),flush=True)
        tensor=assemble_tensor(loads,5e-6);tensor['elapsed_s']=time.perf_counter()-t;cost+=tensor['elapsed_s']
        save_result(out/'tensor_256',tensor,dict(crop=inspection['crops']['256'],source_doi=manifest['doi'],reference_comparison='excluded: reference boundary conditions and physical units not recovered'))
cp.get_default_memory_pool().free_all_blocks();free,total=cp.cuda.runtime.memGetInfo()
extrapolated=max(memory,default=0)*(500/256)**3
write_json(out/'full_500_decision.json',dict(distribution_only_float64_bytes=38*500**3*8,
                                         free_bytes=free,observed_256_allocator_peak_bytes=max(memory,default=0),
                                         extrapolated_peak_bytes=extrapolated,headroom_fraction=.2,
                                         attempted=False,reason='float64 peak extrapolation exceeds 80% free VRAM; float32 application not qualified'))
write_json(out/'budget.json',dict(charged_simulation_wall_s=cost,maximum_directional_limits_s=570))
