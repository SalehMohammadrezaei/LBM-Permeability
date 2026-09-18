"""Small CUDA-event profile of unchanged kernels plus shared diagnostic cost.

This is a fixed-step performance probe, not a permeability result.
"""
import sys,time,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lbm_permeability.backends import cp
from lbm_permeability.d3q19_fast import _module
from lbm_permeability.solver import macros,lattice
from lbm_permeability.diagnostics import Monitor
from lbm_permeability.io import write_json,provenance
root=Path('results/2026-09-18-pilot');m=np.load(root/'dataset/blocked_256.npy');shape=m.shape;n=m.size
cp.get_default_memory_pool().free_all_blocks()
bd=cp.asarray(m);solid=bd.astype(cp.uint8);fa=cp.empty((19,)+shape,cp.float64);fb=cp.empty_like(fa)
c,w,pairs=lattice(3)
for q in range(19):fa[q]=w[q]
mod=_module('float64');collide=mod.get_function('collide');stream=mod.get_function('stream')
blocks=((n+255)//256,);force=(1e-6,0.,0.)
ac=(solid,np.int64(n),*(np.float64(f) for f in force),np.float64(1.),np.float64(.5))
ast=(solid,np.int32(shape[2]),np.int32(shape[1]),np.int32(shape[0]))
def update():
 collide(blocks,(256,),(fa,fb)+ac);stream(blocks,(256,),(fb,fa)+ast)
t=time.perf_counter()
for _ in range(100):update()
cp.cuda.get_current_stream().synchronize();warm=time.perf_counter()-t
records=[];whole_start,whole_end=cp.cuda.Event(),cp.cuda.Event()
whole_start.record();t=time.perf_counter();real_start=time.time()
for _ in range(200):
 a,b,d=cp.cuda.Event(),cp.cuda.Event(),cp.cuda.Event()
 a.record();collide(blocks,(256,),(fa,fb)+ac);b.record();stream(blocks,(256,),(fb,fa)+ast);d.record();records.append((a,b,d))
whole_end.record();cp.cuda.get_current_stream().synchronize();wall=time.perf_counter()-t;real_wall=time.time()-real_start;whole_gpu_ms=cp.cuda.get_elapsed_time(whole_start,whole_end)
collide_ms=[cp.cuda.get_elapsed_time(a,b) for a,b,d in records];stream_ms=[cp.cuda.get_elapsed_time(b,d) for a,b,d in records]
mon=Monitor(cp,1e-6,1e-12,3,1e-8);mon.initial_mass=float(n);diagnostics=[]
for j in range(5):
 cp.cuda.get_current_stream().synchronize();tt=time.perf_counter()
 fields,rho=macros(fa,bd,force,c,cp)
 mon.check(300,fields,rho,fa,~bd,1/6,20.)
 cp.cuda.get_current_stream().synchronize();diagnostics.append(time.perf_counter()-tt)
out=dict(provenance=provenance(m),storage_dtype='float64',compute_dtype='float64',shape=list(shape),steps=200,
         warmup_steps=100,warmup_wall_s=warm,timed_wall_s=wall,time_time_wall_s=real_wall,whole_gpu_event_ms=whole_gpu_ms,event_to_wall_ratio=whole_gpu_ms/(1000*wall),collide_median_ms=float(np.median(collide_ms)),
         stream_median_ms=float(np.median(stream_ms)),collide_mean_ms=float(np.mean(collide_ms)),stream_mean_ms=float(np.mean(stream_ms)),
         full_diagnostic_repetitions_s=diagnostics,diagnostic_note='full field/moments, population validity and mass, field/vector monitor; repeated same state for cost only',
         all_steps_note='no convergence claim; profiling event overhead included in wall time',load_average=os.getloadavg(),
         total_charged_s=warm+wall+sum(diagnostics),gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
write_json(root/'rock_gpu_profile_clockcheck.json',out);print(json.dumps(out,indent=2))
