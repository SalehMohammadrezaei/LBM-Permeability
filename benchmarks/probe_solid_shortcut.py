"""Experimental solid shortcut or integer periodic wrapping; no solver edits.

The existing collision skips relaxation in solids but still forms intermediates.
This probe checks whether an explicit early return saves work, with equal states.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.backends import cp
from porewise.d3q19_fast import _src
from porewise.d3q19 import W
from porewise.io import write_json,provenance
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['solid','wrap'],default='solid')
p.add_argument('--output',default='results/2026-09-18-pilot/convergence_review');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
target=root/('solid_shortcut_probe.json' if a.kind=='solid' else 'stream_wrap_probe.json')
if target.exists():raise RuntimeError('choose a new output directory; preserve existing evidence')
records=[];charged=0.
for precision,shape in [('float64',(8,10,12)),('float32',(8,10,12)),('float64',(256,256,256))]:
 real='double' if precision=='float64' else 'float'
 source=_src(real)
 shortcut=f'''    if (solid[i]) {{
        #pragma unroll
        for (int q=0;q<19;q++) fo[(int64_t)q*N+i]=({real})fq[q];
        return;
    }}
'''
 edited=source.replace('    double rho = 0.0;',shortcut+'    double rho = 0.0;',1)
 if a.kind=='wrap':
  edited='__device__ __forceinline__ int wrap_one(int a, int n) { return a<0 ? a+n : (a>=n ? a-n : a); }\n'+source
  for axis,component in [('x','X'),('y','Y'),('z','Z')]:
   for sign in ('+','-'):
    old=f'({axis} {sign} C{component}c[q] + N{axis}) % N{axis}'
    new=f'wrap_one({axis} {sign} C{component}c[q], N{axis})'
    assert old in edited
    edited=edited.replace(old,new)
 assert edited!=source
 m=(np.load('results/2026-09-18-pilot/dataset/blocked_256.npy') if shape[0]==256 else
    np.random.default_rng(17).random(shape)<.4)
 solid=cp.asarray(m,dtype=cp.uint8);n=m.size;blocks=((n+255)//256,)
 ac=(solid,np.int64(n),np.float64(1e-4),np.float64(-2e-5),np.float64(3e-5),np.float64(1.),np.float64(.5))
 ast=(solid,*(np.int32(k) for k in shape[::-1]))
 mods=[cp.RawModule(code=s,options=()) for s in (source,edited)]
 kernels=[(mod.get_function('collide'),mod.get_function('stream')) for mod in mods]
 states=[(cp.empty((19,)+shape,dtype=precision),cp.empty((19,)+shape,dtype=precision)) for _ in range(2)]
 def initialize(pair):
  for q in range(19):pair[0][q]=W[q]
 steps=200 if shape[0]==256 else 100
 reps=[]
 for rep in range(4):
  for label,(f,fb),(collide,stream) in zip(('before','shortcut'),states,kernels):
   initialize((f,fb));cp.cuda.get_current_stream().synchronize();t=time.perf_counter()
   for _ in range(steps):
    collide(blocks,(256,),(f,fb)+ac);stream(blocks,(256,),(fb,f)+ast)
   cp.cuda.get_current_stream().synchronize();wall=time.perf_counter()-t;charged+=wall
   reps.append(dict(variant=label,repetition=rep,warmup=rep==0,wall_s=wall))
  err=float(cp.max(cp.abs(states[0][0]-states[1][0])))
  print(precision,shape,rep,'max population difference',err,flush=True)
  if err!=0:print('NOT bitwise identical',flush=True)
 records.append(dict(shape=shape,precision=precision,compute_dtype='float64',steps=steps,repetitions=reps,
  max_population_difference=err,source_before_sha256=hashlib.sha256(source.encode()).hexdigest(),
  source_shortcut_sha256=hashlib.sha256(edited.encode()).hexdigest()))
 del states,solid,kernels,mods
 cp.get_default_memory_pool().free_all_blocks()
write_json(target,dict(provenance=provenance(),experiment=a.kind,cases=records,charged_seconds=charged,
 note='experimental generated kernel only; synchronized fixed-step loop excludes initialization, diagnostics and exports; no accepted permeability or end-to-end speedup claim'))
