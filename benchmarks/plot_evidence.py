"""Figures from archived outputs only; no simulations or invented data."""
import json,sys,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot')
out=root/'plots';out.mkdir(exist_ok=True)
summary=json.loads((root/'performance/summary.json').read_text());rows=[]
for r in summary['cases']:
 if r.get('status')!='measured_fixed_steps':continue
 reps=r['repetitions'];times=np.array([x['wall_s'] for x in reps]);steps=r['timed_steps'];size=int(np.prod(r['shape']))
 # Main throughput uses synchronized public-call wall time, including final checks.
 lups=steps*size/times/1e6
 rows.append(dict(case=r['id'],ndim=len(r['shape']),n=r['shape'][0],backend=r['backend'],precision=r['precision'],
                  steps=steps,wall_median=float(np.median(times)),wall_min=float(times.min()),wall_max=float(times.max()),
                  public_call_mlups_median=float(np.median(lups)),public_call_mlups_min=float(lups.min()),public_call_mlups_max=float(lups.max()),
                  max_sampled_gpu_pool_bytes=max(x['memory']['gpu_pool_reserved_peak_sampled_bytes'] for x in reps)))
with (out/'performance_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for dim,ax in zip((2,3),axs):
 for backend,precision in [('numpy','float64'),('cupy-array','float64'),('cuda','float64'),('cuda','float32')]:
  rr=sorted([r for r in rows if r['ndim']==dim and r['backend']==backend and r['precision']==precision],key=lambda r:r['n'])
  label=backend+' '+precision+(' storage; f64 compute' if precision=='float32' else '')
  ax.plot([r['n'] for r in rr],[r['public_call_mlups_median'] for r in rr],'o-',label=label)
  ax.fill_between([r['n'] for r in rr],[r['public_call_mlups_min'] for r in rr],[r['public_call_mlups_max'] for r in rr],alpha=.12)
 ax.set(xscale='log',yscale='log',xlabel='side length (cells)',ylabel='MLUPS, synchronized public-call time',title=f'{dim}D fixed-step throughput')
 ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'throughput.png',dpi=180);plt.close(fig)
psd=root/'dataset/morphology_64/result.json'
if psd.exists():
 r=json.loads(psd.read_text());fig,ax=plt.subplots(figsize=(6,4));edges=np.array(r['bin_edges'])*1e6
 ax.stairs(r['probability_mass'],edges,fill=True);ax.set(xlabel='local-thickness diameter (micrometres)',ylabel='pore-volume probability mass',title='Bentheimer 64³ ROI; finite solid exterior')
 fig.tight_layout();fig.savefig(out/'psd.png',dpi=180);plt.close(fig)
for target,name in [('verification/laminate3_48/result.json','laminate_tensor'),('rocks/tensor_256/result.json','bentheimer_tensor')]:
 p=root/target
 if not p.exists():continue
 r=json.loads(p.read_text())
 if not r['valid_for_permeability']:continue
 k=np.array(r['K_lu']);fig,ax=plt.subplots(figsize=(5,4));lim=np.max(abs(k));im=ax.imshow(k,cmap='coolwarm',vmin=-lim,vmax=lim)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{k[i,j]:.4g}',ha='center',va='center')
 ax.set(xticks=range(3),yticks=range(3),xticklabels=list('xyz'),yticklabels=list('xyz'),xlabel='load direction',ylabel='response direction',title=name.replace('_',' ')+' (LU²)')
 fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
pressure=[root/f'verification/pressure_channel_pad{p}/result.json' for p in (0,4,8,16)]
if all(p.exists() for p in pressure):
 fig,axs=plt.subplots(1,3,figsize=(13,4));pads=[];ks=[]
 for p in pressure:
  r=json.loads(p.read_text());pad=r['pad'];pads.append(pad);ks.append(r['k_lu'])
  profile=np.array(r['pressure_profile']);flux=np.array(r['mass_flux_profile']);x=np.arange(len(profile))-pad
  axs[1].plot(x,(profile-profile[pad])*1e6,label=f'pad={pad}')
  axs[2].plot(x,(flux/flux.mean()-1)*1e6,label=f'pad={pad}')
 axs[0].plot(pads,ks,'o-');axs[0].axhline(6,color='k',ls='--',label='fully developed continuum limit')
 axs[0].set(xlabel='reservoir padding (cells per side)',ylabel='Kxx (lattice units²)',title='Padding changes the boundary problem')
 axs[1].set(xlabel='x index relative to original inlet',ylabel='p − p(original inlet), LU × 10⁻⁶',title='Fluid-mean pressure profile')
 axs[2].set(xlabel='x index relative to original inlet',ylabel='mass-flux deviation from mean (ppm)',title='Section flux conservation')
 for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'pressure_diagnostics.png',dpi=180);plt.close(fig)
e2e=root/'end_to_end/summary.json'
if e2e.exists():
 data=json.loads(e2e.read_text());fig,axs=plt.subplots(1,2,figsize=(10,4))
 for dim,ax in zip((2,3),axs):
  rr=[r for r in data['cases'] if r['ndim']==dim and r['accepted_repetitions']]
  for x,r in enumerate(rr):
   times=[rep['wall_s'] for rep in r['repetitions'] if not rep['warmup']]
   ax.scatter([x]*len(times),times,s=20)
   ax.plot([x-.15,x+.15],[np.median(times)]*2,'k-',lw=2)
  ax.set(xticks=range(len(rr)),xticklabels=[r['backend'] for r in rr],yscale='log',
   ylabel='synchronized public-call time (s)',title=f'{dim}D accepted permeability; f64 storage/compute')
  ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(out/'accepted_permeability_timing.png',dpi=180);plt.close(fig)
print(out)
