"""Convergence and tensor figures from saved rock results only."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot')
out=root/'plots';out.mkdir(exist_ok=True)
for size in (128,256):
 paths=[root/f'convergence_review/rock_{size}_{label}/result.json' for label in ('standard','tight')]
 if not all(p.exists() for p in paths):continue
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for label,p in zip(('standard','100× tighter'),paths):
  r=json.loads(p.read_text());h=r['convergence_history'];s=r['settings']
  it=[x['iterations'] for x in h];k=[r['nu']*x['superficial_velocity'][0]/r['F_x'] for x in h]
  axs[0].plot(it,k,label=label)
  hs=[x for x in h if 'field_change_rms' in x]
  axs[1].semilogy([x['iterations'] for x in hs],[x['field_change_rms']/(s['conv_atol']+s['conv_tol']*x['velocity_rms']) for x in hs],label=label)
  axs[2].semilogy(it,np.maximum([x['mass_drift'] for x in h],1e-17),label=label)
 axs[0].set(ylabel='Kxx (lattice units²)',title=f'Bentheimer origin {size}³ crop')
 axs[1].axhline(1,color='k',ls='--',lw=.8);axs[1].set(ylabel='field change / allowed change',title='Field criterion (also requires vector checks)')
 axs[2].axhline(1e-8,color='k',ls='--',lw=.8);axs[2].set(ylabel='relative total population mass drift',title='Periodic conservation')
 for ax in axs:ax.set_xlabel('completed updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/f'rock_{size}_convergence.png',dpi=180);fig.savefig(out/f'rock_{size}_convergence.pdf');plt.close(fig)
for i in (0,1):
 path=root/f'long_rock/tensor_force_{i}/result.json'
 if not path.exists():continue
 r=json.loads(path.read_text())
 if not r['valid_for_permeability']:continue
 k=np.asarray(r['K_m2'])*1e12
 fig,ax=plt.subplots(figsize=(5.5,4.5));lim=np.max(abs(k));im=ax.imshow(k,cmap='coolwarm',vmin=-lim,vmax=lim)
 for y in range(3):
  for x in range(3):ax.text(x,y,f'{k[y,x]:.5f}',ha='center',va='center')
 ax.set(xticks=range(3),yticks=range(3),xticklabels=list('xyz'),yticklabels=list('xyz'),
        xlabel='load direction',ylabel='response direction',title=f'256³ crop raw tensor; force set {i}\npermeability (µm²), periodic application')
 fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f'bentheimer_tensor_force_{i}.png',dpi=180);fig.savefig(out/f'bentheimer_tensor_force_{i}.pdf');plt.close(fig)
print(out)
