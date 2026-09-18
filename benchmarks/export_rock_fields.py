"""Accepted 128³ velocity export and a slice plot; lattice-unit velocities."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lbm_permeability.solver import periodic
from lbm_permeability.io import save_result,write_json,provenance
p=argparse.ArgumentParser();p.add_argument('--output',default='results/2026-09-18-pilot/accepted_128_fields');a=p.parse_args()
out=Path(a.output)
if (out/'result.json').exists():raise RuntimeError('choose a new output directory')
root=Path('results/2026-09-18-pilot')
prior=json.loads((root/'convergence_review/rock_128_standard/result.json').read_text())
m=np.load(root/'dataset/blocked_128.npy');settings=dict(prior['settings']);force=settings.pop('force')
settings.update(backend='cuda',precision='float64',return_fields=True,verbose=False)
r=periodic(m,force,**settings)
t=time.perf_counter();save_result(out,r,dict(mask=provenance(m),voxel_size_m=5e-6,
 velocity_units='lattice dx/dt; physical time scale not supplied',density_units='lattice, reference density one',
 axes=['z','y','x'],components=['x','y','z'],crop_bounds_half_open=[[0,128]]*3))
export=time.perf_counter()-t
delta=abs(r['k_lu']/prior['k_lu']-1) if r['valid_for_permeability'] else None
write_json(out/'export_evidence.json',dict(solver_wall_s=r['elapsed_s'],file_export_wall_s=export,
 relative_k_change_vs_prior=delta,history_exactly_equal=r['convergence_history']==prior['convergence_history'],
 charged_seconds=r['elapsed_s'],accepted=r['valid_for_permeability']))
if not r['valid_for_permeability']:raise RuntimeError('no accepted field; see saved failure')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
speed=np.sqrt(sum(r['u'+c][64]**2 for c in 'xyz'))
fig,axs=plt.subplots(1,2,figsize=(10,4))
axs[0].imshow(m[64],origin='lower',cmap='gray_r',vmin=0,vmax=1);axs[0].set_title('Segmentation: dark = solid')
field=np.ma.masked_where(m[64],speed);cm=plt.get_cmap('magma').copy();cm.set_bad('.65')
im=axs[1].imshow(field,origin='lower',cmap=cm,vmin=0);axs[1].set_title('Accepted speed; grey = solid')
for ax in axs:ax.set(xlabel='x array index',ylabel='y array index')
fig.colorbar(im,ax=axs[1],label='lattice speed (dx/dt)')
fig.suptitle('Bentheimer origin 128³ crop, z index 64; periodic x load')
fig.tight_layout();fig.savefig(out/'velocity_slice.png',dpi=180);fig.savefig(out/'velocity_slice.pdf');plt.close(fig)
print(out,r['termination_reason'],r['iterations'],r['elapsed_s'],export,flush=True)
