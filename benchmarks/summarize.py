"""Generate numerical comparison tables and diagnostics from saved results only."""
import argparse,json,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',default='results/2026-09-18-pilot/verification');a=p.parse_args()
    root=Path(a.campaign)
    data={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/result.json')}
    comparisons=[]
    def compare(label,left,right,tolerance):
        l,r=data.get(left),data.get(right)
        if not l or not r:return
        if not l['valid_for_permeability'] or not r['valid_for_permeability']:
            comparisons.append(dict(case=label,passed=False,reason='one or both runs invalid/unconverged'));return
        x=np.asarray(l.get('K_lu',l.get('k_lu')));y=np.asarray(r.get('K_lu',r.get('k_lu')))
        error=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-30))
        comparisons.append(dict(case=label,relative_error=error,tolerance=tolerance,passed=error<tolerance))
    for j in range(3):
        for b in ('cupy-array','cuda'):
            compare(f'backend_{j}_{b}',f'parity_converged_{j}_{b}',f'parity_converged_{j}_numpy',.001)
            left=root/f'parity_fixed_{j}_{b}'/'fields.npz';right=root/f'parity_fixed_{j}_numpy'/'fields.npz'
            if left.exists() and right.exists():
                x=np.load(left);y=np.load(right);delta=max(float(np.max(abs(x[c]-y[c]))) for c in ('ux','uy','uz'))
                comparisons.append(dict(case=f'fixed_fields_{j}_{b}',max_absolute_velocity_error=delta,tolerance=2e-14,passed=delta<2e-14))
        compare('float32_'+str(j),'precision32_'+str(j),'parity_converged_'+str(j)+'_cuda',.01)
    for name,tolerance in (('half',.01),('reverse',.01),('tighter',.001)):
        compare('force_or_tolerance_'+name,'sensitivity_'+name,'sensitivity_F',tolerance)
    for name,r in data.items():
        if name.startswith('laminate') and r['valid_for_permeability']:
            comparisons.append(dict(case=name+'_reciprocity',relative_error=r['reciprocity_error'],tolerance=.01,passed=r['reciprocity_error']<.01,
                                    continuum_error=r['comparison']['tensor_relative_error'],principal_values=r['principal_values_lu']))
    write_json(root/'comparisons.json',comparisons)
    rows=[]
    for name,r in sorted(data.items()):
        loads=r.get('loads',[r]);valid=r['valid_for_permeability']
        k=r.get('k_lu') if 'k_lu' in r else (np.diag(r['K_lu']).tolist() if valid else None)
        rows.append(dict(case=name,valid=valid,status=r.get('termination_reason','tensor'),k_lu=k,
                         wall_s=r['case_wall_s'],iterations=sum(v.get('iterations_completed',0) for v in loads),
                         max_mach=max(v.get('diagnostics',{}).get('mach_max',0) for v in loads),
                         error=r.get('comparison',{}).get('relative_error',r.get('comparison',{}).get('tensor_relative_error'))))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for b in ('numpy','cuda'):
        h=[];err=[]
        for n in (12,24,48):
            r=data.get(f'channel_h{n}_{b}')
            if r and r['valid_for_permeability']:h.append(n);err.append(r['comparison']['relative_error'])
        axs[0].loglog(h,err,'o-',label=b)
    axs[0].set(xlabel='fluid aperture (cells)',ylabel='relative permeability error',title='Fixed-porosity channel refinement');axs[0].legend()
    for tau in (.7,.85,1.,1.2,1.6):
        h=[];err=[]
        for n in (12,48):
            r=data.get(f'tau_{tau}_h{n}')
            if r and r['valid_for_permeability']:h.append(n);err.append(r['comparison']['relative_error'])
        axs[1].loglog(h,err,'o-',label=f'tau={tau}')
    axs[1].set(xlabel='fluid aperture (cells)',ylabel='relative permeability error',title='Tau sensitivity');axs[1].legend(fontsize=8)
    n=[];err=[]
    for size in (24,36,48):
        r=data.get('laminate3_'+str(size))
        if r and r['valid_for_permeability']:n.append(size);err.append(r['comparison']['tensor_relative_error'])
    axs[2].loglog(n,err,'o-');axs[2].set(xlabel='period (cells)',ylabel='relative tensor Frobenius error',title='Oblique 3D laminate')
    fig.tight_layout();fig.savefig(root/'verification.png',dpi=180)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
