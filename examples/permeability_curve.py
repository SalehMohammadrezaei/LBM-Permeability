"""Illustrative random-disk sweep; every outcome retained, optional fitted trend.

Not an independent Kozeny-Carman validation or a statistical porosity law.
"""
import argparse,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lbm_permeability import geometry,lbm_stokes
from lbm_permeability.io import save_result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/porosity-illustration');p.add_argument('--backend',default='numpy');p.add_argument('--size',type=int,default=32);p.add_argument('--steps',type=int,default=10000);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for nd in (3,5,7,9):
        m=geometry.random_disks(a.size,a.size,nd,max(2,a.size/10),seed=100+nd)
        r=lbm_stokes(m,F_x=1e-6,backend=a.backend,n_steps_max=a.steps,conv_tol=1e-6,conv_window=100,wall_timeout_s=30,verbose=False,return_fields=False)
        save_result(out/f'disks_{nd}',r,dict(seed=100+nd,overlapping_disks=nd))
        k=r['k_lu'];rows.append(dict(disks=nd,porosity=geometry.porosity(m),k_lu=k,status=r['termination_reason'],
                                    fit_exclusion_reason='' if k is not None and k>0 else ('zero or nonpositive valid permeability' if r['valid_for_permeability'] else r['termination_reason'])))
    with (out/'all_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    valid=[r for r in rows if r['k_lu'] is not None]
    if valid:ax.scatter([r['porosity'] for r in valid],[r['k_lu'] for r in valid],label='accepted individual masks')
    positive=[r for r in valid if r['k_lu']>0]
    if len(positive)>=2:
        phi=np.array([r['porosity'] for r in positive]);k=np.array([r['k_lu'] for r in positive]);shape=phi**3/(1-phi)**2
        coefficient=np.exp(np.mean(np.log(k/shape)));pp=np.linspace(phi.min(),phi.max(),100)
        ax.plot(pp,coefficient*pp**3/(1-pp)**2,'--',label='fitted Kozeny-Carman shape (illustration)')
    ax.set(xlabel='total porosity',ylabel='permeability (lattice cells squared)',title='Random overlapping disks; all statuses in CSV')
    if valid:ax.legend()
    fig.savefig(out/'illustration.png',dpi=150)

if __name__=='__main__':main()
