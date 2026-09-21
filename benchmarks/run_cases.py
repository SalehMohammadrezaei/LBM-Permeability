"""Bounded sequential verification; each case retained, including failures.

Run from repository: .venv/bin/python benchmarks/run_cases.py --suite verification
"""
import argparse,json,sys,time,traceback,os,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise import geometry
from porewise.solver import periodic
from porewise.tensor import compute_permeability_tensor
from porewise.d2q9_pressure import lbm_stokes_2d_pressure
from porewise.io import save_result,write_json,provenance
from porewise.backends import cp,HAS_GPU


def generate(g):
    kind=g['kind']
    if kind=='channel':
        h=g['gap'];m=geometry.parallel_plates(2*h,g.get('nx',12),h)
        if g.get('ndim',2)==3:m=np.broadcast_to(m,(g.get('nz',8),)+m.shape).copy()
        if 'permutation' in g:m=np.transpose(m,g['permutation']).copy()
        return m
    if kind=='laminate':
        n=g['n'];ndim=g.get('ndim',3)
        coords=np.ogrid[tuple(slice(0,n) for _ in range(ndim))]
        return sum(coords)%n>=n//2
    if kind=='porous':
        shape=g['shape'];count=g.get('count',8);radius=g.get('radius',3)
        if len(shape)==3:return geometry.random_spheres(*shape,count,radius,seed=g.get('seed',42))
        return geometry.random_disks(*shape,count,radius,seed=g.get('seed',42))
    if kind in ('sphere','cylinder'):
        n=g['n'];dim=3 if kind=='sphere' else 2
        coords=np.ogrid[tuple(slice(0,n) for _ in range(dim))]
        a=n*(g['solid_fraction']*3/(4*np.pi))**(1/3) if dim==3 else n*np.sqrt(g['solid_fraction']/np.pi)
        return sum((x-(n-1)/2)**2 for x in coords)<=a*a
    if kind=='npy':return np.load(g['path'],allow_pickle=False)
    raise ValueError(kind)


def cases():
    out=[]
    def add(cid,g,backend='cuda',mode='scalar',**s):
        out.append(dict(id=cid,geometry=g,backend=backend,mode=mode,settings=s))
    # Fixed physical channel aperture, phi=.5, voxel size=1/gap.
    for h in (12,24,48):
        for b in ('numpy','cuda'):
            add(f'channel_h{h}_{b}',dict(kind='channel',gap=h),b,conv_tol=1e-7,force=1e-6*(12/h)**2,characteristic_length=h)
    for tau in (.7,.85,1.,1.2,1.6):
        for h in (12,48):add(f'tau_{tau}_h{h}',dict(kind='channel',gap=h),tau=tau,force=1e-6*(12/h)**2,characteristic_length=h)
    for j,perm in enumerate(((0,1,2),(2,0,1),(1,2,0))):
        # Original x is old axis 2, new array axis index perm.index(2).
        axis=2-list(perm).index(2)
        add(f'extruded_axis_{j}',dict(kind='channel',gap=12,ndim=3,permutation=perm),direction=axis,force=1e-6)
    g=dict(kind='porous',shape=[16,20,24],count=12,radius=3)
    for j in range(3):
        for b in ('numpy','cupy-array','cuda'):
            add(f'parity_fixed_{j}_{b}',g,b,n_steps_max=31,conv_window=10,direction=j,return_fields=True)
            add(f'parity_converged_{j}_{b}',g,b,direction=j,conv_tol=1e-6,wall_timeout_s=35)
        add(f'precision32_{j}',g,direction=j,precision='float32',conv_tol=1e-5,conv_atol=1e-10)
    for force,label in ((1e-6,'F'),(5e-7,'half'),(-1e-6,'reverse')):
        add('sensitivity_'+label,g,mode='tensor',force=force,conv_tol=1e-6)
    add('sensitivity_tighter',g,mode='tensor',force=1e-6,conv_tol=1e-8)
    for n in (24,36,48):add(f'laminate3_{n}',dict(kind='laminate',n=n),mode='tensor',conv_tol=1e-7,force=1e-6,characteristic_length=n/(2*np.sqrt(3)))
    for n in (24,48):add(f'laminate2_{n}',dict(kind='laminate',n=n,ndim=2),mode='tensor',conv_tol=1e-7)
    for kind in ('sphere','cylinder'):
        for n in (16,24,32):add(f'{kind}_{n}',dict(kind=kind,n=n,solid_fraction=.1),mode='tensor' if kind=='sphere' else 'scalar',force=1e-7,conv_tol=1e-7)
    for pad in (0,4,8,16):
        add(f'pressure_channel_pad{pad}',dict(kind='channel',gap=12,nx=48),mode='pressure',pad=pad)
    add('pressure_closed_y',dict(kind='npy',path='benchmarks/configs/open_channel.npy'),mode='pressure',walls_y=True,pad=8)
    for pad in (0,4,8,16):
        add(f'pressure_porous_pad{pad}',dict(kind='porous',shape=[24,48],count=8,radius=3),mode='pressure',pad=pad)
    return out


def run(case,root,budget):
    directory=root/case['id']
    if (directory/'result.json').exists():
        old=json.loads((directory/'result.json').read_text())
        if old.get('metadata',{}).get('case')==case and old.get('provenance',{}).get('source_sha256')==provenance()['source_sha256']:
            print('resume verified',case['id'],flush=True);return 0
        raise RuntimeError(f'existing case differs from configuration/source: {directory}; select a new campaign directory')
    m=generate(case['geometry']);s=dict(case['settings']);mode=case['mode']
    t=time.perf_counter()
    settings=dict(tau=1.,n_steps_max=40000,conv_tol=1e-6,conv_window=100,verbose=False,
                  return_fields=False,wall_timeout_s=min(40.,budget),stability_every=50)
    settings.update(s);force=settings.pop('force',1e-6);j=settings.pop('direction',0)
    settings['wall_timeout_s']=min(settings['wall_timeout_s'],budget/(m.ndim if mode=='tensor' else 1))
    if HAS_GPU:cp.get_default_memory_pool().free_all_blocks()
    pre=dict(load_average=os.getloadavg(),host_available_bytes=None,
             gpu_free_total_bytes=cp.cuda.runtime.memGetInfo() if HAS_GPU else None)
    try:
        if mode=='pressure':
            settings.pop('conv_tol');settings.pop('conv_window')
            settings.update(energy_eps=1e-6,energy_window=500,sample_every=50,deltaP=1e-5)
            r=lbm_stokes_2d_pressure(m,**settings)
        elif mode=='tensor':r=compute_permeability_tensor(m,force_magnitude=force,backend=case['backend'],**settings)
        else:
            f=np.zeros(m.ndim);f[j]=force
            r=periodic(m,f,backend=case['backend'],**settings)
        quality={}
        if case['geometry']['kind']=='channel':
            h=case['geometry']['gap'];target=.5*h*h/12
            quality['analytic_k_lu']=target
            if r.get('k_lu') is not None:quality['relative_error']=abs(r['k_lu']/target-1)
        if case['geometry']['kind']=='laminate':
            n=case['geometry']['n'];normal=np.ones(m.ndim)/np.sqrt(m.ndim);h=n/(2*np.sqrt(m.ndim))
            ref=.5*h*h/12*(np.eye(m.ndim)-np.outer(normal,normal))
            quality['continuum_tensor_lu']=ref
            if r.get('valid_for_permeability'):quality['tensor_relative_error']=np.linalg.norm(r['K_lu']-ref)/np.linalg.norm(ref)
        r['comparison']=quality
    except Exception as e:
        r=dict(valid_for_permeability=False,termination_reason='exception',exception=repr(e),traceback=traceback.format_exc())
    r['case_wall_s']=time.perf_counter()-t
    save_result(directory,r,dict(case=case,mask=provenance(m),resources_before=pre,
                                 initial_state='rho=1, equilibrium at rest; Guo half-force velocity',
                                 voxel_size_m=1/case['geometry']['gap'] if case['geometry']['kind']=='channel' else 1.,
                                 unit_note='synthetic physical spacing only; default 1 metre outside channel refinement'))
    print(case['id'],r.get('termination_reason',r['valid_for_permeability']),f"{r['case_wall_s']:.2f}s",flush=True)
    return r['case_wall_s']


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['verification'],default='verification');p.add_argument('--campaign',default='results/2026-09-18-pilot/verification');p.add_argument('--budget',type=float,default=1000);p.add_argument('--case');p.add_argument('--config')
    a=p.parse_args();root=Path(a.campaign);root.mkdir(parents=True,exist_ok=True)
    np.save('benchmarks/configs/open_channel.npy',np.zeros((12,48),bool))
    definitions=json.loads(Path(a.config).read_text()) if a.config else cases()
    write_json(root/'case_definitions.json',definitions)
    used=0.
    for c in definitions:
        if a.case and a.case not in c['id']:continue
        if used>=a.budget:
            write_json(root/(c['id']+'_deferred.json'),dict(case=c,reason='pilot budget exhausted'));continue
        used+=run(c,root,a.budget-used)
    write_json(root/'budget.json',dict(case_wall_seconds=used,limit_seconds=a.budget))

if __name__=='__main__':main()
