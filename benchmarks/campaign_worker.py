"""Single campaign task process. Parent enforces resource and wall-time limits."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from porewise.io import write_json, provenance


def load_mask(job):
    raw = np.memmap(job['dataset'], dtype='u1', mode='r', shape=tuple(job['dataset_shape']))
    bounds = job['crop']
    cut = np.asarray(raw[tuple(slice(a,b) for a,b in bounds)])
    if not np.isin(cut, [0,255]).all():
        raise ValueError('Expected unchanged DRP29 labels 0 pore, 255 solid')
    return np.ascontiguousarray(cut == 255)


def run(job):
    out = Path(job['output']); out.mkdir(parents=True, exist_ok=True)
    mask = load_mask(job)
    ident = provenance(mask)
    write_json(out/'provenance.json', ident)
    start = time.perf_counter()
    kind = job['kind']
    if kind == 'solve':
        from porewise.solver import periodic
        from porewise.field_export import write_vti
        import cupy as cp
        if job['settings']['backend'] != 'numpy':
            cp.get_default_memory_pool().set_limit(size=job['gpu_pool_limit_bytes'])
            cp.cuda.runtime.deviceSynchronize()
        reps=[]
        count=job.get('repetitions',1)
        # Warmup and measurements use matching settings and diagnostics in this process.
        for rep in range(-int(job.get('warmup',False)),count):
            cp.get_default_memory_pool().free_all_blocks()
            t=time.perf_counter()
            r=periodic(mask,job['force'],**job['settings'])
            cp.cuda.runtime.deviceSynchronize()
            wall=time.perf_counter()-t
            fields={k:r.pop(k) for k in ('ux','uy','uz','rho') if k in r}
            loop=r['timing']['solve_and_diagnostics_s']
            r.update(public_call_wall_s=wall,
                total_lattice_mlups_public_call=mask.size*r['iterations']/wall/1e6,
                fluid_node_mlups_public_call=int((~mask).sum())*r['iterations']/wall/1e6,
                total_lattice_mlups_iteration_loop=mask.size*r['iterations']/loop/1e6,
                gpu_pool_live_after_call_bytes=cp.get_default_memory_pool().used_bytes(),
                gpu_pool_reserved_after_call_bytes=cp.get_default_memory_pool().total_bytes())
            if rep<0:
                write_json(out/'warmup.json',r)
                del fields,r
                continue
            reps.append(r)
            write_json(out/f'repetition_{rep}.json',r)
            write_json(out/'result.json',r) # Preserve solve even if subsequent export fails.
            if fields and (r['valid_for_permeability'] or job.get('preflight')):
                t=time.perf_counter()
                for key,value in fields.items(): np.save(out/(key+'.npy'),value)
                meta=write_vti(out/'fields.vti',mask,fields,job['voxel_size_m'],
                    origin=[a*job['voxel_size_m'] for a,b in job['crop'][::-1]])
                meta['array_format']='individual uncompressed NPY, float64 C-order (z,y,x)'
                meta['accepted']=r['valid_for_permeability']
                meta['export_wall_s']=time.perf_counter()-t
                write_json(out/'field_metadata.json',meta)
            hist=r['convergence_history']
            if hist:
                with (out/'convergence.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=sorted(set().union(*hist)))
                    w.writeheader();w.writerows(hist)
            del fields,r
        write_json(out/'repetitions.json',reps)
    elif kind == 'geometry':
        np.save(out/'blocked.npy',mask)
        r=dict(shape=mask.shape,porosity=float((~mask).mean()),crop=job['crop'],
               axis_order='zyx',voxel_size_m=job['voxel_size_m'],
               profiles={c:(~mask).mean(axis=tuple(a for a in range(3) if a!=i)).tolist()
                         for i,c in enumerate('zyx')})
        write_json(out/'result.json',r)
    elif kind == 'morphology':
        from porewise.morphology import local_thickness_distribution
        r=local_thickness_distribution(mask,job['voxel_size_m'],return_map=True,
                                      max_voxels=mask.size)
        diam=r.pop('diameter_map');np.save(out/'diameter_map.npy',diam)
        r['explicit_size_limit']=mask.size
        r['planning']=job.get('planning')
        r['boundary_influence']=[]
        for exclusion in (0,5,10,20):
            if min(mask.shape)<=2*exclusion:continue
            sl=tuple(slice(exclusion,-exclusion or None) for _ in range(3))
            values=diam[sl][~mask[sl]]
            r['boundary_influence'].append(dict(excluded_layers=exclusion,pore_voxels=values.size,
                percentiles_m=np.percentile(values,[10,50,90]) if values.size else None,
                interpretation='interior subset of the same full-support thickness map'))
        write_json(out/'result.json',r)
    elif kind == 'summarize':
        from campaign_report import report
        report(Path(job['campaign']))
    else: raise ValueError(kind)
    write_json(out/'complete.json',dict(fingerprint=job['fingerprint'],elapsed_s=time.perf_counter()-start))


if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text())
    try: run(job)
    except Exception as e:
        write_json(Path(job['output'])/'error.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
