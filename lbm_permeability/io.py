"""Strict JSON and reusable array exports; provenance excludes environment secrets."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def _git_provenance():
    """Git is optional, including in installed packages outside a checkout."""
    values = {}
    for key, args in (('commit', ['rev-parse', 'HEAD']),
                      ('working_tree_status', ['status', '--porcelain=v1'])):
        try:
            result = subprocess.run(['git', *args], text=True, capture_output=True,
                                    timeout=5)
            if result.returncode != 0:
                reason = 'git {} exited with status {}'.format(args[0], result.returncode)
                break
            values[key] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError, UnicodeError) as error:
            reason = 'git {} unavailable ({})'.format(args[0], type(error).__name__)
            break
    else:
        values['git_provenance_status'] = dict(status='available', reason=None)
        return values
    return dict(commit=None, working_tree_status=None,
                git_provenance_status=dict(status='unavailable', reason=reason))


def provenance(blocked=None):
    import importlib.metadata as im
    packages={}
    for name in ('numpy','cupy-cuda12x','scipy','lbm-permeability'):
        try: packages[name]=im.version(name)
        except im.PackageNotFoundError: packages[name]=None
    source=Path(__file__).parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}
    out=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=platform.node(),
             python=sys.version,packages=packages,source_sha256=hashes,
             **_git_provenance())
    if blocked is not None:
        out['mask_sha256_c_order_bool']=hashlib.sha256(np.ascontiguousarray(blocked,dtype=np.bool_).tobytes()).hexdigest()
    return out


def save_result(directory,result,metadata=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data=dict(result);arrays={}
    if 'loads' in data:
        saved_loads=[]
        for j,load in enumerate(data['loads']):
            subdir=directory/f'load_{j}'
            save_result(subdir,load,{'tensor_load_column':j})
            saved_loads.append(json.loads((subdir/'result.json').read_text()))
            if 'fields_file' in saved_loads[-1]: saved_loads[-1]['fields_file']=f'load_{j}/fields.npz'
        data['loads']=saved_loads
    for key in ('ux','uy','uz','rho','diameter_map'):
        if key in data:
            arrays[key]=data.pop(key)
    if arrays:
        np.savez_compressed(directory/'fields.npz',**arrays)
        data['fields_file']='fields.npz'
    data['provenance']=provenance()
    data['metadata']=metadata or {}
    write_json(directory/'result.json',data)
    if 'convergence_history' in result:
        import csv
        history=result['convergence_history']
        keys=sorted(set().union(*(d.keys() for d in history))) if history else ['iterations']
        with (directory/'convergence.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(clean(history))
