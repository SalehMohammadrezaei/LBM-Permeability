"""Sequential converged loads on one crop with a chosen backend/collision; results only, no fields."""
import argparse, json, sys, time, subprocess, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lbm_permeability import lbm_stokes_3d

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--shape', type=int, nargs=3, default=[500, 500, 500])
p.add_argument('--crop', type=int, nargs=2, help='cubic crop lo hi; omit for the full image'); p.add_argument('--output', required=True)
p.add_argument('--cases', required=True, help='JSON list of {name,axis,tau,collision,backend}')
p.add_argument('--tol', type=float, default=1e-6); p.add_argument('--length', type=float, default=20)
p.add_argument('--solid-value', type=int, default=255)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
raw = Path(a.dataset).read_bytes()
image = np.frombuffer(raw, dtype=np.uint8).reshape(a.shape)
if a.crop:
    lo, hi = a.crop
    image = image[lo:hi, lo:hi, lo:hi]
blocked = np.ascontiguousarray(image == a.solid_value)
digest = hashlib.sha256(raw).hexdigest()
del raw, image
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
for case in json.loads(a.cases):
    target = out / (case['name'] + '.json')
    if target.exists():
        continue
    force = [0., 0., 0.]; force[case['axis']] = 1e-6
    print(time.strftime('%H:%M:%S'), 'START', case, flush=True)
    r = lbm_stokes_3d(blocked, F_x=force[0], F_y=force[1], F_z=force[2], tau=case['tau'],
                      collision=case['collision'], backend=case['backend'], precision=case.get('precision', 'float64'),
                      verbose=bool(case.get('verbose')), heartbeat=5000,
                      n_steps_max=case.get('steps', 150000), conv_tol=a.tol, conv_window=100, consecutive=3,
                      characteristic_length=a.length, wall_timeout_s=case.get('timeout', 5400), max_mach=case.get('max_mach', .05), return_fields=False)
    r.update(case=case, dataset=str(a.dataset), dataset_sha256=digest, crop=a.crop,
             source_commit=commit, source_dirty=bool(dirty))
    target.write_text(json.dumps(r, indent=1, default=float))
    print(time.strftime('%H:%M:%S'), 'END', case['name'], r['termination_reason'], r['iterations'],
          'k_diag', r['nu'] * r[f"u_{'xyz'[case['axis']]}_mean_total"] / 1e-6, f"{r['elapsed_s']:.0f}s", flush=True)
