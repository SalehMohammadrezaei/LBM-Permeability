"""Ten-minute regression: three headline cases with the software versions recorded.

  channel     plane channel, 8 nodes wide, TRT, tau 1.5: k = (8^3/12 + 8/24)/16 lattice units
  spheres     simple cubic array, solid fraction 0.216, 64-cube, TRT, tau 1: Zick and Homsy (1982)
  bentheimer  DRP-29 crop [58:442]^3, x load, TRT, tau 0.6, periodic wrap: 4.1024 darcy

Run:  python benchmarks/regression.py --output results/regression [--dataset Seg_Oxyz_0001_0001_0001.raw]
"""
import argparse, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import porewise
from porewise import lbm_stokes_3d, geometry

EXPECTED = dict(channel=(8 ** 3 / 12 + 8 / 24) / 16, spheres=None, bentheimer=4.1024)
p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--dataset')
p.add_argument('--backend', default='cuda-sparse'); a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as error: return f'unavailable: {error!r}'


env = dict(porewise=porewise.__version__, python=platform.python_version(), numpy=np.__version__,
           commit=sh(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']), tag=sh(['git', '-C', str(ROOT), 'describe', '--tags', '--always']),
           nvidia_smi=sh(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
           pip_freeze=sh([sys.executable, '-m', 'pip', 'freeze']).splitlines())
(out / 'environment.json').write_text(json.dumps(env, indent=1))
common = dict(collision='trt', backend=a.backend, verbose=False, conv_window=200, return_fields=False)
report = {}

m = np.broadcast_to(geometry.parallel_plates(16, 4, 8), (3, 16, 4)).copy()
r = lbm_stokes_3d(m, F_x=2e-6, tau=1.5, n_steps_max=200000, conv_tol=1e-10, **common)
report['channel'] = dict(k_lu=r['k_lu'], expected=EXPECTED['channel'], accepted=r['valid_for_permeability'], iterations=r['iterations'])

n, frac, drag = 64, 0.216, 7.442
rad = n * (3 * frac / (4 * math.pi)) ** (1 / 3)
z, y, x = np.mgrid[0:n, 0:n, 0:n] + .5 - n / 2
sph = x * x + y * y + z * z <= rad * rad
r = lbm_stokes_3d(sph, F_x=1e-6 * .1 * (32 / n) ** 3, tau=1.0, n_steps_max=2000000, conv_tol=1e-9, characteristic_length=n, **common)
ref = n ** 3 / (6 * math.pi * rad * drag)
report['spheres'] = dict(k_lu=r['k_lu'], expected=ref, relative_error=(r['k_lu'] - ref) / ref, accepted=r['valid_for_permeability'],
                         note='expected error about -2.0 % at this resolution (docs/verification.md)')

if a.dataset:
    image = np.fromfile(a.dataset, dtype=np.uint8).reshape(500, 500, 500)
    blocked = np.ascontiguousarray(image[58:442, 58:442, 58:442] == 255); del image
    r = lbm_stokes_3d(blocked, F_x=1e-6, tau=0.6, n_steps_max=100000, conv_tol=1e-6, consecutive=3,
                      characteristic_length=20, **{**common, 'conv_window': 100})
    k = r['k_lu'] * 25e-12 / 9.869233e-13 if r['k_lu'] else None
    report['bentheimer'] = dict(k_darcy=k, expected=EXPECTED['bentheimer'], accepted=r['valid_for_permeability'], iterations=r['iterations'])

for name, c in report.items():
    got = c.get('k_darcy', c.get('k_lu')); exp = c['expected']
    c['pass'] = bool(c['accepted']) and (abs(got - exp) / exp < 1e-4 if name != 'spheres' else abs(c['relative_error'] + 0.020) < 0.005)
    print(f"{name:11s} {'PASS' if c['pass'] else 'FAIL'}  got {got:.6g}  expected {exp:.6g}")
(out / 'regression.json').write_text(json.dumps(dict(environment=env, cases=report, finished=time.strftime('%F %T')), indent=1, default=float))
sys.exit(0 if all(c['pass'] for c in report.values()) else 1)
