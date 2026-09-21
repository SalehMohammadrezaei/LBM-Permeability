"""Fixed-step throughput and memory of the dense and pore-only CUDA backends on one crop."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from porewise import lbm_stokes_3d
from porewise.backends import cp

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--shape', type=int, nargs=3, default=[500, 500, 500])
p.add_argument('--crop', type=int, nargs=2, default=[58, 442]); p.add_argument('--steps', type=int, default=1000)
p.add_argument('--backend', required=True); p.add_argument('--collision', default='bgk')
p.add_argument('--precision', default='float64'); p.add_argument('--output', required=True)
a = p.parse_args()
image = np.fromfile(a.dataset, dtype=np.uint8).reshape(a.shape)
lo, hi = a.crop
blocked = np.ascontiguousarray(image[lo:hi, lo:hi, lo:hi] != 0)   # 0 pore, 255 grain
kw = dict(F_x=1e-6, tau=1., backend=a.backend, collision=a.collision, precision=a.precision, verbose=False,
          conv_tol=1e-30, conv_atol=1e-300, conv_window=10 ** 9, stability_every=10 ** 9)
lbm_stokes_3d(blocked, n_steps_max=20, **kw)                       # compile and warm up
cp.get_default_memory_pool().free_all_blocks()
t = time.perf_counter(); r = lbm_stokes_3d(blocked, n_steps_max=a.steps, **kw); wall = time.perf_counter() - t
loop = r['timing']['solve_and_diagnostics_s']
out = dict(backend=a.backend, collision=a.collision, precision=a.precision, shape=list(blocked.shape),
           porosity=r['porosity'], fluid_nodes=int((~blocked).sum()), steps=a.steps, loop_s=loop, wall_s=wall,
           setup_s=r['timing']['setup_s'],
           mlups_total_voxels=blocked.size * a.steps / loop / 1e6,
           mfluid_lups=int((~blocked).sum()) * a.steps / loop / 1e6,
           gpu_pool_peak_bytes=r['memory']['gpu_pool_reserved_peak_sampled_bytes'],
           sparse_bytes=r['memory'].get('sparse_bytes'), u_x_mean_total=r['u_x_mean_total'])
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(out, indent=1)); print(json.dumps(out))
