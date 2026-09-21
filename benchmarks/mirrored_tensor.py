"""Directional permeabilities of the mirrored Bentheimer crop (sealed sample), one JSON per load."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from porewise import lbm_stokes_3d, mirrored

dataset, output = sys.argv[1], Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True)
image = np.fromfile(dataset, dtype=np.uint8).reshape(500, 500, 500)
mask = mirrored(np.ascontiguousarray(image[58:442, 58:442, 58:442] == 255)); del image
for j, c in enumerate('xyz'):
    target = output / f'bentheimer384_mirrored_trt_tau0.6_{c}.json'
    if target.exists():
        continue
    F = [0., 0., 0.]; F[j] = 1e-6
    print(time.strftime('%H:%M:%S'), 'START', c, flush=True)
    r = lbm_stokes_3d(mask, F_x=F[0], F_y=F[1], F_z=F[2], tau=0.6, collision='trt', backend='cuda-sparse',
                      precision='float32', verbose=False, n_steps_max=400000, conv_tol=1e-6, conv_window=100,
                      consecutive=3, characteristic_length=20, wall_timeout_s=14400, return_fields=False)
    r.update(load=c, k_column_lu=[r['nu'] * r[f'u_{a}_mean_total'] / 1e-6 for a in 'xyz'], crop=[58, 442], voxel_size_m=5e-6)
    r['k_darcy'] = r['k_column_lu'][j] * 25e-12 / 9.869233e-13
    target.write_text(json.dumps(r, indent=1, default=float))
    print(time.strftime('%H:%M:%S'), 'END', c, r['termination_reason'], r['iterations'], 'k_darcy %.4f' % r['k_darcy'], flush=True)
