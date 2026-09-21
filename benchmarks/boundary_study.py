"""Quantify two errors on a real rock: the periodic wrap of a non-periodic image, and voxel resolution.

wrap      the image as it is, periodic in every direction
mirror_x  the image followed by its mirror along the flow axis, so pores meet across the wrap
mirror    mirrored in all three directions
refine2   every voxel split in 2x2x2, same geometry at half the voxel size
Each case stores k, the plane-mean density profile along the flow axis (the pressure is rho/3)
and the software versions.
"""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from porewise import lbm_stokes_3d

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
p.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 384]); a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
image = np.fromfile(a.dataset, dtype=np.uint8).reshape(500, 500, 500)
import cupy, numba
versions = dict(python=platform.python_version(), numpy=np.__version__, cupy=cupy.__version__, numba=numba.__version__,
                gpu=subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
                commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip())


def variants(solid):
    yield 'wrap', solid, 1
    yield 'mirror_x', np.concatenate([solid, solid[:, :, ::-1]], axis=2), 1
    m = np.concatenate([solid, solid[:, :, ::-1]], axis=2)
    m = np.concatenate([m, m[:, ::-1]], axis=1)
    yield 'mirror', np.concatenate([m, m[::-1]], axis=0), 1
    if solid.shape[0] <= 256:
        yield 'refine2', solid.repeat(2, 0).repeat(2, 1).repeat(2, 2), 2


for n in a.sizes:
    lo = 250 - n // 2
    solid = np.ascontiguousarray(image[lo:lo + n, lo:lo + n, lo:lo + n] == 255)
    for name, mask, factor in variants(solid):
        target = out / f'bentheimer{n}_{name}.json'
        if target.exists():
            continue
        mask = np.ascontiguousarray(mask)
        force = 1e-6 / factor ** 3                       # same Reynolds number on the refined grid
        print(time.strftime('%H:%M:%S'), 'START', target.name, mask.shape, flush=True)
        r = lbm_stokes_3d(mask, F_x=force, tau=0.6, collision='trt', backend='cuda-sparse',
                          precision='float32' if mask.size > 3e8 else 'float64', verbose=False,
                          n_steps_max=400000, conv_tol=1e-6, conv_window=100, consecutive=3,
                          characteristic_length=20 * factor, wall_timeout_s=14400, return_fields=True)
        rho = r.pop('rho'); fluid = ~mask
        profile = [(float(rho[:, :, i][fluid[:, :, i]].mean()) if fluid[:, :, i].any() else None) for i in range(mask.shape[2])]
        for c in 'xyz': r.pop('u' + c)
        k_lu = r['nu'] * r['u_x_mean_total'] / force
        r.update(case=name, crop_size=n, refinement=factor, k_lu=k_lu, k_in_original_voxels=k_lu / factor ** 2,
                 k_darcy=k_lu / factor ** 2 * 25e-12 / 9.869233e-13, plane_mean_density_along_x=profile,
                 force=force, versions=versions)
        target.write_text(json.dumps(r, indent=1, default=float))
        print(time.strftime('%H:%M:%S'), 'END', target.name, r['termination_reason'], r['iterations'], 'k_darcy %.4f' % r['k_darcy'], flush=True)
