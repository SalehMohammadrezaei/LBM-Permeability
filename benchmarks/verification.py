"""Verification ladder: geometries with known permeability, BGK beside TRT.

Groups (select with --groups):
  channel   plane Poiseuille, tau sweep and resolution sweep (2D, CUDA)
  spheres   simple-cubic sphere array against Zick & Homsy (1982) drag coefficients
  slit      inclined periodic slit: the tensor must recover the slit direction
  pipes     straight pipes of Saxena et al. (2017): circle, square, triangle cross-sections
  wagner    thin micromodel cell with one cylinder, Wagner et al. (2021) Table 3

Every case is written to its own JSON file and skipped when that file exists.
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lbm_permeability import lbm_stokes_2d_fast, lbm_stokes_3d, compute_permeability_tensor, geometry

TAUS = (0.55, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
# Zick & Homsy, J. Fluid Mech. 115 (1982), simple cubic: solid fraction -> drag coefficient K,
# drag on one sphere = 6 pi mu a U K with U the superficial velocity.
ZICK_HOMSY_SC = {0.027: 2.008, 0.064: 2.810, 0.125: 4.292, 0.216: 7.442, 0.343: 15.4, 0.45: 28.1, 0.5236: 42.1}


def force_for(tau):
    # constant F/nu keeps the velocity, Mach and Reynolds numbers equal across tau
    return 1e-6 * (tau - .5) * 2


def run(out, name, reference, fn, **meta):
    target = out / (name + '.json')
    if target.exists():
        return
    started = time.time()
    r = fn()
    k = r.pop('k')
    r.update(name=name, k_lu=k, reference_k_lu=reference,
             relative_error=None if reference is None else (k - reference) / reference,
             wall_s=time.time() - started, **meta)
    target.write_text(json.dumps(r, indent=1, default=float))
    print(time.strftime('%H:%M:%S'), name, 'k=%.8g' % k, 'ref=%s' % reference,
          'err=%s' % ('n/a' if reference is None else '%.4e' % r['relative_error']), flush=True)


def solve(blocked, axis, tau, collision, tol, **kw):
    force = [0., 0., 0.][:blocked.ndim]; force[axis] = force_for(tau)
    common = dict(tau=tau, collision=collision, verbose=False, conv_tol=tol, conv_window=200,
                  n_steps_max=kw.pop('steps', 2000000), wall_timeout_s=kw.pop('timeout', 3600), return_fields=False)
    if blocked.ndim == 2:
        r = lbm_stokes_2d_fast(blocked, F_x=force[0], F_y=force[1], **common)
    else:
        r = lbm_stokes_3d(blocked, F_x=force[0], F_y=force[1], F_z=force[2], backend='cuda-sparse', **common)
    u = r[f"u_{'xyz'[axis]}_mean_total"]
    return dict(k=r['nu'] * u / force[axis], accepted=r['valid_for_permeability'], reason=r['termination_reason'],
                iterations=r['iterations'], shape=list(blocked.shape), porosity=r['porosity'],
                mach_max=r['diagnostics'].get('mach_max'), elapsed_s=r['elapsed_s'])


def channel(out, tol):
    for gap in (4, 8, 16, 32, 64):
        ny = gap + 8
        m = geometry.parallel_plates(ny, 4, gap)
        for collision in ('bgk', 'trt'):
            for tau in (TAUS if gap in (8, 32) else (1.0,)):
                run(out, f'channel_gap{gap}_{collision}_tau{tau}', gap ** 3 / 12 / ny,
                    lambda: solve(m, 0, tau, collision, tol), group='channel', gap=gap, tau=tau, collision=collision,
                    reference='continuum gap^3/(12 Ny), wall half a cell outside the last fluid node',
                    nodal_reference_k_lu=(gap ** 3 / 12 + gap / 24) / ny)


def sphere_array(n, fraction):
    a = n * (3 * fraction / (4 * math.pi)) ** (1 / 3)
    z, y, x = np.mgrid[0:n, 0:n, 0:n] + .5 - n / 2
    return x * x + y * y + z * z <= a * a, a


def spheres(out, tol):
    for fraction, drag in ZICK_HOMSY_SC.items():
        for n in (32, 64, 128):
            m, a = sphere_array(n, fraction)
            reference = n ** 3 / (6 * math.pi * a * drag)
            for collision in ('bgk', 'trt'):
                taus = TAUS if (n == 64 and fraction in (0.216, 0.45)) else (1.0,)
                for tau in taus:
                    run(out, f'sc_c{fraction}_n{n}_{collision}_tau{tau}', reference,
                        lambda: solve(m, 0, tau, collision, tol), group='spheres', n=n, solid_fraction=fraction,
                        voxel_solid_fraction=float(m.mean()), radius_lu=a, tau=tau, collision=collision,
                        reference='Zick & Homsy (1982) simple cubic: k = L^3/(6 pi a K)', drag_coefficient=drag)


def slit(out, tol):
    n, nz, width = 60, 4, 12
    for p, q in ((0, 1), (1, 3), (1, 2), (1, 1), (2, 1)):
        y, x = np.mgrid[0:n, 0:n]
        plane = ((q * y - p * x) % n) >= width
        m = np.broadcast_to(plane, (nz, n, n)).copy()
        angle = math.degrees(math.atan2(p, q))
        for collision in ('bgk', 'trt'):
            def tensor():
                t = compute_permeability_tensor(m, force_magnitude=1e-6, backend='cuda-sparse', tau=1., collision=collision,
                                                verbose=False, conv_tol=tol, conv_window=200, n_steps_max=2000000)
                K = np.array(t['K_lu']); sym = (K + K.T) / 2
                vals, vecs = np.linalg.eigh(sym[:2, :2])
                v = vecs[:, -1]
                found = math.degrees(math.atan2(v[1], v[0])) % 180
                return dict(k=float(vals[-1]), K_lu=K.tolist(), in_plane_eigenvalues=vals.tolist(),
                            found_angle_deg=found, angle_error_deg=(found - angle + 90) % 180 - 90,
                            reciprocity_error=t['reciprocity_error'], accepted=t['valid_for_permeability'],
                            k_zz=float(K[2, 2]))
            run(out, f'slit_{p}_{q}_{collision}', None, tensor, group='slit', angle_deg=angle, collision=collision,
                width_along_y=width, note='direction of the largest in-plane eigenvalue must equal the slit direction')


def pipes(out, tol, folder):
    # k = porosity * U_mean/(G/mu) with the classical mean-velocity factors of each cross-section
    shapes = {'Circle': lambda area, size: area / (8 * math.pi),          # R^2/8 with R^2 = area/pi
              'Square': lambda area, size: 0.0351443 * area,               # side^2 * 0.0351443
              'Triangle': lambda area, size: area / (20 * math.sqrt(3))}   # side^2/80, area = sqrt(3)/4 side^2
    for shape, mean_factor in shapes.items():
        for size in (200, 400, 800):
            path = Path(folder) / f'{shape}_{size}_1024cube_1um.raw'
            if not path.exists():
                continue
            image = np.memmap(path, dtype=np.uint8, mode='r', shape=(1024, 1024, 1024))
            # label 0 is pore, 1 is solid; every pipe is uniform along array axis 0
            assert (image[0] == image[500]).all()
            pore = np.array(image[0]) == 0
            area = float(pore.sum())
            m = np.broadcast_to(~pore, (4,) + pore.shape).copy()      # flow along array axis 0 = component z
            reference = pore.mean() * mean_factor(area, size)
            for collision in ('bgk', 'trt'):
                run(out, f'pipe_{shape}_{size}_{collision}', reference, lambda: solve(m, 2, 1.0, collision, tol),
                    group='pipes', shape_name=shape, nominal_size=size, pore_area_voxels=area, collision=collision,
                    reference='analytical mean velocity of the cross-section with the voxel-counted area',
                    source='Saxena et al. (2017) Mendeley Data 4g723tr5v3 v2, CC BY 4.0')


# Wagner et al., Transp. Porous Media 138 (2021) Table 3: k in 1e-11 m^2 for cylinder radius in mm,
# unit cell 1 mm x 1 mm, depth 0.091 mm between no-slip plates.
WAGNER = {0.35: dict(fem=25.7, lbm=26.7, sph=22.9, hom3d=25.8), 0.40: dict(fem=17.7, lbm=17.7, sph=16.4, hom3d=17.4),
          0.45: dict(fem=7.54, lbm=8.11, sph=8.59, hom3d=8.14), 0.47: dict(fem=3.62, lbm=3.86, sph=5.35, hom3d=3.97),
          0.49: dict(fem=0.46, lbm=0.54, sph=None, hom3d=0.47)}


def wagner(out, tol):
    for depth in (16, 32, 91):                       # fluid nodes across the 0.091 mm depth; 91 is exact at 1 um
        dx = 0.091e-3 / depth
        n = round(1e-3 / dx)
        y, x = (np.mgrid[0:n, 0:n] + .5) * (1e-3 / n) - .5e-3
        for radius, published in WAGNER.items():
            if depth == 91 and radius not in (0.40, 0.49):
                continue
            section = x * x + y * y <= (radius * 1e-3) ** 2
            m = np.broadcast_to(section, (depth + 1, n, n)).copy()
            m[0] = True                               # one solid layer closes both plates through periodicity
            for collision in ('bgk', 'trt'):
                def cell():
                    r = solve(m, 0, 1.0, collision, tol)
                    # published k averages over the cell between the plates, not over the wall layer
                    r['k_m2'] = r['k'] * (depth + 1) / depth * (1e-3 / n) ** 2
                    r['k_1e-11_m2'] = r['k_m2'] / 1e-11
                    return r
                run(out, f'wagner_r{radius}_d{depth}_{collision}', None, cell, group='wagner', radius_mm=radius,
                    depth_nodes=depth, cell_nodes=n, collision=collision, published_1e11_m2=published,
                    note='cell edge is n*dx; for depth 16 and 32 it differs from 1 mm by under 0.2 %')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--groups', nargs='+', default=['channel', 'spheres', 'slit'])
    p.add_argument('--tol', type=float, default=1e-9); p.add_argument('--saxena')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / 'provenance.json').write_text(json.dumps(dict(source_commit=commit, started=time.strftime('%F %T'), tol=a.tol)))
    for g in a.groups:
        {'channel': channel, 'spheres': spheres, 'slit': slit, 'wagner': wagner}.get(g, lambda o, t: pipes(o, t, a.saxena))(out, a.tol)
