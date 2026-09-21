"""Render the README figure: grains of a real rock with streamlines coloured by velocity magnitude."""
import argparse
import numpy as np
import pyvista as pv
from scipy import ndimage

p = argparse.ArgumentParser(); p.add_argument('--fields', required=True); p.add_argument('--output', required=True)
p.add_argument('--voxel-um', type=float, default=5.0); p.add_argument('--width', type=int, default=3600)
p.add_argument('--seeds', type=int, default=1400); a = p.parse_args()
d = np.load(a.fields)
solid = d['solid']; nz, ny, nx = solid.shape; h = a.voxel_um
magnitude = np.sqrt(d['ux'] ** 2 + d['uy'] ** 2 + d['uz'] ** 2)
scale = float(np.percentile(magnitude[~solid], 99.5))

order = lambda f: np.ascontiguousarray(f).ravel(order='C')            # (z,y,x) C order is VTK x-fastest
grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(h,) * 3)
grid['velocity'] = np.stack([order(d['ux']), order(d['uy']), order(d['uz'])], axis=1) / scale
grid['velocity magnitude'] = np.linalg.norm(grid['velocity'], axis=1)
# a lightly smoothed indicator gives grain surfaces without voxel stair-steps; the flow field is untouched
grid['grain'] = order(ndimage.gaussian_filter(solid.astype(np.float32), 0.9))

pv.OFF_SCREEN = True
height = int(a.width * 0.74)
plot = pv.Plotter(off_screen=True, window_size=(a.width, height), lighting='none')
plot.set_background('white')
for position, intensity in (((2.5, -2.0, 3.0), 0.95), ((-2.0, -1.0, 1.0), 0.35), ((0.5, 2.5, 0.5), 0.25)):
    light = pv.Light(position=tuple(c * nx * h for c in position), focal_point=(nx * h / 2, ny * h / 2, nz * h / 2),
                     intensity=intensity, light_type='scene light')
    plot.add_light(light)

back = grid.clip(normal=(0, -1, 0), origin=(0, ny * h * .5, 0))
grains = back.contour([.5], scalars='grain').smooth_taubin(n_iter=40, pass_band=.08).compute_normals(auto_orient_normals=True)
plot.add_mesh(grains, color='#cbb994', ambient=.22, diffuse=.75, specular=.18, specular_power=25, smooth_shading=True)

rng = np.random.default_rng(7)
pore = np.argwhere(~solid[:, : ny // 2 + 24, :4])                       # seeds on the inlet face, front part
pick = pore[rng.choice(len(pore), size=min(a.seeds, len(pore)), replace=False)]
seeds = pv.PolyData(np.c_[pick[:, 2] + .5, pick[:, 1] + .5, pick[:, 0] + .5] * h)
lines = grid.streamlines_from_source(seeds, vectors='velocity', max_length=nx * h * 6.0,
                                     initial_step_length=.4, integration_direction='forward')
bar = dict(title='velocity\nmagnitude\n(normalised)\n', color='black', vertical=True, position_x=.90, position_y=.20,
           width=.035, height=.55, title_font_size=int(a.width / 95), label_font_size=int(a.width / 110), n_labels=5, fmt='%.2f')
plot.add_mesh(lines.tube(radius=h * .42, n_sides=14), scalars='velocity magnitude', cmap='turbo', clim=(0, 1),
              ambient=.3, diffuse=.8, specular=.3, specular_power=30, smooth_shading=True, scalar_bar_args=bar)
plot.add_mesh(grid.outline(), color='#555555', line_width=max(2, a.width // 900))

centre = np.array([nx, ny, nz]) * h / 2
plot.camera_position = [tuple(centre + np.array([1.75, -2.05, 1.25]) * nx * h), tuple(centre + np.array([.0, 0, -.03]) * nx * h), (0, 0, 1)]
plot.camera.zoom(0.92)
# shift the view so the cube sits left of centre and the colour bar has a white margin of its own
plot.camera.SetWindowCenter(.20, 0.)
plot.enable_anti_aliasing('ssaa')
try:
    plot.enable_ssao(radius=h * 6, bias=h * .5, kernel_size=128, blur=True)   # contact shadows between grains
except Exception as error:                                                     # not available on every OpenGL stack
    print('ssao unavailable:', error)
plot.screenshot(a.output)
print('streamlines', lines.n_cells, 'size', (a.width, height), 'saved', a.output)
