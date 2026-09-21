"""Render the README figure: grains of a real rock with TRT streamlines coloured by speed."""
import argparse
import numpy as np
import pyvista as pv

p = argparse.ArgumentParser(); p.add_argument('--fields', required=True); p.add_argument('--output', required=True)
p.add_argument('--voxel-um', type=float, default=5.0); a = p.parse_args()
d = np.load(a.fields)
solid = d['solid']; nz, ny, nx = solid.shape
speed_scale = float(np.percentile(np.sqrt(d['ux'] ** 2 + d['uy'] ** 2 + d['uz'] ** 2)[~solid], 99.5))

grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(a.voxel_um,) * 3)
order = lambda f: np.ascontiguousarray(f).ravel(order='C')          # (z,y,x) C order == VTK x-fastest
grid['solid'] = order(solid.astype(np.float32))
vel = np.stack([order(d['ux']), order(d['uy']), order(d['uz'])], axis=1) / speed_scale
grid['velocity'] = vel
grid['speed'] = np.linalg.norm(vel, axis=1)

pv.OFF_SCREEN = True
plot = pv.Plotter(off_screen=True, window_size=(2000, 1500))
plot.set_background('white')
# grains: keep the back half so the flow in the front half is visible
back = grid.clip(normal=(0, -1, 0), origin=(0, ny * a.voxel_um * .5, 0))
grains = back.contour([.5], scalars='solid')
plot.add_mesh(grains.smooth(n_iter=30), color='#d9cfc1', opacity=1.0, specular=.15, smooth_shading=True)
rng = np.random.default_rng(7)
pore = np.argwhere(~solid[:, : ny // 2 + 20, :4])                     # seeds on the inlet face, front part
pick = pore[rng.choice(len(pore), size=min(900, len(pore)), replace=False)]
seeds = pv.PolyData(np.c_[pick[:, 2] + .5, pick[:, 1] + .5, pick[:, 0] + .5] * a.voxel_um)
lines = grid.streamlines_from_source(seeds, vectors='velocity', max_length=nx * a.voxel_um * 6.0,
                                     initial_step_length=.5, integration_direction='forward')
plot.add_mesh(lines.tube(radius=a.voxel_um * .45), scalars='speed', cmap='turbo', clim=(0, 1),
              scalar_bar_args=dict(title='speed / 99.5th percentile', color='black', vertical=False,
                                   position_x=.3, position_y=.03, width=.4, height=.05, title_font_size=30, label_font_size=26))
plot.add_mesh(grid.outline(), color='black', line_width=2)
plot.camera_position = [(nx * a.voxel_um * 2.3, -ny * a.voxel_um * 1.9, nz * a.voxel_um * 1.6),
                        (nx * a.voxel_um * .5, ny * a.voxel_um * .5, nz * a.voxel_um * .45), (0, 0, 1)]
plot.enable_anti_aliasing('ssaa')
plot.screenshot(a.output)
print('streamlines', lines.n_cells, 'saved', a.output)
