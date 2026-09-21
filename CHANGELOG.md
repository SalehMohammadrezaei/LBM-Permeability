# Changelog

## 0.1.0

First public release. The project was developed as LBM-Permeability and renamed PoreWise;
`import lbm_permeability` still works and warns.

### Capabilities
- Absolute permeability in one direction, or the full tensor with principal values and
  directions, from 2D (D2Q9) and 3D (D3Q19) binary images, periodic body-force-driven flow
- BGK and two-relaxation-time (magic parameter 3/16) collision
- Backends: `numpy`, `cupy-array`, `cuda` (dense), `cuda-sparse` and `numba-sparse`
  (pore voxels only, half-way bounce-back, in-place streaming, float32 or float64 storage)
- Acceptance checks on convergence, mass, density and Mach number; JSON output with
  provenance; VTI field export; porosity, connectivity and local-thickness tools
- Command line interface `python -m porewise`

### Verification
See `docs/verification.md`: plane channel, pipes, simple cubic sphere arrays (Zick and Homsy
1982), inclined-slit tensor test, the micromodel benchmark of Wagner et al. (2021) and the
1024-cube images of Saxena et al. (2017).

### Limitations
Periodic boundaries only; single-phase creeping flow; binary images with isotropic voxels;
one GPU (no MPI or multi-GPU); pore-only backends are 3D only; the 2D pressure-driven solver
is experimental.

### Results in the accompanying paper
The benchmark results were produced by commits `b89d47f` to `8b70ed6` of this repository
(before the rename); each result file records its own `source_commit`. The numerical kernels
are unchanged in this release.
