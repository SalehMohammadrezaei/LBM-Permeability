# LBM-Permeability

NumPy reference solvers, CuPy array reference solvers, and CUDA implementations
for permeability of binary pore images (D2Q9) and volumes (D3Q19). Supports
sequential 2×2/3×3 permeability tensors, porosity, optional local-thickness
morphology, and existing velocity-field output. The GPU-only 2D pressure solver
is experimental. There is no 3D pressure solver.

The current changes are under local verification. Read
[the numerical conventions and limits](docs/numerical_limits.md) and
[the technical handoff](CODE_AND_BENCHMARK_HANDOFF.md) before using results.
Existing MIT licensing and package author metadata are retained.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test,morphology]'
# Optional, on a compatible CUDA workstation:
python -m pip install '.[gpu]'
python -m pytest
```

The pilot was executed with Python 3.14.7. Older versions in the package's existing
`>=3.8` metadata have not all been tested; this is a compatibility validation gap.
CuPy import alone is insufficient: the package checks actual device execution.

## Small reproducible run

```bash
python -m lbm_permeability --demo --dimension 2 --backend numpy \
  --dx 2e-6 --steps 20000 --output results/demo
python -m lbm_permeability mask.npy --tensor --backend cuda \
  --solid-value 255 --pore-value 0 --dx 5e-6 --timeout 60 \
  --output results/tensor
```

Core masks must be Boolean (`True=solid`). Image labels must be declared explicitly.
Masks must be nonempty, forcing finite, tau>0.5, and voxel size a positive isotropic
scalar. A missing input does not launch a large demonstration. CLI failures have
nonzero exit codes and retain diagnostic output.

```python
from lbm_permeability import lbm_stokes, k_from_run, geometry
blocked = geometry.parallel_plates(48, 16, 24)
run = lbm_stokes(blocked, F_x=1e-6, backend="numpy", verbose=False)
k_lu = k_from_run(run, "x")  # rejects invalid or unconverged runs
```

`backend` can be `numpy`, `cupy-array`, `cuda`, `cuda-sparse`, or `auto`. The array
paths use float64; CUDA offers float32 or float64 storage with double collision
arithmetic. `cuda-sparse` (3D) stores pore voxels only, with a neighbour table,
half-way bounce-back and in-place streaming; memory and work scale with porosity,
and populations are kept as deviations from rest so float32 storage resolves weak
flow. Its steady states match `cuda` (`tests/test_sparse.py`).

`collision` is `bgk` (default) or `trt`. BGK with bounce-back places the wall at a
tau-dependent position, so permeability drifts with tau. TRT with `magic=3/16`
removes that dependence for axis-aligned walls (`tests/test_trt.py`):

```bash
python -m lbm_permeability mask.npy --tensor --backend cuda-sparse --collision trt \
  --solid-value 255 --pore-value 0 --dx 5e-6 --output results/tensor_trt
```
The NumPy reference is not an optimized many-core CPU solver. No general speedup
or float32 accuracy guarantee is made.

Permeability uses full sample volume averaging, force density and reference density
one: `K_ij=nu*U_i(load j)/F_j`, `nu=(tau-.5)/3`, `K_m2=K_lu*dx**2`. Arrays are
`(y,x)` or `(z,y,x)`; Cartesian components are `(x,y,z)`. Tensor columns are independent
loads; raw signed entries are retained. Principal values are explicitly derived
from the symmetric part. A legitimate singular tensor is allowed.

The low-Mach BGK model requires resolution, force and convergence sensitivity
checks. A stopped simulation is not necessarily converged, and a converged
simulation is not automatically an accurate continuum permeability. Fully fluid
forced periodic domains have no finite steady resistance. Crops are distinct
boundary-value problems, not grid refinement of a fixed rock sample.

Reproduction commands and archived failures are described in
[benchmark reproduction](docs/benchmark_reproduction.md). The repository includes
no manuscript or publication-readiness claim.
