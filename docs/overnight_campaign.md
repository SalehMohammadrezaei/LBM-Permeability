# Bounded local rock campaign

Run from the repository with the isolated environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u benchmarks/campaign.py \
  --config benchmarks/configs/overnight_rock.json \
  --output results/NEW_CAMPAIGN --budget 28800
```

The configuration accepts a dataset path/checksum/shape, ordered candidate crop
bounds in (z,y,x), spacing and numerical settings. Paths are local inputs, not
redistributed volumes. Source and configuration are frozen before work; changes
require a fresh output. Completed-case reuse validates the frozen source,
checksum-derived geometry identity and complete numerical settings. This is not
population checkpointing. Interrupted loads restart from rest and previous
attempts remain on disk. A resumed invocation explicitly records a new budget.

The parent uses no CUDA context and launches one worker at a time. Sixteen
physical cores are selected by CPU affinity; numerical library thread counts are
one in the launch command. Existing processes are untouched. Worker memory is
limited to 80% of free VRAM observed before that task, with device-wide reserve
observations every three seconds. This cannot prevent instantaneous competition
from another process; sampled measurements are not exact peaks. Available host
RAM must remain at least 50 GiB. Disk planning uses a 50 GiB campaign ceiling and
5 GiB filesystem reserve. Each worker has a wall limit, with final reporting time
reserved within the global budget.

Three staged 300-update pilots exercise several diagnostics and NPY/VTI export.
Largest eligible crop is chosen using memory and runtime costs before its
permeability is inspected. Main float64 equations and convergence thresholds are
unchanged. The selected crop must also pass a 300-update export preflight.
Fixed-step results deliberately have status `max_steps` and are not accepted
permeabilities. Baseline loads run independently from rest and accepted fields
are saved immediately. The x half-force and 100-fold tighter-tolerance column
checks have a predeclared 0.1% target. Optional full half-force tensor uses the
same 0.1% Frobenius target. No target is relaxed after seeing results.

Exact morphology retains its global default guard. An explicit per-campaign
size bound is supplied only after small measured profiles and a conservative
runtime estimate. Exhaustive covering-ball diameters are pore-volume weighted,
include isolated pores and use a finite solid exterior. This differs from the
flow's periodic exterior. No EDT histogram is substituted. The map, histogram,
CDF, percentiles and interior exclusions preserve the same sample support.

`field_export.write_vti` writes appended-raw VTK ImageData without an additional
full-volume vector allocation. Cells are voxels; origin is their lower corner,
spacing is physical metres, arrays are (z,y,x), vectors are (x,y,z), velocity
remains lattice units. A VTK-reader test checks a non-cubic fixture. VTK is an
optional plotting/test dependency, not a solver dependency. Baseline full fields
are separate NPY arrays and VTI files; they are excluded from the compact bundle.

`benchmarks/campaign_report.py CAMPAIGN` regenerates charts and a cutaway from
saved results. Conservative streamline interpolation rejects any stencil
containing solid; small checked integration segments stop at the displayed edge.
The solid display can be subsampled, but streamline integration uses original
fields and geometry. Unweighted seeds do not represent quantitative flux density.
VTK is optional for the 3D render; rendering errors are retained. Install it only
in the isolated environment (`.venv/bin/python -m pip install vtk`).

## Actual GPU architecture

D3Q19 uses dense direction-major populations `f[q*N+i]`, with x-fastest cells
inside (z,y,x) arrays. Two full population buffers are allocated, including
solid cells. Each launch uses 256 CUDA threads per block and one site per thread.
Collision and pull streaming are separate kernels. Solid-node population storage
and delayed reflection are preserved. This is not fused, sparse, in-place or
multi-GPU. All arithmetic is double precision; this campaign also stores double
populations. GPU reductions/macroscopic fields and previous velocity fields are
used by diagnostics, with scalar transfers/synchronization at checks. The mask
is copied to the device; accepted final fields are copied to the host for export.

Public-call wall time is synchronized and includes setup, required initial/final
checks and requested field transfers. Iteration-loop timing includes scheduled
checks. NPY/VTI writing is separate. Total-lattice and fluid-node MLUPS are named
explicitly. GPU pool live allocation, cached reservation, system-wide GPU memory
and process-lifetime high-water RSS are distinct. Sustained performance uses a
warmup and three measured 1,000-update calls with matching diagnostics. NumPy
is a reference implementation, not an optimized 96-core CPU solver.
