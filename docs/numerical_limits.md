# Numerical conventions and operating limits

## Model and boundary staging

The solver uses D2Q9 or D3Q19, the quadratic low-Mach equilibrium and Guo forcing,
with either BGK collision (`collision='bgk'`, the default, unchanged) or the
two-relaxation-time collision (`collision='trt'`). It approximates creeping flow only in a sufficiently weak-driving
regime. It is not an inertia-free Stokes discretization. Distributions initialize
at rho=1 and equilibrium at rest. The initial reported velocity includes the
Guo half-force correction.

The existing reflection scheme is retained: collide fluid nodes only; stream
periodically into all nodes; swap opposite populations at solid nodes. A population
entering a solid is returned on the subsequent streaming step. This is solid-node
reflection with a storage delay, rather than a newly implemented link-wise reflection.
For the tested steady axis-aligned channels the continuum comparison places walls
half a cell outside the outermost fluid centres, so h is the number of fluid rows.
With BGK the finite-grid permeability has tau-dependent error and no universal
optimal tau is claimed. With TRT the symmetric rate `1/tau` sets the viscosity and
the antisymmetric rate follows from `magic=(tau-1/2)(1/omega_minus-1/2)`. The default
`magic=3/16` places the wall exactly half-way for an axis-aligned channel at every
tau: each node carries the analytical parabola, and the volume-averaged permeability
is `(h**3/12+h/24)/Ny`, where `h/24` is midpoint quadrature of an exact profile
(`tests/test_trt.py`). Setting `magic=(tau-1/2)**2` recovers BGK. Sloping and curved voxel surfaces need their own resolution studies.

## Force, averaging, coordinates, units

`F_x,F_y,F_z` are force **per volume**, used directly in the Guo source and as
`F/2` in the momentum before division by rho. Reference density is one and
`nu=(tau-0.5)/3`; `mu_ref=rho_ref*nu=nu`. Darcy response is
`U=K*(-grad(p)+rho_ref*g)/mu_ref`. All solids contribute zero velocity and remain
in the averaging denominator. No connectivity pruning occurs.

Arrays are `(y,x)` or `(z,y,x)`; vectors and tensor rows/columns are `(x,y[,z])`.
For independent force load j, `K[i,j]=nu*mean_total(u_i)/F_j`. Raw columns are
retained. Reciprocity is `norm(K-K.T)/max(norm(K),tiny)` (Frobenius norm).
Principal eigenvectors are columns of `eigh((K+K.T)/2)` in ascending eigenvalue
order. Signed cross terms and zero eigenvalues are allowed; no eigenvalue clipping
or forced symmetry is used. Near-degenerate eigenvectors are not unique.

`K_m2=K_lu*dx_m**2`, for one positive isotropic voxel size. Anisotropic spacing
is rejected. A 2D image is not a substitute for a 3D rock calculation.
Forcing, viscosity and exported velocities/density are in lattice units unless
an explicit conversion is supplied. Voxel spacing alone converts permeability,
but does not determine physical velocity or pressure: those additionally require
a time scale and density scale. No physical forcing or viscosity was inferred
from the digital-rock reference fields.

## Acceptance and result interpretation

All entry points share strict Boolean mask and parameter validation. Convert
images with `decode_mask(image, solid_value=255, pore_value=0)` or other explicitly
declared labels. No implicit thresholding or NaN replacement is performed.

Each result states `termination_reason`, `converged`, `iterations_completed`, and
`valid_for_permeability`. Steps count completed updates. `step_converged` is null
unless accepted, with zero for the all-solid zero solution. Normal scalar/tensor
extraction refuses unconverged/invalid results. `allow_unconverged=True` is an
explicit finite diagnostic estimate, never an acceptance override in the result.
For multiple forcing components a scalar quotient is ambiguous; use independent
loads through the tensor API.

Periodic degenerate geometries: all-solid with nonzero load has valid zero response
without allocation; zero forcing cannot identify permeability; fully fluid forced
periodic flow has no finite steady resistance and returns an invalid special status.
Nonpositive fluid density or nonfinite populations/velocity cause failure. There is
no velocity clipping. Stability checks default to every 50 steps with mandatory
final checks; set `stability_every=1` for debugging. Detection between samples can
be delayed. Timeout is checked between steps; in-flight GPU work and final diagnostics
can exceed the requested wall limit slightly. `None` disables timeout; zero is invalid.

Convergence requires three consecutive sampled full-vector AND RMS velocity-field
changes below absolute-plus-relative tolerances, after a minimum duration. The
previous velocity field costs 2 or 3 additional double arrays. Periodic population
mass (including solid storage nodes) must conserve its initial total within the
specified relative tolerance. Field residuals compare the same post-update state
that is returned. Mach uses max fluid speed divided by `1/sqrt(3)`; acceptance
also requires `mach_max<=max_mach` (default .05). Reynolds diagnostic is
`norm(U)/phi * characteristic_length / nu`; the default length is one lattice
cell and is not a measured pore scale. Supply a physically meaningful length.
The initial campaign targets Re<.1 and verifies force sensitivity separately.

A converged result certifies these numerical checks, not continuum accuracy,
creeping-flow independence, correct segmentation or representative volume.

## Backend and precision scope

| Path | Storage | Collision arithmetic | Driving |
|---|---|---|---|
| NumPy reference | float64 | float64 | periodic body force, 2D/3D |
| CuPy array reference | float64 | float64 | periodic body force, 2D/3D |
| CUDA kernels | float64 or float32 | float64 | periodic body force, 2D/3D |
| Experimental pressure CUDA | float64 or float32 | float64 | 2D x pressure planes |

Float32 storage is mixed precision. Diagnostic moments are accumulated in float64.
Array paths reject float32 rather than silently ignoring it. CUDA indexes have a
checked 64-bit ABI; dimension arguments are int32. Fast-math is disabled for this
verification baseline. Explicit GPU selection fails on an unusable runtime;
auto mode probes allocation/elementwise execution. OOM never triggers CPU fallback.
The NumPy solver is a readable reference, not an optimized 96-core solver.
The legacy `mempool_flush` argument is validated for call compatibility; the
shared periodic loop does not periodically flush CuPy's allocator cache. Memory
planning uses measured allocator reservation. Benchmark runners may clear unused
blocks belonging to their own process between cases.

## Experimental 2D pressure mode

The existing Zou-He density reconstruction is preserved. It fixes rho_in=1 and
rho_out=1-3*deltaP, with zero tangential velocity on fluid boundary nodes.
Signed reverse drops are supported, zero drop and nonpositive prescribed density
are rejected. Both original sample faces must contain fluid. Artificial x
reservoirs and user-supplied `walls_y` behavior remain available.

Permeability uses signed fluid-mean pressures at original sample face centres:
`G=(p_in-p_out)/(Nx_sample-1)` and `k=nu*U_sample/G`. U uses the entire ORIGINAL
sample volume, excluding added reservoirs and wall rows. This uses dynamic
viscosity at rho_ref=1 and must remain weakly compressible. Histories monitor
full fields, velocity vector, pressure-gradient stability and inlet/outlet mass
flux; acceptance requires the span of cross-sectional mass flux across every x
section within `flux_tol`. Padding changes the boundary-value problem. Pressure
and body-force porous samples are not interchangeable independent benchmarks.
No 3D pressure path is implemented.

## Morphology

Total porosity is exact pore count divided by original voxel count. Optional local
thickness uses SciPy EDT with an exhaustive sweep over distinct radii. For every
pore centre c define r(c) as distance to the nearest solid **centre**. The largest
open ball `distance(x,c)<r(c)` covering x supplies diameter `2*r(c)*dx`. A union of
eligible balls is computed at each radius; this is not an EDT histogram. Discrete
voxel-centre radii have boundary/voxelization bias relative to continuum surfaces.

The exterior is explicitly a one-voxel solid layer: this is a crop-conditioned
finite-image measurement with artificial exterior walls. Periodic morphology is
rejected, not simulated with an undocumented 27-fold tiling. Histogram mass weights
each pore voxel equally; isolated pores are included. Units are metres, bin mass
sums to one, PDF units are inverse metres, and cumulative values are undersize.
No-pore masks give an empty/no-pore result. An all-fluid finite image is bounded
by the declared exterior walls. Default maximum ROI is 2,100,000 voxels; the exact
radius sweep is intended for small documented ROIs, not full 500-cubed morphology.
PoreSpy dependency resolution failed on the pilot Python 3.14 environment; its
integration is not claimed. SciPy's actual version is saved in every PSD result.

## Connectivity diagnostics

`topology.connectivity_report` explicitly selects finite or periodic boundaries
and face adjacency (4/6) or lattice-link adjacency (8 in 2D, 18 in 3D). Finite
spanning porosity joins fluid on the two planes of a specified array axis.
Periodic conducting geometry requires a winding cycle, detected using lifted
integer coordinates; merely touching both faces does not count. The periodic
graph is limited to 250,000 voxels by default. Diagonal-only contact can connect
LBM links while face adjacency is disconnected, indicating unresolved geometry.
These reports never remove pores and always divide by the entire original volume.
They are geometric diagnostics, not replacements for a converged flow solution.

For nondegenerate periodic flow, forcing below one machine epsilon of the
population storage type is rejected: such a drive can leave populations at rest
while the half-force velocity alone gives a false finite quotient. Passing this
conservative representability guard does not establish a safe force range. In
particular, the tested porous float32 cases at 1e-6 and 1e-5 failed strict
convergence, while the x-load at 1e-4 passed against float64 (a narrow verified
case, not a general float32 guarantee). Absolute convergence tolerances must also
be small relative to resolved velocities, and force/tolerance sensitivity remains
necessary. Zero-force and all-solid special states retain their explicit semantics.
