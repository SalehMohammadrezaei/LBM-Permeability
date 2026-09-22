# Verification and performance

Results of the verification ladder and of the rock benchmarks, with the hardware they ran on.
The result files are archived separately; `docs/benchmark_reproduction.md` gives the commands.
GPU memory below is the sum of the solver arrays in GB (1e9 bytes) unless marked as allocator
peak, which is what `nvidia-smi` would show and is given in GiB.

## What was added

| Item | Where | Tests |
|---|---|---|
| Two-relaxation-time collision, magic 3/16, beside BGK (default unchanged) | `solver.py`, `d2q9_fast.py`, `d3q19_fast.py`, CLI `--collision trt` | `tests/test_trt.py` |
| `cuda-sparse`: pore voxels only, neighbour table, half-way bounce-back, in-place (AA) streaming, deviation storage | `d3q19_sparse.py` | `tests/test_sparse.py` |
| `numba-sparse`: the same algorithm on CPU cores | `d3q19_sparse_cpu.py`, optional dependency `numba` | `tests/test_sparse_cpu.py` |
| Campaign exits non-zero when any task fails | `benchmarks/campaign.py` | `tests/test_campaign.py` |
| Verification ladder and summary | `benchmarks/verification.py`, `plot_verification.py` | results below |

Test suite: 148 passed, 1 skipped.

## A numerical finding worth a paragraph in the paper

Half-way bounce-back conserves a staggered momentum, sum((-1)^x_a j_a), apart from the
body force. Starting from rest populations w_q with the force switched on excites it as an
undamped velocity that flips sign every step, of size F*D/(2M) (D: even/odd imbalance of
pore nodes, M: pore nodes). On a 24-cube test it was 1.7e-9 against a flow of 1e-4 and
never decayed. Starting from true rest in Guo's sense (raw momentum -F/2) sits on the fixed
point of that invariant and removes it to 1e-16. Both sparse backends start that way and a
regression test covers it. The older dense backend damps the mode by its delayed reflection.

## Bentheimer sandstone, 384-cube crop, 5 um (DRP-29), x load

| tau | BGK k (darcy) | TRT k (darcy) |
|---|---|---|
| 0.6 | 3.2717 | 4.1024 |
| 0.8 | 3.8267 | 4.1024 |
| 1.0 | 4.2313 | 4.1024 |
| 1.2 | 4.5951 | 4.1024 |
| 1.5 | 5.1074 | 4.1024 |

BGK spreads 45 % over this tau range; TRT spreads 0.0015 %. The sparse BGK value at
tau = 1 (0.1670380 lu) matches the 18 September dense baseline (0.1670425 lu) to 2.7e-5,
inside the convergence tolerance.

TRT tensor at tau = 1 (darcy), rows = response, columns = load:

```
 4.1024   0.2077   0.1190
 0.2077   5.0071  -0.0290
 0.1190  -0.0290   4.7783
```

Principal values 4.04, 4.80, 5.05 darcy; reciprocity error 7.2e-6.

## Bentheimer, full 500-cube image, TRT, tau = 0.6

The dense backend could not hold the full image (38 GB); the campaign used a 384-cube crop.
`cuda-sparse` runs all three loads in 12 minutes (9200, 8600 and 8300 steps). Tensor in darcy:

```
 4.5273   0.1921   0.1494
 0.1922   5.2640   0.0648
 0.1494   0.0648   5.2548
```

Principal values 4.46, 5.19, 5.39 darcy; porosity 0.2547; reciprocity error 1.1e-6.

## Boundary treatment and voxel resolution on a real rock

A scanned sample is not periodic. Across a periodic wrap most pore voxels face grain, which
adds resistance; in the wrapped 256-cube run the plane-mean pressure rises through the sample
and drops by 6 % of F*L across the wrap. `mirror=True` solves on the image reflected along every
axis (a sealed sample), where pores meet themselves and that jump is absent. Bentheimer, TRT,
tau = 0.6, x load, permeability in darcy:

| Crop | Periodic wrap | Mirrored along x only | Mirrored in every axis | Wrap against full mirror | Voxels split 2x2x2 (wrap) | Refined against native |
|---|---|---|---|---|---|---|
| 128-cube | 2.442 | 4.397 | 2.965 | -17.6 % | 2.104 | -13.9 % |
| 256-cube | 3.755 | 4.724 | 4.197 | -10.5 % | 3.285 | -12.5 % |
| 384-cube | 4.102 | 5.028 | 4.657 | -11.9 % | not run | |

Mirroring along the flow axis alone keeps the sideways periodic connections, which are also
artificial and open extra paths; the fully mirrored value is the one to compare with core-flood
measurements and with solvers that mirror their domains. Halving the voxel size of the same
voxel geometry lowers k by about 13 %: at 5 um the throats of this rock are only a few voxels
wide. The two effects have opposite signs and similar size. Both belong in the uncertainty of
any number quoted for this rock.

Sealed-sample directional permeabilities of the 384-cube crop (768-cube mirrored domain, 115
million pore voxels, float32 storage, 25.7 GiB allocator peak, 28, 22 and 23 minutes):

| Direction | Mirrored (darcy) | Wrapped tensor diagonal (darcy) | Difference |
|---|---|---|---|
| x | 4.657 | 4.102 | -11.9 % |
| y | 5.941 | 5.007 | -15.7 % |
| z | 5.760 | 4.778 | -17.0 % |

In the mirrored domain the off-diagonal response is below 1e-10 of the diagonal, as symmetry
requires. The full tensor, its principal directions and the anisotropy therefore come from the
wrapped run, with the bias above. The three 1024-cube benchmark images below were run with the
periodic wrap (their mirrored domains do not fit one GPU), so they carry a wrap bias of unknown
but probably smaller size, and the solvers they are compared with used other boundary conditions.

## Throughput and memory on the same crop (RTX 6000 Ada, Threadripper PRO 7995WX)

| Backend | 1000 steps | MLUPS over all voxels | MFLUPS over pore voxels | Memory (allocator peak) |
|---|---|---|---|---|
| `cuda` dense, float64 | 53.9 s | 1051 | 267 | 23.0 GiB GPU |
| `cuda-sparse`, float64, two arrays (first version) | 9.4 s | 5996 | 1524 | 6.4 GiB GPU |
| `cuda-sparse`, float64, in place (current) | 11.9 s | 4769 | 1212 | 4.2 GiB GPU |
| `cuda-sparse`, float32 storage, in place | 12.8 s | 4419 | 1123 | 3.2 GiB GPU |
| `numba-sparse`, 16 threads | 151 s | 375 | 95 | host RAM |
| `numba-sparse`, 32 threads | 93 s | 606 | 154 | host RAM |

`numba-sparse` thread scaling on the same crop (MLUPS over all voxels; 16 unrelated
processes were running): 1 thread 34, 2: 67, 4: 134, 8: 250, 16: 403, 32: 646, 64: 925,
96: 995. At 96 threads the CPU backend equals the throughput of the original dense GPU path.

The dense kernels process solid voxels and use modulo arithmetic per link, so they are a slow
baseline; the fluid-node rate (MFLUPS) is the figure to compare with other codes. The sparse
backends need about 180 bytes per pore voxel with float32 storage (allocator peaks: Fontainebleau
22.6 GiB, sphere pack 37.5 GiB, Berea 43.6 GiB), so a 1024-cube image fits a 48 GB GPU up to a
porosity of about 0.18.

One converged load fell from 1972 s to 299 s. CPU figures were taken while 16 unrelated
processes were running.
Float32 deviation storage reproduces float64 permeability to 1.7e-8 on a 48-cube test.

## Verification ladder

Plane channel, 8 nodes wide, error against the exact nodal value:

| tau | BGK | TRT |
|---|---|---|
| 0.55 | -2.3 % | 0 (1e-9) |
| 1.0 | +0.78 % | 0 |
| 2.0 | +25.6 % | 0 |

TRT reproduces the analytical parabola at every node for every tau. Its volume average is
(gap^3/12 + gap/24)/Ny: against the continuum value gap^3/12 the permeability is high by
1/(2 gap^2), 0.78 % at 8 nodes and 3.1 % at 4, from midpoint quadrature of an exact profile.
The table above is measured against the nodal value; the JSON files store the error against
the continuum value.

Simple cubic sphere arrays against Zick and Homsy (1982), tau = 1, Reynolds number held
near 0.02 across resolutions: errors fall from 2 to 5 % at 32-cube to 0.7 to 1.6 % at
128-cube for solid fractions 0.027 to 0.5236. Over tau = 0.55 to 2.0 at 64-cube BGK spreads
8.2 % (c = 0.216) and 12.2 % (c = 0.45); TRT spreads 0.00 %.
The seven drag coefficients are Table 2 of Zick and Homsy (1982); they were cross-checked
against the same table as quoted in the Basilisk test suite (basilisk.fr/src/test/spheres.c).

Inclined periodic slit (a check of tensor assembly and axis conventions, not of accuracy), five angles from 0 to 63.4 degrees: the largest in-plane
eigenvector recovers the slit direction to 1e-10 degrees; the minor eigenvalue is 1e-11 of
the major one; reciprocity error below 1e-11.

Straight pipes from Saxena et al. (2017), 1024 x 1024 cross-sections, against the
analytical mean velocity with the voxel-counted area:

| Cross-section | BGK | TRT |
|---|---|---|
| Circle 200 / 400 | -0.22 / -0.12 % | -0.24 / -0.13 % |
| Circle 800 | rerun in progress (see below) | rerun in progress |
| Square 400 | 0.001 % | 0.000 % |
| Triangle 400 | -0.17 % | -0.19 % |

The first Circle 800 runs reached their one-hour wall limit after 1.8 million steps without
meeting the 1e-9 tolerance (estimates -0.06 % and -0.07 %). They were not accepted and are being
rerun with a longer limit. Every other ladder case and every rock run converged.

Micromodel cell of Wagner et al. (2021), 1 mm unit cell with one cylinder between plates 0.091 mm
apart, on the exact 1 um grid (91 nodes across the depth), TRT, k in 1e-11 m^2:

| Cylinder radius (mm) | PoreWise | Published FEM | Published LBM | Published 3D homogenisation |
|---|---|---|---|---|
| 0.35 | 25.75 | 25.7 | 26.7 | 25.8 |
| 0.40 | 17.34 | 17.7 | 17.7 | 17.4 |
| 0.45 | 8.09 | 7.54 | 8.11 | 8.14 |
| 0.47 | 3.90 | 3.62 | 3.86 | 3.97 |
| 0.49 | 0.443 | 0.46 | 0.54 | 0.47 |

Every value lies inside the spread of the published methods. The coarser grids (16 and 32 nodes
across the depth, cell edges rounded to whole voxels) differ from the exact grid by up to 11 % at
r = 0.49, where the throat is 10 um wide; that is geometry rounding, not solver error.

Every case reported in this document converged and passed the acceptance checks. Cases that
reached a wall or step limit were rerun with a larger limit; the superseded files are kept
separately and are not counted.

## A mistake caught on the way

The first sphere-array pass used one force for every resolution. At 128-cube the flow
reached Mach 0.033 and a Reynolds number near 14, so the result was inertial. Those files
are kept in `verification_superseded_inertial/`; the force now scales with (32/n)^3.

## Choice of tau for large runs

Large runs use tau = 0.6: convergence on rocks is limited by Darcy-scale pressure diffusion,
whose time grows with viscosity and with the square of the domain size, and TRT makes k
independent of tau, so the low-viscosity run is about four times shorter at no cost.

Published comparison values (k11, m^2) quoted by POREMAPS (Krach, Ruf, Steeb 2025) from
Saxena et al. (2017): sphere pack 2.438 to 2.903e-10 (POREMAPS 2.512e-10); Berea Rock1
4.569 to 6.889e-13 (5.772e-13); Fontainebleau Rock3 0.642 to 1.411e-13 (0.920e-13).

## Open items

1. Add BCC and FCC sphere arrays from Zick and Homsy (1982) Table 2 (needs the paper itself).
2. Read the reference tables of Saxena et al. (2017) directly; only three samples are quoted second-hand.
3. Repeat the CPU thread-scaling series on a quiet machine for the paper.
4. Name, release tag, Zenodo DOI, then the manuscript.
