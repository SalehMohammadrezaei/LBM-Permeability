# TRT collision and pore-only backends: status, 20 September 2026

Branch `trt-collision` (on GitHub). `main` is unchanged. All numbers below come from
`results/trt-sparse-20260920/` (stored on D:), produced by the commits on this branch.
The 18 September campaign is untouched and keeps its own labels.

## What was added

| Item | Where | Tests |
|---|---|---|
| Two-relaxation-time collision, magic 3/16, beside BGK (default unchanged) | `solver.py`, `d2q9_fast.py`, `d3q19_fast.py`, CLI `--collision trt` | `tests/test_trt.py` |
| `cuda-sparse`: pore voxels only, neighbour table, half-way bounce-back, in-place (AA) streaming, deviation storage | `d3q19_sparse.py` | `tests/test_sparse.py` |
| `numba-sparse`: the same algorithm on CPU cores | `d3q19_sparse_cpu.py`, optional dependency `numba` | `tests/test_sparse_cpu.py` |
| Campaign exits non-zero when any task fails | `benchmarks/campaign.py` | `tests/test_campaign.py` |
| Verification ladder and summary | `benchmarks/verification.py`, `plot_verification.py` | results below |

Test suite: 147 passed, 1 skipped.

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

## Throughput and memory on the same crop (RTX 6000 Ada, Threadripper PRO 7995WX)

| Backend | 1000 steps | MLUPS (all voxels) | Memory |
|---|---|---|---|
| `cuda` dense, float64 | 53.9 s | 1051 | 23.0 GB GPU |
| `cuda-sparse`, float64, two arrays (first version) | 9.4 s | 5996 | 6.4 GB GPU |
| `cuda-sparse`, float64, in place (current) | 11.9 s | 4769 | 4.2 GB GPU |
| `cuda-sparse`, float32 storage, in place | 12.8 s | 4419 | 3.2 GB GPU |
| `numba-sparse`, 16 threads | 151 s | 375 | host RAM |
| `numba-sparse`, 32 threads | 93 s | 606 | host RAM |

`numba-sparse` thread scaling on the same crop (MLUPS over all voxels; 16 unrelated
processes were running): 1 thread 34, 2: 67, 4: 134, 8: 250, 16: 403, 32: 646, 64: 925,
96: 995. At 96 threads the CPU backend equals the throughput of the original dense GPU path.

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
(gap^3/12 + gap/24)/Ny; the gap/24 term is midpoint quadrature of an exact profile.

Simple cubic sphere arrays against Zick and Homsy (1982), tau = 1, Reynolds number held
near 0.02 across resolutions: errors fall from 2 to 5 % at 32-cube to 0.7 to 1.6 % at
128-cube for solid fractions 0.027 to 0.5236. Over tau = 0.55 to 2.0 at 64-cube BGK spreads
8.2 % (c = 0.216) and 12.2 % (c = 0.45); TRT spreads 0.00 %.
The seven drag coefficients are Table 2 of Zick and Homsy (1982); they were cross-checked
against the same table as quoted in the Basilisk test suite (basilisk.fr/src/test/spheres.c).

Inclined periodic slit, five angles from 0 to 63.4 degrees: the largest in-plane
eigenvector recovers the slit direction to 1e-10 degrees; the minor eigenvalue is 1e-11 of
the major one; reciprocity error below 1e-11.

Straight pipes from Saxena et al. (2017), 1024 x 1024 cross-sections, against the
analytical mean velocity with the voxel-counted area:

| Cross-section | BGK | TRT |
|---|---|---|
| Circle 200 / 400 / 800 | -0.22 / -0.12 / -0.06 % | -0.24 / -0.13 / -0.07 % |
| Square 400 | 0.001 % | 0.000 % |
| Triangle 400 | -0.17 % | -0.19 % |

Micromodel cell of Wagner et al. (2021), k in 1e-11 m^2 (TRT, finest grid run):

| Cylinder radius (mm) | This code | Published FEM | Published LBM | Published 3D homogenisation |
|---|---|---|---|---|
| 0.35 | 25.71 (32 nodes deep) | 25.7 | 26.7 | 25.8 |
| 0.40 | 17.34 (91 nodes, exact 1 um geometry) | 17.7 | 17.7 | 17.4 |
| 0.45 | 8.10 (32) | 7.54 | 8.11 | 8.14 |
| 0.47 | 3.94 (32) | 3.62 | 3.86 | 3.97 |
| 0.49 | 0.443 (91, exact) | 0.46 | 0.54 | 0.47 |

## Published 1024-cube benchmarks of Saxena et al. (2017), TRT, tau = 0.6, x load

| Sample | Porosity | This code k11 (m^2) | LBM range in Saxena et al. | POREMAPS | Hardware, time |
|---|---|---|---|---|---|
| Rock3, Fontainebleau, 2.072 um | 0.0953 | 0.798e-13 | 0.642 to 1.411e-13 | 0.920e-13 | one RTX 6000 Ada, 18.5 GB, 189400 steps, 5.0 h |
| Sphere pack, 788x791x793, 7 um | 0.3433 | 2.720e-10 | 2.438 to 2.903e-10 | 2.512e-10 | same GPU, 28.4 GB, 28800 steps, 1.2 h (pore Re 0.44; a ten times weaker force is being run as a check) |
| Rock1, Berea, 2.114 um | 0.184 | 5.079e-13 | 4.569 to 6.889e-13 | 5.772e-13 | `numba-sparse`, 48 CPU threads, 42400 steps, 11.5 h |

All three fall inside the spread of the LBM solvers in Saxena et al. (2017). POREMAPS reports
36 h on two cluster nodes for its Berea case (mirrored to 2048-cube, so not a like-for-like time).

## A mistake caught on the way

The first sphere-array pass used one force for every resolution. At 128-cube the flow
reached Mach 0.033 and a Reynolds number near 14, so the result was inertial. Those files
are kept in `verification_superseded_inertial/`; the force now scales with (32/n)^3.

## Running when this was last updated

* Berea Rock1 on the GPU (float32 storage) as a CPU-against-GPU cross-check on a full rock.
* Sphere pack with a ten times weaker force.

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
