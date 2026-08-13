# Correctness audit — LBM permeability solver

Line-by-line audit of the D2Q9 / D3Q19 Stokes-flow permeability code. One real
bug was found and fixed; everything else checked out. Summary of what was
verified and what changed.

## The bug: collision was applied at solid nodes

**Where:** the BGK-collision + Guo-forcing update in all three solvers
(`d2q9.py`, `d3q19.py`, `d3q19_fast.py`).

**What was wrong:** the collision update `f += -(f-feq)/tau + S` was applied to
*every* lattice node, including solid ones. The intended wall model is halfway
bounce-back, which assumes the solid node still holds its **pre-collision**
populations when the reflection happens. Relaxing the solid populations toward a
(spurious) local equilibrium first corrupts what gets bounced back.

**Effect:** it injects about half a lattice spacing of slip at every wall, so
the *effective* aperture of a channel becomes `gap + 1` instead of `gap`.
Permeability scales like aperture³, so the error is:

| throat width | k overestimate |
|---|---|
| 20 px | ~16 % |
| 10 px | ~33 % |
| 3 px  | ~95 % |

The error is worst exactly where it hurts most — thin, few-pixel pore throats,
which dominate tight micromodels and digital-rock samples.

**The fix:** collide fluid nodes only; leave solid populations untouched so the
bounce-back reflects the correct pre-collision distribution.

- `d2q9.py` / `d3q19.py` (array): multiply the collision update by a fluid mask
  `fluid_f = (~blocked)`.
- `d3q19_fast.py` (fused CUDA kernel): the collide kernel now writes
  `solid ? f_q : collided` instead of the collided value unconditionally.

## Validation after the fix

Plane-Poiseuille flow between parallel plates has the exact superficial
permeability `k = gap³ / (12·N)`. After the fix, the measured aperture equals
the geometric `gap` and the permeability converges to the analytic value at
**second order**:

```
2D:        aperture=20.008 (gap=20)   k=16.708   analytic=16.667   err=0.25%
3D array : aperture=16.010 (gap=16)   k=10.708   analytic=10.667   err=0.39%
3D kernel: aperture=16.010 (gap=16)   k=10.708   analytic=10.667   err=0.39%

convergence (2D):  gap=10 -> 1.00%,  gap=20 -> 0.25%,  gap=40 -> 0.062%
                   (error quarters per doubling = 2nd order, as it should)
```

Independent, non-Poiseuille benchmark — Sangani & Acrivos (1982) square array of
cylinders, transverse Stokes flow:

```
 c (solid)   k/a^2 LBM    k/a^2 S&A    rel.err
   0.100     1.248e+00    1.257e+00     0.73%
   0.151     5.661e-01    5.728e-01     1.17%
   0.199     3.064e-01    3.123e-01     1.90%
   0.300     1.006e-01    1.160e-01    13.3%   (correlation + staircase error near c->0.4)
```

Note on the fused kernel: this machine has no CuPy/GPU, so the fused CUDA kernel
could not be compiled and run here. Its fix was instead validated by
transcribing the *exact* two-kernel algorithm (guarded collide + pull-based
bounce-back stream) into NumPy — that transcription reproduces the bug
(aperture=17) without the guard and the correct aperture=16 with it, matching
the array path. Re-run the Poiseuille check on a GPU box to confirm end-to-end.

## Residual (not a bug)

A ~0.3 % permeability variation remains over `tau = 0.7…1.6`. This is the
well-known viscosity-dependent slip of BGK + bounce-back; it is minimized at
`tau = 1` (the default) and would require TRT/MRT to remove entirely. Left as-is.

## Everything else — verified correct

- **Lattice constants:** D2Q9 and D3Q19 `CX/CY/CZ`, weights `W`, opposite-index
  tables — all consistent (Σw = 1, Σ w·cc = cs²·I, opposite pairs correct).
- **Equilibrium** `feq = w·ρ·(1 + 3(c·u) + 4.5(c·u)² − 1.5u²)` — correct (cs²=1/3).
- **Guo forcing** `S = w·(1 − 1/2τ)·(3 cF + 9 cu·cF − 3 uF)` with the half-force
  velocity correction `u = (Σf·c + F/2)/ρ` — correct.
- **Momentum sums** in `_mom_{x,y,z}` — match the velocity-set tables.
- **Viscosity / permeability:** `ν = (τ−½)/3`, `k = <u>·ν/F` with ρ=1, μ=ν — correct.
- **Streaming** (periodic roll; kernel pull) and **bounce-back** (pair swap;
  kernel pull-opposite) — correct.
- **Unit chain** (`units.py`): `k_LU·dx²` → m², `/9.869233e-16` → mD — correct.
- **Geometry, drivers, curve, cylinder validation:** `True = solid` convention
  used consistently throughout; F selection and conversions correct.

## Files changed

```
lbm_permeability/d2q9.py        collide fluid nodes only
lbm_permeability/d3q19.py       collide fluid nodes only
lbm_permeability/d3q19_fast.py  collide kernel: keep solid populations unchanged
tests/test_poiseuille.py        reference formula corrected to gap^3/(12N);
                                tightened tolerance; assert 2nd-order convergence
```
