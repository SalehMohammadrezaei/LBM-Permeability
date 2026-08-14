"""Pressure-driven D2Q9 Stokes permeability solver (GPU).

Instead of a periodic body force, the flow is driven by a fixed pressure
(density) difference between an inlet and an outlet plane, with the lateral
direction periodic and no-slip (bounce-back) at solids. On long, elongated
samples this reaches steady state much faster than the body-force/periodic
approach, because the pressure gradient is imposed at the boundaries rather
than self-established through slow density diffusion.

Method
------
* BGK collision + halfway bounce-back at solids (fused collide/stream kernels
  shared with :mod:`d2q9_fast`, periodic; the inlet/outlet planes are then
  corrected each step).
* Zou & He (1997) pressure boundary conditions at x=0 (rho_in) and x=Nx-1
  (rho_out = rho_in - deltaP*3), applied on the fluid cells of those planes.
* A few pure-fluid buffer columns are padded at inlet/outlet so the pressure
  planes sit in fluid (standard practice).
* Darcy permeability  k = nu * <u_x> / (deltaP/(Nx-1)),  <u_x> = x-velocity
  averaged over the whole domain (superficial velocity).
* Convergence on the average kinetic energy: stop when its coefficient of
  variation over a ~1000-step window falls below ``energy_eps`` (default 1e-4).
"""
from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False

from .d2q9 import W
from .d2q9_fast import _module


def lbm_stokes_2d_pressure(blocked, deltaP=1e-4, tau=1.0, pad=4,
                           n_steps_max=400000, energy_eps=1e-4, energy_window=1000,
                           sample_every=50, precision="float64", verbose=True,
                           heartbeat=5000, return_fields=True):
    """Pressure-driven D2Q9 permeability. ``blocked`` is (Ny,Nx) bool (True=solid),
    flow driven along +x by pressure drop ``deltaP`` (lattice units). Returns a
    dict with ``k_lu`` (cells^2), the velocity fields (unpadded), and run metadata."""
    if not HAS_GPU:
        raise RuntimeError("requires a CUDA GPU (CuPy).")
    dt = cp.float32 if precision == "float32" else cp.float64
    Ny, Nx0 = blocked.shape
    if pad > 0:
        col = np.zeros((Ny, pad), dtype=bool)
        blk = np.concatenate([col, blocked, col], axis=1)
    else:
        blk = np.asarray(blocked, dtype=bool)
    Ny, Nx = blk.shape
    N = Nx * Ny
    nu = (tau - 0.5) / 3.0
    rho_in = 1.0
    rho_out = 1.0 - deltaP * 3.0                      # invCs2 = 3

    mod = _module(precision)
    k_collide = mod.get_function("collide")
    k_stream = mod.get_function("stream")

    solid = cp.ascontiguousarray(cp.asarray(blk, dtype=cp.uint8).reshape(-1))
    blocked_d = cp.asarray(blk, dtype=cp.bool_)
    fa = cp.empty((9, Ny, Nx), dtype=dt)
    for q in range(9):
        fa[q] = W[q]
    fb = cp.empty_like(fa)

    tpb = 256
    bpg = (N + tpb - 1) // tpb
    cN = np.int64(N)
    args_c = (solid, cN, np.float64(0.0), np.float64(0.0), np.float64(tau), np.float64(0.0))
    args_s = (solid, np.int32(Nx), np.int32(Ny))

    inlet_fluid = ~blocked_d[:, 0]                    # (Ny,)
    outlet_fluid = ~blocked_d[:, -1]

    def zou_he(f):
        # inlet x=0: unknown incoming q1,q5,q8 (cx=+1); rho fixed, u_y=0
        c = f[:, :, 0]
        C = c[0] + c[3] + c[4]
        B = c[2] + c[6] + c[7]
        ux = 1.0 - (C + 2.0 * B) / rho_in
        f1 = c[2] + (2.0 / 3.0) * rho_in * ux
        half = 0.5 * (c[3] - c[4])
        f5 = c[7] - half + (1.0 / 6.0) * rho_in * ux
        f8 = c[6] + half + (1.0 / 6.0) * rho_in * ux
        f[1, :, 0] = cp.where(inlet_fluid, f1, c[1])
        f[5, :, 0] = cp.where(inlet_fluid, f5, c[5])
        f[8, :, 0] = cp.where(inlet_fluid, f8, c[8])
        # outlet x=Nx-1: unknown incoming q2,q6,q7 (cx=-1); rho fixed, u_y=0
        d = f[:, :, -1]
        C = d[0] + d[3] + d[4]
        A = d[1] + d[5] + d[8]
        ux = -1.0 + (C + 2.0 * A) / rho_out
        f2 = d[1] - (2.0 / 3.0) * rho_out * ux
        half = 0.5 * (d[3] - d[4])
        f6 = d[8] - half - (1.0 / 6.0) * rho_out * ux
        f7 = d[5] + half - (1.0 / 6.0) * rho_out * ux
        f[2, :, -1] = cp.where(outlet_fluid, f2, d[2])
        f[6, :, -1] = cp.where(outlet_fluid, f6, d[6])
        f[7, :, -1] = cp.where(outlet_fluid, f7, d[7])

    fluid_d = ~blocked_d
    # columns bounding the actual sample (just inside the fluid padding)
    c_in, c_out = pad, Nx - 1 - pad
    fcount_in = float(fluid_d[:, c_in].sum())
    fcount_out = float(fluid_d[:, c_out].sum())

    def fields():
        rho = fa.sum(axis=0)
        rs = cp.where(rho > 1e-12, rho, 1.0)
        ux = cp.where(blocked_d, 0.0, (fa[1]-fa[2]+fa[5]-fa[6]-fa[7]+fa[8]) / rs)
        uy = cp.where(blocked_d, 0.0, (fa[3]-fa[4]+fa[5]+fa[6]-fa[7]-fa[8]) / rs)
        return ux, uy, rho

    def permeability(ux, rho):
        # superficial velocity over the sample region (padding stripped)
        meanU = float(ux[:, c_in:c_out + 1].mean())
        # true pressure gradient from the density field at the sample faces
        p_in = float((rho[:, c_in] * fluid_d[:, c_in]).sum()) / max(fcount_in, 1) / 3.0
        p_out = float((rho[:, c_out] * fluid_d[:, c_out]).sum()) / max(fcount_out, 1) / 3.0
        L = max(c_out - c_in, 1)
        gradP = (p_in - p_out) / L
        return (nu * meanU / gradP if gradP > 0 else float("nan")), meanU

    e_hist = []
    k_hist = []
    t0 = time.time()
    converged_step = n_steps_max
    for step in range(n_steps_max):
        k_collide((bpg,), (tpb,), (fa, fb) + args_c)
        k_stream((bpg,), (tpb,), (fb, fa) + args_s)
        zou_he(fa)

        if step % sample_every == 0 and step > 0:
            ux, uy, rho = fields()
            E = float((ux * ux + uy * uy).mean())
            e_hist.append(E)
            nwin = max(2, energy_window // sample_every)
            if len(e_hist) >= nwin:
                w = np.array(e_hist[-nwin:])
                cov = w.std() / max(abs(w.mean()), 1e-30)
                k_lu, meanU = permeability(ux, rho)
                k_hist.append((step, k_lu))
                if verbose and step % heartbeat == 0:
                    print(f"    step={step:>7}  k={k_lu:.6e}  E_cov={cov:.2e}  "
                          f"{(time.time()-t0)/step*1000:.2f} ms/step", flush=True)
                if cov < energy_eps:
                    converged_step = step
                    if verbose:
                        print(f"    converged at step {step}  k={k_lu:.6e}  E_cov={cov:.2e}", flush=True)
                    break

    cp.cuda.runtime.deviceSynchronize()
    ux, uy, rho = fields()
    k_lu, meanU = permeability(ux, rho)
    out = {
        "k_lu": k_lu, "nu": nu, "deltaP": deltaP, "u_x_mean_total": meanU,
        "step_converged": converged_step, "elapsed_s": time.time() - t0,
        "Nx_padded": Nx, "pad": pad, "precision": precision,
        "k_history": np.array(k_hist) if k_hist else np.zeros((0, 2)),
    }
    if return_fields:
        # strip the padding columns so fields align with the input image
        uxn = cp.asnumpy(ux)[:, pad:pad + Nx0] if pad > 0 else cp.asnumpy(ux)
        uyn = cp.asnumpy(uy)[:, pad:pad + Nx0] if pad > 0 else cp.asnumpy(uy)
        out["ux"], out["uy"] = uxn, uyn
    del fa, fb, solid, blocked_d
    cp.get_default_memory_pool().free_all_blocks()
    return out
