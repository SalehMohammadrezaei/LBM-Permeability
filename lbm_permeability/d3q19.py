"""3D single-phase lattice-Boltzmann Stokes solver (D3Q19, BGK, Guo forcing).

Same numerics as the 2D solver (``d2q9``) extended to D3Q19, with a few
production niceties for the long runs that 3D volumes require:

* a heartbeat print so a multi-hour run is visibly alive,
* periodic release of the CuPy memory pool to avoid fragmentation,
* a wall-clock safety timeout,
* bounce-back done as in-place pair swaps (no copy of the 4D array).
"""
from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False

# ---------- D3Q19 lattice ----------
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int8)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int8)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
W = np.array([1 / 3,
              1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
              1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
              1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
# opposite-direction pairs (q, -q) for bounce-back
OPP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 10), (8, 9),
             (11, 14), (12, 13), (15, 18), (16, 17)]


def _mom_x(f, F_x):
    return (f[1] - f[2] + f[7] - f[8] + f[9] - f[10]
            + f[11] - f[12] + f[13] - f[14] + 0.5 * F_x)


def _mom_y(f, F_y):
    return (f[3] - f[4] + f[7] + f[8] - f[9] - f[10]
            + f[15] - f[16] + f[17] - f[18] + 0.5 * F_y)


def _mom_z(f, F_z):
    return (f[5] - f[6] + f[11] + f[12] - f[13] - f[14]
            + f[15] + f[16] - f[17] - f[18] + 0.5 * F_z)


def lbm_stokes_3d(blocked, F_x=1e-6, F_y=0.0, F_z=0.0, tau=1.0,
                  n_steps_max=20000, conv_tol=1e-4, conv_window=500,
                  use_gpu=True, verbose=True, return_fields=False,
                  heartbeat=500, mempool_flush=2000, wall_timeout_s=7200):
    """Run D3Q19 Stokes flow until ``<|u|>`` converges.

    Parameters mirror :func:`lbm_permeability.d2q9.lbm_stokes`.  ``blocked``
    is a ``(Nz, Ny, Nx)`` bool array.  Returns a dict with the superficial
    velocities ``u_{x,y,z}_mean_total`` plus run metadata.  Set
    ``return_fields=True`` to also get the full ``ux``, ``uy``, ``uz`` arrays
    (NumPy, for visualization) -- off by default to save memory on big volumes.
    """
    xp = cp if (use_gpu and HAS_GPU) else np
    blocked_d = xp.asarray(blocked, dtype=xp.bool_)
    mempool = cp.get_default_memory_pool() if (use_gpu and HAS_GPU) else None

    Nz, Ny, Nx = blocked.shape
    nu = (tau - 0.5) / 3.0
    half_inv_tau = 1.0 - 0.5 / tau

    f = xp.zeros((19, Nz, Ny, Nx), dtype=xp.float64)
    for q in range(19):
        f[q] = W[q]

    cx = xp.asarray(CX, dtype=xp.float64)
    cy = xp.asarray(CY, dtype=xp.float64)
    cz = xp.asarray(CZ, dtype=xp.float64)
    w = xp.asarray(W, dtype=xp.float64)

    u_hist = []
    t0 = time.time()
    last_hb = t0
    converged_step = n_steps_max

    for step in range(n_steps_max):
        # --- macros ---
        rho = f.sum(axis=0)
        rho_safe = xp.where(rho > 1e-12, rho, 1.0)
        ux = xp.where(blocked_d, 0.0, _mom_x(f, F_x) / rho_safe)
        uy = xp.where(blocked_d, 0.0, _mom_y(f, F_y) / rho_safe)
        uz = xp.where(blocked_d, 0.0, _mom_z(f, F_z) / rho_safe)

        # --- BGK collision + Guo forcing (per-direction to save memory) ---
        u2 = ux * ux + uy * uy + uz * uz
        uF = ux * F_x + uy * F_y + uz * F_z
        for q in range(19):
            cu = cx[q] * ux + cy[q] * uy + cz[q] * uz
            cF = cx[q] * F_x + cy[q] * F_y + cz[q] * F_z
            feq = w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
            S = w[q] * half_inv_tau * (3.0 * cF + 9.0 * cu * cF - 3.0 * uF)
            f[q] += -(f[q] - feq) / tau + S
        del u2, uF

        # --- streaming (periodic) ---
        for q in range(19):
            f[q] = xp.roll(f[q], shift=(int(CZ[q]), int(CY[q]), int(CX[q])),
                           axis=(0, 1, 2))

        # --- bounce-back via in-place pair swap ---
        for q_a, q_b in OPP_PAIRS:
            tmp_a = xp.where(blocked_d, f[q_b], f[q_a])
            f[q_b] = xp.where(blocked_d, f[q_a], f[q_b])
            f[q_a] = tmp_a
        del tmp_a

        # --- heartbeat ---
        if verbose and step > 0 and step % heartbeat == 0:
            ms = (time.time() - last_hb) / heartbeat * 1000
            last_hb = time.time()
            print(f"    step={step:>6}/{n_steps_max}  "
                  f"elapsed={time.time() - t0:.0f}s  {ms:.1f} ms/step", flush=True)

        # --- periodic GPU memory flush ---
        if mempool is not None and step > 0 and step % mempool_flush == 0:
            mempool.free_all_blocks()

        # --- wall-clock safety ---
        if wall_timeout_s and (time.time() - t0) > wall_timeout_s:
            print(f"    !! wall timeout at step {step}; returning current state",
                  flush=True)
            converged_step = step
            break

        # --- convergence check ---
        if step % conv_window == 0 and step > 0:
            u_mean = float(xp.sqrt(ux * ux + uy * uy + uz * uz).mean())
            u_hist.append(u_mean)
            if len(u_hist) >= 2:
                rel = abs(u_mean - u_hist[-2]) / max(u_mean, 1e-30)
                if rel < conv_tol and u_mean > 0:
                    converged_step = step
                    if verbose:
                        print(f"    converged at step {step}  <|u|>={u_mean:.4e}",
                              flush=True)
                    break

    # --- final macros ---
    rho = f.sum(axis=0)
    rho_safe = xp.where(rho > 1e-12, rho, 1.0)
    ux = xp.where(blocked_d, 0.0, _mom_x(f, F_x) / rho_safe)
    uy = xp.where(blocked_d, 0.0, _mom_y(f, F_y) / rho_safe)
    uz = xp.where(blocked_d, 0.0, _mom_z(f, F_z) / rho_safe)

    result = {
        "u_x_mean_total": float(ux.mean()),
        "u_y_mean_total": float(uy.mean()),
        "u_z_mean_total": float(uz.mean()),
        "nu": nu, "F_x": F_x, "F_y": F_y, "F_z": F_z,
        "step_converged": converged_step,
        "elapsed_s": time.time() - t0,
    }
    if return_fields:
        to_np = (lambda a: cp.asnumpy(a)) if (use_gpu and HAS_GPU) else (lambda a: a)
        result["ux"], result["uy"], result["uz"] = to_np(ux), to_np(uy), to_np(uz)
    del f, ux, uy, uz, rho, rho_safe, blocked_d
    if mempool is not None:
        mempool.free_all_blocks()
    return result
