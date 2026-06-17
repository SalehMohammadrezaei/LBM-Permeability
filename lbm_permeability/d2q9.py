"""2D single-phase lattice-Boltzmann Stokes solver (D2Q9, BGK, Guo forcing).

Computes the absolute (Darcy) permeability of a binary pore-scale image by
driving flow with a uniform body force and measuring the steady-state
superficial velocity.

Recipe
------
* D2Q9 velocity set, single-relaxation-time (BGK) collision.
* Guo body force (replaces the pressure gradient: at steady state rho*F
  balances grad P).
* Half-way bounce-back at solid cells -> no-slip walls.
* Fully periodic boundaries in both directions.
* Steady state declared when the mean speed <|u|> stops changing.

The solver runs on the GPU through CuPy when available and transparently
falls back to NumPy on the CPU.
"""
from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:  # pragma: no cover - depends on the machine
    HAS_GPU = False

# ---------- D2Q9 lattice ----------
CX = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1], dtype=np.int8)
CY = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=np.int8)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
              1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 7, 8, 5, 6], dtype=np.int8)


def lbm_stokes(blocked, F_x=0.0, F_y=0.0, tau=1.0,
               n_steps_max=50000, conv_tol=1e-5, conv_window=200,
               use_gpu=True, verbose=True):
    """Run D2Q9 Stokes flow until ``<|u|>`` converges.

    Parameters
    ----------
    blocked : (Ny, Nx) bool array
        ``True`` marks solid / immobile cells.
    F_x, F_y : float
        Body-force components in lattice units (per unit mass).
    tau : float
        BGK relaxation time (``tau = 1.0`` gives nu = 1/6).
    n_steps_max : int
        Safety cap on iterations.
    conv_tol : float
        Stop when the relative change in ``<|u|>`` over ``conv_window``
        steps drops below this value.
    use_gpu : bool
        Use CuPy when it is installed.

    Returns
    -------
    dict
        Keys: ``ux``, ``uy`` (velocity fields), ``u_x_mean_total``,
        ``u_y_mean_total`` (superficial velocities over the whole domain),
        ``nu``, ``F_x``, ``F_y``, ``step_converged``, ``elapsed_s``.
    """
    xp = cp if (use_gpu and HAS_GPU) else np
    blocked_d = xp.asarray(blocked, dtype=xp.bool_)

    Ny, Nx = blocked.shape
    nu = (tau - 0.5) / 3.0
    half_inv_tau = 1.0 - 0.5 / tau  # Guo prefactor

    # Initialise distributions at rest (rho = 1, u = 0  ->  f_q = w_q).
    f = xp.zeros((9, Ny, Nx), dtype=xp.float64)
    for q in range(9):
        f[q] = W[q]

    cx = xp.asarray(CX, dtype=xp.float64)
    cy = xp.asarray(CY, dtype=xp.float64)
    w = xp.asarray(W, dtype=xp.float64)

    u_hist = []
    t0 = time.time()
    converged_step = n_steps_max
    ux = uy = None

    for step in range(n_steps_max):
        # --- macroscopic moments (with Guo half-force correction) ---
        rho = f.sum(axis=0)
        mom_x = f[1] - f[2] + f[5] - f[6] - f[7] + f[8] + 0.5 * F_x
        mom_y = f[3] - f[4] + f[5] + f[6] - f[7] - f[8] + 0.5 * F_y
        rho_safe = xp.where(rho > 1e-12, rho, 1.0)
        ux = xp.where(blocked_d, 0.0, mom_x / rho_safe)
        uy = xp.where(blocked_d, 0.0, mom_y / rho_safe)

        # --- equilibrium + Guo forcing source ---
        u2 = ux * ux + uy * uy
        uF = ux * F_x + uy * F_y
        feq = xp.empty_like(f)
        S = xp.empty_like(f)
        for q in range(9):
            cu = cx[q] * ux + cy[q] * uy
            cF = cx[q] * F_x + cy[q] * F_y
            feq[q] = w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
            S[q] = w[q] * half_inv_tau * (3.0 * cF + 9.0 * cu * cF - 3.0 * uF)

        # --- BGK collision + body force ---
        f += -(f - feq) / tau + S

        # --- streaming (periodic via roll) ---
        for q in range(9):
            f[q] = xp.roll(f[q], shift=(int(CY[q]), int(CX[q])), axis=(0, 1))

        # --- half-way bounce-back at solid cells ---
        f_pre = f.copy()
        for q in range(1, 9):
            f[q] = xp.where(blocked_d, f_pre[int(OPP[q])], f[q])

        # --- convergence check ---
        if step % conv_window == 0 and step > 0:
            u_mean = float(xp.sqrt(ux * ux + uy * uy).mean())
            u_hist.append(u_mean)
            if len(u_hist) >= 2:
                rel = abs(u_mean - u_hist[-2]) / max(u_mean, 1e-30)
                if verbose and step % 1000 == 0:
                    print(f"  step={step:>6}  <|u|>={u_mean:.4e}  rel={rel:.2e}")
                if rel < conv_tol and u_mean > 0:
                    converged_step = step
                    if verbose:
                        print(f"  converged at step {step}  <|u|>={u_mean:.4e}")
                    break

    # --- final macroscopic field ---
    rho = f.sum(axis=0)
    mom_x = f[1] - f[2] + f[5] - f[6] - f[7] + f[8] + 0.5 * F_x
    mom_y = f[3] - f[4] + f[5] + f[6] - f[7] - f[8] + 0.5 * F_y
    rho_safe = xp.where(rho > 1e-12, rho, 1.0)
    ux = xp.where(blocked_d, 0.0, mom_x / rho_safe)
    uy = xp.where(blocked_d, 0.0, mom_y / rho_safe)

    to_np = (lambda a: cp.asnumpy(a)) if (use_gpu and HAS_GPU) else (lambda a: a)
    return {
        "ux": to_np(ux), "uy": to_np(uy),
        "u_x_mean_total": float(ux.mean()),
        "u_y_mean_total": float(uy.mean()),
        "nu": nu, "F_x": F_x, "F_y": F_y,
        "step_converged": converged_step,
        "elapsed_s": time.time() - t0,
    }
