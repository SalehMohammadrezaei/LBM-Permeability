"""Fused-kernel D2Q9 Stokes solver — same numerics as :mod:`d2q9`, far faster,
with permeability-based convergence.

The array solver streams with ``roll`` and fires ~50 elementwise kernels plus
temporaries per step (~5x off peak bandwidth). This module does the whole step
in **two** custom CUDA kernels (collide + pull-stream-with-bounce-back), one
read + one write of ``f`` each — the 2D counterpart of :mod:`d3q19_fast`.

It also stops on the change in **permeability** (``|dk/k| < conv_tol``), which
settles long before the full velocity field does — the right criterion for
porous media with large stagnant/dead-end pore volume, where a velocity-based
criterion chases slow recirculation that does not affect k.
"""
from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False

from .d2q9 import CX, CY, W, OPP


def _src(real):
    cx = ",".join(str(int(v)) for v in CX)
    cy = ",".join(str(int(v)) for v in CY)
    opp = ",".join(str(int(v)) for v in OPP)
    ww = ",".join(repr(float(v)) for v in W)
    return f"""
__device__ const int    CXc[9] = {{{cx}}};
__device__ const int    CYc[9] = {{{cy}}};
__device__ const int    OPPc[9] = {{{opp}}};
__device__ const double Wc[9]  = {{{ww}}};

extern "C" __global__
void collide(const {real}* __restrict__ f, {real}* __restrict__ fo,
             const unsigned char* __restrict__ solid, const long N,
             const double Fx, const double Fy, const double tau, const double hit) {{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double fq[9];
    #pragma unroll
    for (int q = 0; q < 9; q++) fq[q] = (double)f[(long)q * N + i];
    double rho = 0.0;
    #pragma unroll
    for (int q = 0; q < 9; q++) rho += fq[q];
    double rs = rho > 1e-12 ? rho : 1.0;
    double ux = 0.0, uy = 0.0;
    if (solid[i] == 0) {{
        double mx = fq[1]-fq[2]+fq[5]-fq[6]-fq[7]+fq[8] + 0.5*Fx;
        double my = fq[3]-fq[4]+fq[5]+fq[6]-fq[7]-fq[8] + 0.5*Fy;
        ux = mx/rs; uy = my/rs;
    }}
    double u2 = ux*ux + uy*uy;
    double uF = ux*Fx + uy*Fy;
    #pragma unroll
    for (int q = 0; q < 9; q++) {{
        double cu = CXc[q]*ux + CYc[q]*uy;
        double cF = CXc[q]*Fx + CYc[q]*Fy;
        double feq = Wc[q]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
        double S   = Wc[q]*hit*(3.0*cF + 9.0*cu*cF - 3.0*uF);
        // collide fluid nodes ONLY (keep solid populations for correct bounce-back)
        fo[(long)q * N + i] = solid[i] ? ({real})fq[q] : ({real})(fq[q] - (fq[q]-feq)/tau + S);
    }}
}}

extern "C" __global__
void stream(const {real}* __restrict__ fc, {real}* __restrict__ fo,
            const unsigned char* __restrict__ solid, const int nx, const int ny) {{
    long N = (long)nx * ny;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = (int)(i % nx);
    int y = (int)(i / nx);
    bool s = solid[i] != 0;
    #pragma unroll
    for (int q = 0; q < 9; q++) {{
        int srcq, sx, sy;
        if (s) {{ srcq = OPPc[q]; sx = (x + CXc[q] + nx) % nx; sy = (y + CYc[q] + ny) % ny; }}
        else   {{ srcq = q;       sx = (x - CXc[q] + nx) % nx; sy = (y - CYc[q] + ny) % ny; }}
        long si = (long)sy * nx + sx;
        fo[(long)q * N + i] = fc[(long)srcq * N + si];
    }}
}}
"""


_MODULES = {}


def _module(precision):
    if precision not in _MODULES:
        real = "float" if precision == "float32" else "double"
        _MODULES[precision] = cp.RawModule(code=_src(real), options=("--use_fast_math",))
    return _MODULES[precision]


def lbm_stokes_2d_fast(blocked, F_x=1e-6, F_y=0.0, tau=1.0,
                       n_steps_max=400000, conv_tol=1e-5, conv_window=500,
                       precision="float64", verbose=True, heartbeat=2000):
    """Fused-kernel D2Q9 Stokes solver (GPU only). Returns the same keys as
    :func:`lbm_permeability.d2q9.lbm_stokes` plus ``k_lu`` (permeability in
    cells^2 along the driven direction). Convergence is on ``|dk/k|``."""
    if not HAS_GPU:
        raise RuntimeError("lbm_stokes_2d_fast requires a CUDA GPU (CuPy).")
    dt = cp.float32 if precision == "float32" else cp.float64
    Ny, Nx = blocked.shape
    N = Nx * Ny
    nu = (tau - 0.5) / 3.0
    hit = 1.0 - 0.5 / tau
    F = F_x if abs(F_x) >= abs(F_y) else F_y
    driven_x = abs(F_x) >= abs(F_y)

    mod = _module(precision)
    k_collide = mod.get_function("collide")
    k_stream = mod.get_function("stream")

    solid = cp.ascontiguousarray(cp.asarray(blocked, dtype=cp.uint8).reshape(-1))
    blocked_d = cp.asarray(blocked, dtype=cp.bool_)
    fa = cp.empty((9, Ny, Nx), dtype=dt)
    for q in range(9):
        fa[q] = W[q]
    fb = cp.empty_like(fa)

    tpb = 256
    bpg = (N + tpb - 1) // tpb
    cN = np.int64(N)
    args_c = (solid, cN, np.float64(F_x), np.float64(F_y), np.float64(tau), np.float64(hit))
    args_s = (solid, np.int32(Nx), np.int32(Ny))

    def superficial():
        rho = fa.sum(axis=0)
        rs = cp.where(rho > 1e-12, rho, 1.0)
        if driven_x:
            mom = fa[1]-fa[2]+fa[5]-fa[6]-fa[7]+fa[8] + 0.5*F_x
        else:
            mom = fa[3]-fa[4]+fa[5]+fa[6]-fa[7]-fa[8] + 0.5*F_y
        u = cp.where(blocked_d, 0.0, mom / rs)
        return float(u.mean())

    t0 = time.time()
    k_prev = None
    converged_step = n_steps_max
    for step in range(n_steps_max):
        k_collide((bpg,), (tpb,), (fa, fb) + args_c)
        k_stream((bpg,), (tpb,), (fb, fa) + args_s)

        if step % conv_window == 0 and step > 0:
            u_mean = superficial()
            k_lu = u_mean * nu / F
            if verbose and step % heartbeat == 0:
                cp.cuda.runtime.deviceSynchronize()
                dk = abs(k_lu - k_prev) / max(abs(k_lu), 1e-30) if k_prev is not None else 1.0
                print(f"    step={step:>7}/{n_steps_max}  k={k_lu:.6e}  dk/k={dk:.2e}  "
                      f"{(time.time()-t0)/step*1000:.2f} ms/step", flush=True)
            if k_prev is not None:
                dk = abs(k_lu - k_prev) / max(abs(k_lu), 1e-30)
                if dk < conv_tol and k_lu > 0:
                    converged_step = step
                    if verbose:
                        print(f"    converged at step {step}  k={k_lu:.6e}  dk/k={dk:.2e}", flush=True)
                    break
            k_prev = k_lu

    cp.cuda.runtime.deviceSynchronize()
    # final fields
    rho = fa.sum(axis=0)
    rs = cp.where(rho > 1e-12, rho, 1.0)
    mom_x = fa[1]-fa[2]+fa[5]-fa[6]-fa[7]+fa[8] + 0.5*F_x
    mom_y = fa[3]-fa[4]+fa[5]+fa[6]-fa[7]-fa[8] + 0.5*F_y
    ux = cp.where(blocked_d, 0.0, mom_x / rs)
    uy = cp.where(blocked_d, 0.0, mom_y / rs)
    k_lu = float((ux if driven_x else uy).mean()) * nu / F
    result = {
        "ux": cp.asnumpy(ux), "uy": cp.asnumpy(uy),
        "u_x_mean_total": float(ux.mean()), "u_y_mean_total": float(uy.mean()),
        "nu": nu, "F_x": F_x, "F_y": F_y, "k_lu": k_lu,
        "step_converged": converged_step, "elapsed_s": time.time() - t0,
        "precision": precision,
    }
    del fa, fb, solid, blocked_d
    cp.get_default_memory_pool().free_all_blocks()
    return result
