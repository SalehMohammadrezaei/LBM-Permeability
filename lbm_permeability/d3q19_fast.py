"""Fused-kernel D3Q19 Stokes solver — same numerics as :mod:`d3q19`, far faster.

The array solver fires ~40 CuPy kernels and allocates several full-volume
temporaries per step, so it runs at only ~2 % of the GPU's memory bandwidth.
This module does the whole step in **two** custom CUDA kernels:

* ``collide`` — BGK relaxation + Guo body force (one read + one write of ``f``);
* ``stream``  — periodic streaming with on-node bounce-back at solid voxels,
  written as a pull so it is also one read + one write.

It reproduces the array solver's scheme exactly (BGK + Guo, periodic streaming,
solid-node pair-swap bounce-back), so ``float64`` agrees with :func:`d3q19.lbm_stokes_3d`
to rounding.  ``float32`` halves the memory and is faster still, at ~1e-3 relative
accuracy on the permeability.
"""
from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False

from .d3q19 import CX, CY, CZ, W, _mom_x, _mom_y, _mom_z

# opposite-direction index for every q (from d3q19.OPP_PAIRS)
_OPP = [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]


def _src(real):
    cx = ",".join(str(int(v)) for v in CX)
    cy = ",".join(str(int(v)) for v in CY)
    cz = ",".join(str(int(v)) for v in CZ)
    opp = ",".join(str(v) for v in _OPP)
    ww = ",".join(repr(float(v)) for v in W)
    return f"""
__device__ const int   CXc[19] = {{{cx}}};
__device__ const int   CYc[19] = {{{cy}}};
__device__ const int   CZc[19] = {{{cz}}};
__device__ const int   OPPc[19] = {{{opp}}};
__device__ const double Wc[19] = {{{ww}}};

extern "C" __global__
void collide(const {real}* __restrict__ f, {real}* __restrict__ fo,
             const unsigned char* __restrict__ solid, const long N,
             const double Fx, const double Fy, const double Fz,
             const double tau, const double hit) {{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double fq[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fq[q] = (double)f[(long)q * N + i];
    double rho = 0.0;
    #pragma unroll
    for (int q = 0; q < 19; q++) rho += fq[q];
    double rs = rho > 1e-12 ? rho : 1.0;
    double ux = 0.0, uy = 0.0, uz = 0.0;
    if (solid[i] == 0) {{
        double mx = fq[1]-fq[2]+fq[7]-fq[8]+fq[9]-fq[10]+fq[11]-fq[12]+fq[13]-fq[14] + 0.5*Fx;
        double my = fq[3]-fq[4]+fq[7]+fq[8]-fq[9]-fq[10]+fq[15]-fq[16]+fq[17]-fq[18] + 0.5*Fy;
        double mz = fq[5]-fq[6]+fq[11]+fq[12]-fq[13]-fq[14]+fq[15]+fq[16]-fq[17]-fq[18] + 0.5*Fz;
        ux = mx/rs; uy = my/rs; uz = mz/rs;
    }}
    double u2 = ux*ux + uy*uy + uz*uz;
    double uF = ux*Fx + uy*Fy + uz*Fz;
    #pragma unroll
    for (int q = 0; q < 19; q++) {{
        double cu = CXc[q]*ux + CYc[q]*uy + CZc[q]*uz;
        double cF = CXc[q]*Fx + CYc[q]*Fy + CZc[q]*Fz;
        double feq = Wc[q]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
        double S   = Wc[q]*hit*(3.0*cF + 9.0*cu*cF - 3.0*uF);
        fo[(long)q * N + i] = ({real})(fq[q] - (fq[q]-feq)/tau + S);
    }}
}}

extern "C" __global__
void stream(const {real}* __restrict__ fc, {real}* __restrict__ fo,
            const unsigned char* __restrict__ solid,
            const int Nx, const int Ny, const int Nz) {{
    long N = (long)Nx * Ny * Nz;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = (int)(i % Nx);
    int y = (int)((i / Nx) % Ny);
    int z = (int)(i / ((long)Nx * Ny));
    bool s = solid[i] != 0;
    #pragma unroll
    for (int q = 0; q < 19; q++) {{
        int srcq, sx, sy, sz;
        if (s) {{                       // solid node: pull opposite from x + c_q
            srcq = OPPc[q];
            sx = (x + CXc[q] + Nx) % Nx; sy = (y + CYc[q] + Ny) % Ny; sz = (z + CZc[q] + Nz) % Nz;
        }} else {{                       // fluid node: pull same dir from x - c_q
            srcq = q;
            sx = (x - CXc[q] + Nx) % Nx; sy = (y - CYc[q] + Ny) % Ny; sz = (z - CZc[q] + Nz) % Nz;
        }}
        long si = ((long)sz * Ny + sy) * Nx + sx;
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


def lbm_stokes_3d_fast(blocked, F_x=1e-6, F_y=0.0, F_z=0.0, tau=1.0,
                       n_steps_max=20000, conv_tol=1e-4, conv_window=500,
                       precision="float64", verbose=True, return_fields=False,
                       heartbeat=500, wall_timeout_s=7200):
    """Fused-kernel D3Q19 Stokes solver (GPU only).  Same result dict as
    :func:`lbm_permeability.d3q19.lbm_stokes_3d`.  ``precision`` is
    ``"float64"`` (default, matches the reference) or ``"float32"`` (half the
    memory, faster, ~1e-3 relative accuracy)."""
    if not HAS_GPU:
        raise RuntimeError("lbm_stokes_3d_fast requires a CUDA GPU (CuPy).")
    dt = cp.float32 if precision == "float32" else cp.float64
    Nz, Ny, Nx = blocked.shape
    N = Nx * Ny * Nz
    nu = (tau - 0.5) / 3.0
    hit = 1.0 - 0.5 / tau

    mod = _module(precision)
    k_collide = mod.get_function("collide")
    k_stream = mod.get_function("stream")

    solid = cp.ascontiguousarray(cp.asarray(blocked, dtype=cp.uint8).reshape(-1))
    blocked_d = cp.asarray(blocked, dtype=cp.bool_)
    fa = cp.empty((19, Nz, Ny, Nx), dtype=dt)
    for q in range(19):
        fa[q] = W[q]
    fb = cp.empty_like(fa)

    tpb = 256
    bpg = (N + tpb - 1) // tpb
    cN = np.int64(N)
    args_c = (solid, cN, np.float64(F_x), np.float64(F_y), np.float64(F_z),
              np.float64(tau), np.float64(hit))
    args_s = (solid, np.int32(Nx), np.int32(Ny), np.int32(Nz))

    def macro_umean():
        rho = fa.sum(axis=0)
        rs = cp.where(rho > 1e-12, rho, 1.0)
        ux = cp.where(blocked_d, 0.0, _mom_x(fa, F_x) / rs)
        uy = cp.where(blocked_d, 0.0, _mom_y(fa, F_y) / rs)
        uz = cp.where(blocked_d, 0.0, _mom_z(fa, F_z) / rs)
        return ux, uy, uz

    u_hist = []
    t0 = time.time(); last_hb = t0
    converged_step = n_steps_max
    for step in range(n_steps_max):
        k_collide((bpg,), (tpb,), (fa, fb) + args_c)
        k_stream((bpg,), (tpb,), (fb, fa) + args_s)

        if verbose and step > 0 and step % heartbeat == 0:
            cp.cuda.runtime.deviceSynchronize()
            ms = (time.time() - last_hb) / heartbeat * 1000
            last_hb = time.time()
            print(f"    step={step:>6}/{n_steps_max}  elapsed={time.time()-t0:.0f}s  {ms:.1f} ms/step", flush=True)

        if wall_timeout_s and (time.time() - t0) > wall_timeout_s:
            cp.cuda.runtime.deviceSynchronize()
            print(f"    !! wall timeout at step {step}; returning current state", flush=True)
            converged_step = step
            break

        if step % conv_window == 0 and step > 0:
            ux, uy, uz = macro_umean()
            u_mean = float(cp.sqrt(ux*ux + uy*uy + uz*uz).mean())
            u_hist.append(u_mean)
            if len(u_hist) >= 2:
                rel = abs(u_mean - u_hist[-2]) / max(u_mean, 1e-30)
                if rel < conv_tol and u_mean > 0:
                    converged_step = step
                    if verbose:
                        print(f"    converged at step {step}  <|u|>={u_mean:.4e}", flush=True)
                    break

    cp.cuda.runtime.deviceSynchronize()
    ux, uy, uz = macro_umean()
    result = {
        "u_x_mean_total": float(ux.mean()),
        "u_y_mean_total": float(uy.mean()),
        "u_z_mean_total": float(uz.mean()),
        "nu": nu, "F_x": F_x, "F_y": F_y, "F_z": F_z,
        "step_converged": converged_step,
        "elapsed_s": time.time() - t0,
        "precision": precision,
    }
    if return_fields:
        result["ux"], result["uy"], result["uz"] = (cp.asnumpy(ux), cp.asnumpy(uy), cp.asnumpy(uz))
    del fa, fb, solid, blocked_d
    cp.get_default_memory_pool().free_all_blocks()
    return result
