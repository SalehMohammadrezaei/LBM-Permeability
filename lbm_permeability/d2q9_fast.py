"""CUDA BGK/Guo kernels: selectable storage, double collision arithmetic."""
from __future__ import annotations

import time
import numpy as np

from .backends import cp, HAS_GPU

from .d2q9 import CX, CY, W, OPP


def _src(real):
    cx = ",".join(str(int(v)) for v in CX)
    cy = ",".join(str(int(v)) for v in CY)
    opp = ",".join(str(int(v)) for v in OPP)
    ww = ",".join(repr(float(v)) for v in W)
    return f"""
typedef long long int64_t;
static_assert(sizeof(int64_t) == 8, "64-bit index ABI required");
__device__ const int    CXc[9] = {{{cx}}};
__device__ const int    CYc[9] = {{{cy}}};
__device__ const int    OPPc[9] = {{{opp}}};
__device__ const double Wc[9]  = {{{ww}}};

extern "C" __global__
void collide(const {real}* __restrict__ f, {real}* __restrict__ fo,
             const unsigned char* __restrict__ solid, const int64_t N,
             const double Fx, const double Fy, const double tau, const double hit) {{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double fq[9];
    #pragma unroll
    for (int q = 0; q < 9; q++) fq[q] = (double)f[(int64_t)q * N + i];
    double rho = 0.0;
    #pragma unroll
    for (int q = 0; q < 9; q++) rho += fq[q];
    double rs = rho;
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
        fo[(int64_t)q * N + i] = solid[i] ? ({real})fq[q] : ({real})(fq[q] - (fq[q]-feq)/tau + S);
    }}
}}

extern "C" __global__
void stream(const {real}* __restrict__ fc, {real}* __restrict__ fo,
            const unsigned char* __restrict__ solid, const int nx, const int ny) {{
    int64_t N = (int64_t)nx * ny;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = (int)(i % nx);
    int y = (int)(i / nx);
    bool s = solid[i] != 0;
    #pragma unroll
    for (int q = 0; q < 9; q++) {{
        int srcq, sx, sy;
        if (s) {{ srcq = OPPc[q]; sx = (x + CXc[q] + nx) % nx; sy = (y + CYc[q] + ny) % ny; }}
        else   {{ srcq = q;       sx = (x - CXc[q] + nx) % nx; sy = (y - CYc[q] + ny) % ny; }}
        int64_t si = (int64_t)sy * nx + sx;
        fo[(int64_t)q * N + i] = fc[(int64_t)srcq * N + si];
    }}
}}
"""


_MODULES = {}


def _module(precision):
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    if precision not in _MODULES:
        real = "float" if precision == "float32" else "double"
        _MODULES[precision] = cp.RawModule(code=_src(real), options=())
    return _MODULES[precision]


def lbm_stokes_2d_fast(blocked, F_x=1e-6, F_y=0., tau=1.,
                       n_steps_max=400000, conv_tol=1e-5, conv_window=500,
                       precision="float64", verbose=True, heartbeat=2000,
                       *, return_fields=True, wall_timeout_s=None, **controls):
    from .solver import periodic
    return periodic(blocked,(F_x,F_y),tau=tau,n_steps_max=n_steps_max,
                    conv_tol=conv_tol,conv_window=conv_window,backend="cuda",
                    precision=precision,verbose=verbose,heartbeat=heartbeat,
                    return_fields=return_fields,wall_timeout_s=wall_timeout_s,**controls)
