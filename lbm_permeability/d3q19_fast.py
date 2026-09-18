"""CUDA BGK/Guo kernels: selectable storage, double collision arithmetic."""
from __future__ import annotations

import time
import numpy as np

from .backends import cp, HAS_GPU

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
typedef long long int64_t;
static_assert(sizeof(int64_t) == 8, "64-bit index ABI required");
__device__ const int   CXc[19] = {{{cx}}};
__device__ const int   CYc[19] = {{{cy}}};
__device__ const int   CZc[19] = {{{cz}}};
__device__ const int   OPPc[19] = {{{opp}}};
__device__ const double Wc[19] = {{{ww}}};

extern "C" __global__
void collide(const {real}* __restrict__ f, {real}* __restrict__ fo,
             const unsigned char* __restrict__ solid, const int64_t N,
             const double Fx, const double Fy, const double Fz,
             const double tau, const double hit) {{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double fq[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fq[q] = (double)f[(int64_t)q * N + i];
    double rho = 0.0;
    #pragma unroll
    for (int q = 0; q < 19; q++) rho += fq[q];
    double rs = rho;
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
        // collide fluid nodes ONLY — colliding solids corrupts the bounce-back (wall slip)
        fo[(int64_t)q * N + i] = solid[i] ? ({real})fq[q] : ({real})(fq[q] - (fq[q]-feq)/tau + S);
    }}
}}

extern "C" __global__
void stream(const {real}* __restrict__ fc, {real}* __restrict__ fo,
            const unsigned char* __restrict__ solid,
            const int Nx, const int Ny, const int Nz) {{
    int64_t N = (int64_t)Nx * Ny * Nz;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = (int)(i % Nx);
    int y = (int)((i / Nx) % Ny);
    int z = (int)(i / ((int64_t)Nx * Ny));
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
        int64_t si = ((int64_t)sz * Ny + sy) * Nx + sx;
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


def lbm_stokes_3d_fast(blocked, F_x=1e-6, F_y=0., F_z=0., tau=1.,
                       n_steps_max=20000, conv_tol=1e-4, conv_window=500,
                       precision="float64", verbose=True, return_fields=False,
                       heartbeat=500, wall_timeout_s=7200, **controls):
    from .solver import periodic
    return periodic(blocked,(F_x,F_y,F_z),tau=tau,n_steps_max=n_steps_max,
                    conv_tol=conv_tol,conv_window=conv_window,backend="cuda",
                    precision=precision,verbose=verbose,heartbeat=heartbeat,
                    return_fields=return_fields,wall_timeout_s=wall_timeout_s,**controls)
