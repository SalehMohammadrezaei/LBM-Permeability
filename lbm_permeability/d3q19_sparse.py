"""CUDA D3Q19 on pore voxels only: indirect addressing, fused pull and collision.

Only fluid nodes are stored.  A neighbour table gives, for every fluid node and
direction q, the compact index of the node at x - c_q; a negative entry marks a
solid neighbour, where half-way bounce-back returns the node's own opposite
post-collision population.  Solid voxels cost no memory and no GPU threads, and
the kernel needs no coordinate arithmetic.  Populations are stored as deviations
f_q - w_q, so float32 storage keeps the small flow signal.  Steady states agree with the dense
solid-node reflection, which delivers the same population one step later.
"""
from __future__ import annotations

import numpy as np

from .backends import cp
from .d3q19 import CX, CY, CZ, W
from .d3q19_fast import _OPP


def _src(real):
    cx = ",".join(str(int(v)) for v in CX)
    cy = ",".join(str(int(v)) for v in CY)
    cz = ",".join(str(int(v)) for v in CZ)
    opp = ",".join(str(v) for v in _OPP)
    ww = ",".join(repr(float(v)) for v in W)
    pull = f"""
    double fq[19];
    fq[0] = Wc[0] + (double)fc[i];
    #pragma unroll
    for (int q = 1; q < 19; q++) {{
        int n = nbr[(int64_t)(q - 1) * M + i];
        fq[q] = Wc[q] + (n < 0 ? (double)fc[(int64_t)OPPc[q] * M + i] : (double)fc[(int64_t)q * M + n]);
    }}
    double rho = 0.0;
    #pragma unroll
    for (int q = 0; q < 19; q++) rho += fq[q];
    double ux = (fq[1]-fq[2]+fq[7]-fq[8]+fq[9]-fq[10]+fq[11]-fq[12]+fq[13]-fq[14] + 0.5*Fx)/rho;
    double uy = (fq[3]-fq[4]+fq[7]+fq[8]-fq[9]-fq[10]+fq[15]-fq[16]+fq[17]-fq[18] + 0.5*Fy)/rho;
    double uz = (fq[5]-fq[6]+fq[11]+fq[12]-fq[13]-fq[14]+fq[15]+fq[16]-fq[17]-fq[18] + 0.5*Fz)/rho;
"""
    return f"""
typedef long long int64_t;
static_assert(sizeof(int64_t) == 8, "64-bit index ABI required");
__device__ const int   CXc[19] = {{{cx}}};
__device__ const int   CYc[19] = {{{cy}}};
__device__ const int   CZc[19] = {{{cz}}};
__device__ const int   OPPc[19] = {{{opp}}};
__device__ const double Wc[19] = {{{ww}}};

extern "C" __global__
void step(const {real}* __restrict__ fc, {real}* __restrict__ fo,
          const int* __restrict__ nbr, const int64_t M,
          const double Fx, const double Fy, const double Fz,
          const double om_p, const double hit_p, const double om_m, const double hit_m) {{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
{pull}
    double u2 = ux*ux + uy*uy + uz*uz;
    double uF = ux*Fx + uy*Fy + uz*Fz;
    #pragma unroll
    for (int q = 0; q < 19; q++) {{
        // even and odd parts of the pair (q, opposite q); equal rates are BGK
        double cu = CXc[q]*ux + CYc[q]*uy + CZc[q]*uz;
        double cF = CXc[q]*Fx + CYc[q]*Fy + CZc[q]*Fz;
        double fb = fq[OPPc[q]];
        double even = om_p*(0.5*(fq[q]+fb) - Wc[q]*rho*(1.0 + 4.5*cu*cu - 1.5*u2)) - hit_p*Wc[q]*(9.0*cu*cF - 3.0*uF);
        double odd  = om_m*(0.5*(fq[q]-fb) - Wc[q]*rho*3.0*cu) - hit_m*Wc[q]*3.0*cF;
        fo[(int64_t)q * M + i] = ({real})((fq[q] - Wc[q]) - even - odd);
    }}
}}

extern "C" __global__
void moments(const {real}* __restrict__ fc, const int* __restrict__ nbr, const int64_t M,
             const double Fx, const double Fy, const double Fz,
             double* __restrict__ orho, double* __restrict__ oux,
             double* __restrict__ ouy, double* __restrict__ ouz) {{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
{pull}
    orho[i] = rho; oux[i] = ux; ouy[i] = uy; ouz[i] = uz;
}}
"""


_MODULES = {}


def _module(precision):
    if precision not in _MODULES:
        real = "float" if precision == "float32" else "double"
        _MODULES[precision] = cp.RawModule(code=_src(real), options=())
    return _MODULES[precision]


def neighbour_table(blocked):
    """(18, M) int32 compact index of the fluid node at x - c_q, or -1 for a solid one."""
    solid = cp.asarray(blocked)
    nz, ny, nx = blocked.shape
    index = cp.flatnonzero(~solid.ravel())
    count = int(index.size)
    if count >= np.iinfo(np.int32).max:
        raise ValueError('more fluid nodes than the int32 neighbour table can address')
    compact = cp.full(blocked.size, -1, dtype=cp.int32)
    compact[index] = cp.arange(count, dtype=cp.int32)
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    table = cp.empty((18, count), dtype=cp.int32)
    for q in range(1, 19):
        source = (((z - int(CZ[q])) % nz) * ny + (y - int(CY[q])) % ny) * nx + (x - int(CX[q])) % nx
        table[q - 1] = compact[source]
    del compact, x, y, z, source
    return table, index


class SparseD3Q19:
    def __init__(self, blocked, force, om_p, om_m, precision):
        self.shape = blocked.shape
        self.nbr, index = neighbour_table(blocked)
        self.index = cp.asnumpy(index)     # host copy: only field export needs it
        self.count = int(self.index.size)
        del index
        cp.get_default_memory_pool().free_all_blocks()
        self.f = cp.empty((19, self.count), dtype=precision)
        # Half-way bounce-back conserves the staggered momentum S=sum((-1)**x_a j_a) up to
        # the force: streaming flips its sign and collision adds F*D, D being the even/odd
        # imbalance of fluid nodes, so stored post-collision states obey S' = -S + F*D.
        # Starting from w_q alone (S=0) leaves an undamped step-alternating velocity
        # F*D/(2M).  Raw momentum +F/2 per node is the fixed point S=F*D/2, so the mode is
        # never excited; the first streamed state then reports u=(j+F/2)/rho=F.
        for q in range(19):
            self.f[q] = W[q] * (1.5 * (int(CX[q]) * force[0] + int(CY[q]) * force[1] + int(CZ[q]) * force[2]))   # deviation from w_q
        self.fb = cp.empty_like(self.f)
        mod = _module(precision)
        self._step, self._moments = mod.get_function('step'), mod.get_function('moments')
        self.blocks = ((self.count + 255) // 256,)
        self.force = tuple(np.float64(a) for a in force)
        self.rates = (np.float64(om_p), np.float64(1 - .5 * om_p), np.float64(om_m), np.float64(1 - .5 * om_m))
        self.out = [cp.empty(self.count, dtype=cp.float64) for _ in range(4)]

    def bytes(self):
        return dict(populations=int(self.f.nbytes + self.fb.nbytes), neighbour_table=int(self.nbr.nbytes),
                    moments=int(sum(a.nbytes for a in self.out)), storage='deviation f_q-w_q')

    def step(self):
        self._step(self.blocks, (256,), (self.f, self.fb, self.nbr, np.int64(self.count)) + self.force + self.rates)
        self.f, self.fb = self.fb, self.f

    def macros(self):
        """Velocity and density after streaming, the state the dense solver reports."""
        self._moments(self.blocks, (256,), (self.f, self.nbr, np.int64(self.count)) + self.force + tuple(self.out))
        rho, ux, uy, uz = self.out
        return (ux, uy, uz), rho

    def dense(self, values):
        field = np.zeros(int(np.prod(self.shape)))
        field[self.index] = cp.asnumpy(values)
        return field.reshape(self.shape)
