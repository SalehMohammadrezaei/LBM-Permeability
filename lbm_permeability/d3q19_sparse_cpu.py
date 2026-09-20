"""Multi-core D3Q19 on pore voxels only; the CPU twin of d3q19_sparse (same algorithm).

Neighbour table, half-way bounce-back, AA-pattern in-place streaming, deviation
storage f_q - w_q and the rest start are identical to the CUDA backend, so the two
agree to round-off apart from summation order.  Requires numba.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:          # the package must import without the optional dependency
    numba = None

from .d3q19 import CX, CY, CZ, W
from .d3q19_fast import _OPP

if numba is not None:
    _CX = CX.astype(np.int64); _CY = CY.astype(np.int64); _CZ = CZ.astype(np.int64)
    _W = W.astype(np.float64); _OP = np.array(_OPP, dtype=np.int64)

    @njit(parallel=True, cache=True)
    def _table(blocked, compact, count):
        nz, ny, nx = blocked.shape
        table = np.empty((18, count), dtype=np.int32)
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    i = compact[z, y, x]
                    if i >= 0:
                        for q in range(1, 19):
                            table[q - 1, i] = compact[(z - _CZ[q]) % nz, (y - _CY[q]) % ny, (x - _CX[q]) % nx]
        return table

    @njit(inline='always')
    def _collide(fq, post, Fx, Fy, Fz, om_p, hit_p, om_m, hit_m):
        rho = 0.0
        for q in range(19):
            rho += fq[q]
        ux = (fq[1]-fq[2]+fq[7]-fq[8]+fq[9]-fq[10]+fq[11]-fq[12]+fq[13]-fq[14] + 0.5*Fx)/rho
        uy = (fq[3]-fq[4]+fq[7]+fq[8]-fq[9]-fq[10]+fq[15]-fq[16]+fq[17]-fq[18] + 0.5*Fy)/rho
        uz = (fq[5]-fq[6]+fq[11]+fq[12]-fq[13]-fq[14]+fq[15]+fq[16]-fq[17]-fq[18] + 0.5*Fz)/rho
        u2 = ux*ux + uy*uy + uz*uz
        uF = ux*Fx + uy*Fy + uz*Fz
        for q in range(19):
            cu = _CX[q]*ux + _CY[q]*uy + _CZ[q]*uz
            cF = _CX[q]*Fx + _CY[q]*Fy + _CZ[q]*Fz
            fb = fq[_OP[q]]
            even = om_p*(0.5*(fq[q]+fb) - _W[q]*rho*(1.0 + 4.5*cu*cu - 1.5*u2)) - hit_p*_W[q]*(9.0*cu*cF - 3.0*uF)
            odd = om_m*(0.5*(fq[q]-fb) - _W[q]*rho*3.0*cu) - hit_m*_W[q]*3.0*cF
            post[q] = (fq[q] - _W[q]) - even - odd

    @njit(parallel=True, cache=True)
    def _even(f, Fx, Fy, Fz, om_p, hit_p, om_m, hit_m):
        for i in prange(f.shape[1]):
            fq = np.empty(19); post = np.empty(19)
            for q in range(19):
                fq[q] = _W[q] + f[q, i]
            _collide(fq, post, Fx, Fy, Fz, om_p, hit_p, om_m, hit_m)
            for q in range(19):
                f[_OP[q], i] = post[q]

    @njit(parallel=True, cache=True)
    def _odd(f, nbr, Fx, Fy, Fz, om_p, hit_p, om_m, hit_m):
        # Reads touch slot opp(q) of x-c_q and writes touch slot q of x+c_q: the same
        # location for one node and no location shared between nodes, so prange is safe.
        for i in prange(f.shape[1]):
            fq = np.empty(19); post = np.empty(19)
            fq[0] = _W[0] + f[0, i]
            for q in range(1, 19):
                n = nbr[q - 1, i]
                fq[q] = _W[q] + (f[q, i] if n < 0 else f[_OP[q], n])
            _collide(fq, post, Fx, Fy, Fz, om_p, hit_p, om_m, hit_m)
            f[0, i] = post[0]
            for q in range(1, 19):
                n = nbr[_OP[q] - 1, i]
                if n < 0:
                    f[_OP[q], i] = post[q]
                else:
                    f[q, n] = post[q]

    @njit(parallel=True, cache=True)
    def _moments(f, nbr, swapped, Fx, Fy, Fz, rho, ux, uy, uz):
        for i in prange(f.shape[1]):
            fq = np.empty(19)
            fq[0] = _W[0] + f[0, i]
            for q in range(1, 19):
                n = nbr[q - 1, i]
                fq[q] = _W[q] + (f[q, i] if (not swapped or n < 0) else f[_OP[q], n])
            r = 0.0
            for q in range(19):
                r += fq[q]
            rho[i] = r
            ux[i] = (fq[1]-fq[2]+fq[7]-fq[8]+fq[9]-fq[10]+fq[11]-fq[12]+fq[13]-fq[14] + 0.5*Fx)/r
            uy[i] = (fq[3]-fq[4]+fq[7]+fq[8]-fq[9]-fq[10]+fq[15]-fq[16]+fq[17]-fq[18] + 0.5*Fy)/r
            uz[i] = (fq[5]-fq[6]+fq[11]+fq[12]-fq[13]-fq[14]+fq[15]+fq[16]-fq[17]-fq[18] + 0.5*Fz)/r


def default_threads():
    """NUMBA_NUM_THREADS when set, otherwise at most 16: a shared workstation default."""
    if 'NUMBA_NUM_THREADS' in os.environ:
        return int(os.environ['NUMBA_NUM_THREADS'])
    return min(16, os.cpu_count() or 1)


class SparseD3Q19CPU:
    def __init__(self, blocked, force, om_p, om_m, precision, threads=None):
        if numba is None:
            raise RuntimeError("numba-sparse requires numba (pip install 'lbm-permeability[cpu]')")
        self.threads = threads or default_threads()
        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))
        self.shape = blocked.shape
        self.index = np.flatnonzero(~blocked.ravel())
        self.count = int(self.index.size)
        if self.count >= np.iinfo(np.int32).max:
            raise ValueError('more fluid nodes than the int32 neighbour table can address')
        compact = np.full(blocked.size, -1, dtype=np.int32)
        compact[self.index] = np.arange(self.count, dtype=np.int32)
        self.nbr = _table(blocked, compact.reshape(blocked.shape), self.count)
        del compact
        self.f = np.empty((19, self.count), dtype=precision)
        for q in range(19):   # true rest, the fixed point of the staggered invariant (see d3q19_sparse)
            self.f[q] = W[q] * (-1.5 * (int(CX[q]) * force[0] + int(CY[q]) * force[1] + int(CZ[q]) * force[2]))
        self.force = tuple(float(a) for a in force)
        self.rates = (float(om_p), float(1 - .5 * om_p), float(om_m), float(1 - .5 * om_m))
        self.out = [np.empty(self.count) for _ in range(4)]
        self.steps = 0

    def bytes(self):
        return dict(populations=int(self.f.nbytes), neighbour_table=int(self.nbr.nbytes),
                    moments=int(sum(a.nbytes for a in self.out)), storage='deviation f_q-w_q', threads=self.threads)

    def step(self):
        if self.steps % 2:
            _odd(self.f, self.nbr, *self.force, *self.rates)
        else:
            _even(self.f, *self.force, *self.rates)
        self.steps += 1

    def macros(self):
        _moments(self.f, self.nbr, bool(self.steps % 2), *self.force, *self.out)
        rho, ux, uy, uz = self.out
        return (ux, uy, uz), rho

    def dense(self, values):
        field = np.zeros(int(np.prod(self.shape)))
        field[self.index] = values
        return field.reshape(self.shape)
