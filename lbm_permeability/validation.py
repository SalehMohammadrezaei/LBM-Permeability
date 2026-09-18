"""Shared input validation. Core masks are Boolean, True means solid."""
import numbers
import numpy as np


def mask(value, ndim=None):
    a = np.asarray(value)
    if a.dtype != np.bool_:
        raise ValueError('blocked must be a Boolean mask (True=solid); use decode_mask for images')
    if a.ndim not in (2, 3) or (ndim is not None and a.ndim != ndim) or not a.size:
        raise ValueError('blocked must be a nonempty 2D/3D mask of the requested dimension')
    if any(n > np.iinfo(np.int32).max // 2 for n in a.shape):
        raise ValueError('axis exceeds CUDA int32 coordinate range')
    return np.ascontiguousarray(a)


def integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def positive(name, value, minimum=0):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= minimum:
        raise ValueError(f'{name} must be finite and > {minimum}')
    return float(value)


def spacing(value):
    return positive('voxel_size (one isotropic scalar)', value)


def parameters(tau, force, n_steps_max, conv_tol, conv_window, precision, wall_timeout_s,
               heartbeat=1, mempool_flush=1, **controls):
    positive('tau', tau, .5)
    if not np.isfinite(force).all():
        raise ValueError('force density must be finite')
    for name,value in dict(n_steps_max=n_steps_max, conv_window=conv_window, heartbeat=heartbeat,
                           mempool_flush=mempool_flush, **controls).items():
        integer(name, value)
    positive('conv_tol', conv_tol)
    if precision not in ('float64', 'float32'):
        raise ValueError('precision must be float64 or float32')
    if wall_timeout_s is not None:
        positive('wall_timeout_s (None disables timeout)', wall_timeout_s)


def decode_mask(image, *, solid_value, pore_value):
    """Decode a declared two-label image without thresholding or guessing inversion."""
    a = np.asarray(image)
    if not np.isfinite([solid_value, pore_value]).all() or solid_value == pore_value:
        raise ValueError('two distinct finite labels are required')
    if a.ndim not in (2,3) or not a.size or not np.isfinite(a).all():
        raise ValueError('image must be a nonempty finite 2D/3D array')
    if not np.all((a == solid_value) | (a == pore_value)):
        raise ValueError('image contains values outside the declared binary encoding')
    return mask(a == solid_value)
