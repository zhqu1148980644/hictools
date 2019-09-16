import numbers

import numpy as np

from scipy.ndimage.filters import _gaussian_kernel1d
from scipy.ndimage import _ni_support
from hictools.utils._numtools import convolve1d as correlate1d
from hictools.utils._numtools import convolve as correlate

MODE_MAP = {
    'nearest': 0,
    'wrap': 1,
    'reflect': 2,
    'mirror': 3,
    'constant': 4
}


def convolve1d(array: np.ndarray,
               weights: np.ndarray,
               axis: int = -1,
               mode: str = 'constant',
               cval: float = 0.0,
               points: np.ndarray = None,
               nonzero: bool = False) -> np.ndarray:
    if weights.size == 2:
        weights = np.r_[weights, 0]

    isint_in = issubclass(array.dtype.type, numbers.Integral)
    isint_wt = issubclass(weights.dtype.type, numbers.Integral)
    if weights.dtype != array.dtype:
        if isint_in and not isint_wt:
            array = array.astype(weights.dtype)
        elif isint_in and isint_wt:
            if array.itemsize > weights.itemsize:
                weights = weights.astype(array.dtype)
            else:
                array = array.astype(weights.dtype)
        else:
            weights = weights.astype(array.dtype)

    weights = np.array(weights[::-1])
    return correlate1d(array, weights, axis, MODE_MAP[mode], cval, points, nonzero)


def gaussian_filter1d(array: np.ndarray,
                      sigma: float,
                      axis: int = -1,
                      order: int = 0,
                      mode: str = 'constant',
                      cval: float = 0.0,
                      truncate: float = 4.0,
                      radius: int = None,
                      points: np.ndarray = None,
                      nonzero: bool = False) -> np.ndarray:
    std = float(sigma)
    if radius is None:
        radius = int(truncate * std + 0.5)
    weights = _gaussian_kernel1d(sigma, order, radius)
    return convolve1d(array, weights, axis, mode, cval, points, nonzero)


def gaussian_filter(array: np.ndarray,
                    sigma: float,
                    order: int = 0,
                    mode='constant',
                    cval: float = 0.0,
                    truncate: float = 4.0,
                    radius: int = None,
                    points: np.ndarray = None,
                    nonzero: bool = False):
    orders = _ni_support._normalize_sequence(order, array.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, array.ndim)
    modes = _ni_support._normalize_sequence(mode, array.ndim)
    axes = list(range(array.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            array = gaussian_filter1d(array, sigma, axis, order, mode, cval,
                                      truncate, radius, points, nonzero)

    return array


def convolve():
    raise NotImplementedError
