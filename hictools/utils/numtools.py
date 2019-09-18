import itertools
import numbers
from typing import Union, Iterable, Callable, Generator

import numpy as np
from hictools.utils._numtools import _apply_along_diags
from hictools.utils._numtools import convolve1d as correlate1d
from scipy import sparse
from scipy.ndimage import _ni_support
from scipy.ndimage.filters import _gaussian_kernel1d

from hictools.utils.utils import suppress_warning

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


def get_diag(mat: np.ndarray, offset: int = 0) -> np.ndarray:
    """Get view of a given diagonal of the 2d ndarray.\n
    Reference: https://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy
    """
    length = mat.shape[1]
    st = max(offset, -length * offset)
    ed = max(0, length - offset) * length
    return mat.ravel()[st: ed: length + 1]


def apply_along_diags(func: Callable,
                      mat: np.ndarray,
                      offsets: Iterable,
                      filter_fn: Callable) -> Generator:
    """Apply a fucntion to a cetain set of diagonals.
    :param func: Callable. Function applied to each diagonal.
    :param mat: np.ndarray. 2d ndarray.
    :param offsets: list. List of diagonal offsets.
    :param filter_fn: Callable. Function applied to each daigonal, should return a mask.
    :return: Generator. Yielding the result of applying func to each diagonal.
    """

    return _apply_along_diags(func, mat, tuple(offsets), filter_fn)


def fill_diags(mat: np.ndarray,
               diags: Union[int, Iterable] = 1,
               fill_values: Union[float, Iterable] = 1.,
               copy: bool = False) -> np.ndarray:
    if isinstance(diags, int):
        diags = range(-diags + 1, diags)

    if isinstance(fill_values, numbers.Number):
        fill_values = itertools.repeat(fill_values)

    if copy:
        mat = mat.copy()

    for diag_index, fill_value in zip(diags, fill_values):
        diag = get_diag(mat, diag_index)
        diag = fill_value

    return mat


@suppress_warning
def is_symmetric(mat: Union[np.ndarray, sparse.spmatrix],
                 rtol: float = 1e-05,
                 atol: float = 1e-08) -> bool:
    """Check if the input matrix is symmetric.

    :param mat: np.ndarray/scipy.sparse.spmatrix.
    :param rtol: float. The relative tolerance parameter. see np.allclose.
    :param atol: float. The absolute tolerance parameter. see np.allclose
    :return: bool. True if the input matrix is symmetric.
    """
    if isinstance(mat, np.ndarray):
        data, data_t = mat, mat.T
        return np.allclose(data, data_t, rtol=rtol, atol=atol, equal_nan=True)
    elif sparse.isspmatrix(mat):
        mat = mat.copy()
        mat.data[np.isnan(mat.data)] = 0
        return (np.abs(mat - mat.T) > rtol).nnz == 0
    else:
        raise ValueError('Only support for np.ndarray and scipy.sparse_matrix')
