import itertools
import numbers
from functools import partial
from typing import Union, Iterable, Callable, Generator, Sequence, Iterator

import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as sp_linalg

from .utils import suppress_warning


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


def mask_array(mask, *args) -> Iterator[np.ndarray]:
    """Mask all ndarray in args with a given Boolean array.

    :param mask: np.ndarray. Boolean array where desired values are marked with True.
    :param args: tuple. tuple of np.ndarray. Masking will be applied to each ndarray.
    :return: np.ndarray. A generator yield a masked ndarray each time.
    """
    for mat in args:
        if isinstance(mat, (tuple, list)):
            yield tuple(mask_array(mask, *mat))
        else:
            if len(mat.shape) == 1:
                yield mat[mask]
            else:
                yield mat[:, mask]


def index_array(index, *args) -> Iterator[np.ndarray]:
    """Index all ndarray in args with a given Integer array. Be cautious of the order of each value in indexed ndarray.

    :param index: np.ndarray. Integer array with indexs of desired values'.
    :param args: tuple. tuple of np.ndarray. Indexing will be applied to each ndarray.
    :return: np.ndarray. A generator yield indexed ndarray each time.
    """
    yield from mask_array(index, *args)


def get_diag(mat: np.ndarray, offset: int = 0) -> np.ndarray:
    """Get view of a given diagonal of the 2d ndarray.\n
    Reference: https://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy
    """
    length = mat.shape[1]
    st = max(offset, -length * offset)
    ed = max(0, length - offset) * length
    return mat.ravel()[st: ed: length + 1]


def fill_diags(mat: np.ndarray,
               diags: Union[int, float, Iterable] = 1,
               fill_values: Union[int, float, Iterable] = 1.,
               copy: bool = False) -> np.ndarray:
    if isinstance(diags, int):
        diags = range(-diags + 1, diags)

    if isinstance(fill_values, numbers.Number):
        fill_values = itertools.repeat(fill_values)

    if copy:
        mat = mat.copy()

    for diag_index, fill_value in zip(diags, fill_values):
        get_diag(mat, diag_index)[:] = fill_value

    return mat


def apply_along_diags(func: Callable,
                      mat: np.ndarray,
                      offsets: Iterable,
                      filter_fn: Callable = None) -> Generator:
    """Apply a function to a cetain set of diagonals.
    :param func: Callable. Function applied to each diagonal.
    :param mat: np.ndarray. 2d ndarray.
    :param offsets: list. List of diagonal offsets.
    :param filter_fn: Callable. Function applied to each daigonal, should return a mask.
    :return: Generator. Yielding the result of applying func to each diagonal.
    """

    max_len = mat.shape[0]
    offsets = tuple(offsets)

    if filter_fn is None:
        for offset in offsets:
            if offset >= max_len:
                break
            diag = mat.diagonal(offset)
            yield func(diag)
    else:
        diag = mat.diagonal(offsets[len(offsets) - 1])
        res = func(diag)
        if isinstance(res, np.ndarray):
            for offset in offsets:
                if offset >= max_len:
                    break
                diag = mat.diagonal(offset)
                mask = filter_fn(diag)
                zeros = np.zeros_like(diag)
                zeros[mask] = func(diag[mask])
                yield zeros, mask
        else:
            for offset in offsets:
                if offset >= max_len:
                    break
                diag = mat.diagonal(offset)
                yield func(diag[filter_fn(diag)])


def get_decay(mat: Union[np.ndarray],
              bin_span: Sequence = None,
              max_diag: int = None,
              func: Callable = np.mean,
              filter_fn: Callable = partial(np.not_equal, 0.),
              agg_fn: Callable = np.mean) -> Generator:
    """Calculate mean contact across each diagonal.

    :param mat:
    :param bin_span:
    :param max_diag:
    :param func:
    :param filter_fn:
    :param agg_fn:
    :return:
    """
    if sparse.isspmatrix(mat):
        raise NotImplementedError('Not implemented')
    length = mat.shape[0]
    if bin_span is None:
        bin_span = list(range(length + 1))
    if max_diag is None:
        max_diag = length
    offsets = range(bin_span[0], bin_span[-1])
    mean_gen = apply_along_diags(func=func, mat=mat,
                                 offsets=offsets,
                                 filter_fn=filter_fn)

    for st, ed in zip(bin_span[:-1], bin_span[1:]):
        if st >= max_diag:
            break
        res_li = [next(mean_gen) for offset in range(st, ed)]
        res = agg_fn(res_li)
        res = 0 if np.isnan(res) else res
        for _ in range(st, ed):
            yield res


def cumsum2d(ma: np.ndarray):
    n = len(ma)
    summa = np.zeros([n + 1, n + 1])
    for i in range(n):
        for j in range(n):
            summa[i + 1][j + 1] = summa[i][j + 1] + summa[i + 1][j] - summa[i][j] + ma[i][j]
    return summa


def eig(mat, vecnum=3):
    """

    :param mat:
    :param vecnum:
    :return:
    """
    if is_symmetric(mat):
        eigvals, eigvecs = sp_linalg.eigsh(mat, vecnum)
    else:
        eigvals, eigvecs = sp_linalg.eigs(mat, vecnum)

    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs.T[order]

    return eigvals, eigvecs


def pca(mat, vecnum=3):
    """

    :param mat:
    :param vecnum:
    :return:
    """
    center = mat - np.mean(mat, axis=0)
    cov = np.dot(center.T, center)
    eigvals, eigvecs = sp_linalg.eigsh(cov, vecnum)
    eigvals /= (mat.shape[0] - 1)

    return eigvals[::-1], eigvecs[:, ::-1].T


class SliceMixin(object):

    @staticmethod
    def _fill_slice(slice_, length):
        if isinstance(slice_, int):
            start, stop, step = slice_, slice_ + 1, 1
        else:
            start, stop, step = slice_.start, slice_.stop, slice_.step
            if start is None:
                start = 0
            if stop is None:
                stop = length
            if step is None:
                step = 1
        start = start if (start >= 0) else 0
        stop = stop if (stop <= length) else length

        return slice(start, stop, step)

    @staticmethod
    def _is_slices(slices):
        try:
            return all(isinstance(slice_, slice) for slice_ in slices)
        except TypeError as e:
            return isinstance(slices, slice)

    def _check_slices(self, slices, lengths, check_forward=False):
        if isinstance(slices, slice):
            slices = (slices,) * len(lengths)

        for slice_, length in zip(slices, lengths):
            filled_slice = self._fill_slice(slice_, length)
            if (check_forward
                    and (filled_slice.stop < filled_slice.start)):
                raise ValueError("Slice's stop is smaller than start")

            yield filled_slice


class Toeplitz(SliceMixin):
    __slots__ = ('_col', '_row')

    def __init__(self, col, row=None):
        self._col = col
        self._row = col if row is None else row

    def __getitem__(self, items):
        row_slice, col_slice = tuple(self._check_slices(
            slices=items,
            lengths=(self._col.size, self._row.size),
            check_forward=True)
        )
        n_diags = col_slice.start - row_slice.start
        height = row_slice.stop - row_slice.start
        width = col_slice.stop - col_slice.start

        if height == 1 and width == 1:
            array = self._row if n_diags >= 0 else self._col
            return array[n_diags]

        else:
            if n_diags >= 0:
                harray = self._row[n_diags: n_diags + width]
                varray = self._row[:n_diags + 1][::-1][:-1]
                if n_diags < height:
                    varray = np.r_[varray, self._row[:height - n_diags]]
                else:
                    varray = varray[:height]
            else:
                n_diags *= -1
                varray = self._col[n_diags: n_diags + height]
                harray = self._col[:n_diags + 1][::-1][:-1]
                if n_diags < width:
                    harray = np.r_[harray, self._col[:width - n_diags]]
                else:
                    harray = harray[:width]

            ma = linalg.toeplitz(varray[::row_slice.step], harray[::col_slice.step])

            return ma.ravel() if ma.size == 1 else ma


class Expected(Toeplitz):
    __slots__ = ('_col', '_row')

    def __init__(self, decay):
        super().__init__(decay)
