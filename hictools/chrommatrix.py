"""Interface."""
from functools import partial, lru_cache, singledispatch, cached_property
from typing import Union, Tuple, Any, TypeVar

import numpy as np
import cooler
from scipy import sparse

Array = Union[np.ndarray, sparse.spmatrix]

from .utils.numtools import get_diag, fill_diags, Expected
from .utils.utils import suppress_warning, MethodWrapper


@singledispatch
def extract_matrix(data: Any, chrom, binsize, start) -> Tuple:
    raise ValueError(f"Invalid data source {data} {chrom}")


@extract_matrix.register
def _(co: cooler.Cooler, chrom, *arg, **kwargs) -> Tuple:
    """Fetch a triangular sparse matrix with no value in gap regions(The original data
    in hdf5)
    """
    from cooler.core import CSRReader
    from cooler.util import open_hdf5
    with open_hdf5(co.store) as h5:
        root = h5[co.root]
        chrom_offset = root['indexes']['chrom_offset']
        cid = co._chromids[chrom]
        row_st, row_ed = chrom_offset[cid], chrom_offset[cid + 1]
        col_st, col_ed = row_st, row_ed
        reader = CSRReader(
            h5=root,
            field='count',
            max_chunk=500000000000000000,
        )
        x, y, values = reader.query(row_st, row_ed, col_st, col_ed)
        mat = sparse.coo_matrix(
            (values, (x - row_st, y - col_st)),
            shape=(row_ed - row_st, col_ed - col_st)
        )
        weights = root["bins"]["weight"][row_st: row_ed]
        is_nan = np.isnan(weights)
        weights[is_nan] = 0
        mat.data = mat.data * weights[mat.row] * weights[mat.col]
        mat.eliminate_zeros()
        weights[is_nan] = np.nan

    return mat.astype(np.float32), weights.astype(np.float32), co, chrom, co.binsize, row_st


@extract_matrix.register
def _(ma: None, *args) -> Tuple:
    raise NotImplementedError("Only support for cooler input")


class ChromMatrix(object):
    # TODO support for initializing from file or 2d-matrix.
    MEAN_NONZERO = False
    STD_NONZERO = True

    def __init__(self, data: Union[str, cooler.Cooler, Array],
                 chrom: str = None, binsize: int = 1, start: int = 0):
        """

        :param data: Union[cooler.Cooler, str]. Path of .cool file or cooler.Cooler object.
        :param chrom: str: Chromosome name.
        """
        if isinstance(data, str) and "cool" in data:
            data = cooler.Cooler(data)
        self._observed, self._weights, self.cool, self.chrom, self.binsize, self.start \
            = extract_matrix(data, chrom, binsize, start)
        self._mask = ~np.isnan(self._weights)
        self._dis = self._observed.col - self._observed.row
        self.shape = self._observed.shape
        self.config = {
            "balance": True,
            "bin_span": None,
            "ignore_ndiags": 1,
            "oe_type": "zscore",
        }

    def __call__(self, **kwargs) -> 'ChromMatrix':
        self.config.update(kwargs)
        return self

    def handle_mask(self, mat: Array, full: bool):
        """Automatically handle mask."""
        full_length = self.shape[0]
        is_matrix, shape = (len(mat.shape) == 2), mat.shape
        intact = shape[0] == full_length
        if len(shape) == 2:
            intact = intact or (shape[1] == full_length)

        if intact != full and isinstance(mat, sparse.spmatrix):
            mat = mat.toarray()

        if intact == full:
            return mat
        # shrink
        elif intact and not full:
            mask = self.mask if len(shape) != 2 else self.mask_index
            # in case for compartment like multivec
            if shape[0] != full_length:
                mask = (np.arange(shape[0])[:, None], mask[1])
            return mat[mask]
        # expand
        else:
            predict_length = self.mask.sum()
            fit = shape[0] == predict_length
            if is_matrix:
                fit = fit or (shape[1] == predict_length)
            if not fit:
                raise ValueError(f"Array of shape {shape} can't be unmasked."
                                 f"Original shape is {(full_length, full_length)}")
            nan_mat_fn = partial(np.full, fill_value=np.nan, dtype=mat.dtype)

            if is_matrix and (shape[0] != shape[1]):
                mat = mat.T if (max(shape) == shape[0]) else mat
                nan_mat = nan_mat_fn(shape=(min(shape), full_length))
                nan_mat[:, self.mask] = mat

            elif is_matrix and (shape[0] == shape[1]):
                nan_mat = nan_mat_fn(shape=(full_length, full_length))
                nan_mat[self.mask_index] = mat

            else:
                nan_mat = nan_mat_fn(shape=full_length)
                nan_mat[self.mask] = mat

            return nan_mat

    @property
    def weights(self):
        return self._weights

    @property
    def mask(self) -> np.ndarray:
        """1-d ndarray with gap regions set to False.
        """
        return self._mask

    @cached_property
    def mask2d(self) -> np.ndarray:
        """2-d ndarray with gap regions set to False.
        """
        return self.mask[:, np.newaxis] * self.mask[np.newaxis, :]

    @cached_property
    def mask_index(self) -> Tuple[np.ndarray]:
        """Tuple of ndarray representing valid regions' row and column index.
        """
        return np.ix_(self.mask, self.mask)

    def _pad1d(self, array: np.ndarray, pad_val: float = 0.) -> np.ndarray:
        return np.pad(array,
                      mode='constant',
                      constant_values=pad_val,
                      pad_width=(0, self.shape[0] - array.size))

    def diag_mask(self, diag: int = 0) -> np.ndarray:
        length = self.shape[0] - diag
        return self._mask[: length] & self._mask[::-1][: length][::-1]

    @cached_property
    def num_valid(self):
        return np.array([self.diag_mask(i).sum()
                         for i in range(self.shape[0])])

    @cached_property
    def num_nonzero(self) -> np.ndarray:
        nonzero_num = np.bincount(self._dis)
        if nonzero_num.size != self.shape[0]:
            nonzero_num = self._pad1d(nonzero_num)

        return nonzero_num.astype(np.int32)

    def ob(self, sparse: bool = False) -> Array:
        """ observed hic contact matrix.

        Parameters
        ----------
        sparse : bool
            If return sprase version of matrix in scipy.sparse.coo_matrix
        Returns
        -------

        """
        observed = self._observed
        balance = self.config.get("balance", True)
        if not balance and self.weights is not None:
            observed = observed.copy()
            observed.data = (observed.data
                             / self.weights[observed.row]
                             / self.weights[observed.col])
        if not sparse:
            observed = (observed + observed.T).toarray()
            get_diag(observed, 0)[:] /= 2

        return observed

    @lru_cache(maxsize=3)
    @suppress_warning
    def mean(self,
             balance: bool = True,
             bin_span: np.ndarray = None) -> np.ndarray:
        ob = self.ob(sparse=True)
        sum_counts = np.bincount(ob.col - ob.row, weights=ob.data)
        if sum_counts.size != self.shape[0]:
            sum_counts = self._pad1d(sum_counts)

        if self.MEAN_NONZERO:
            mean_array = sum_counts / self.num_nonzero
        else:
            mean_array = sum_counts / self.num_valid

        mean_array[np.isnan(mean_array)] = 0

        if bin_span is not None:
            for st, ed in zip(bin_span[:-1], bin_span[1:]):
                mean_array[st: ed] = np.mean(mean_array[st: ed])

        return mean_array.astype(np.float32)

    @lru_cache(maxsize=3)
    @suppress_warning
    def std(self,
            balance: bool = True,
            bin_span: np.ndarray = None) -> np.ndarray:
        ob = self.ob(sparse=True)
        mean_array = self.mean(balance=balance, bin_span=bin_span)
        sum_square = np.bincount(self._dis,
                                 weights=(ob.data - mean_array[self._dis]) ** 2)

        if sum_square.size != self.shape[0]:
            sum_square = self._pad1d(sum_square)

        if self.STD_NONZERO:
            std_array = np.sqrt(sum_square / self.num_nonzero)

        else:
            sum_square += ((mean_array ** 2) *
                           (self.num_valid - self.num_nonzero))
            std_array = np.sqrt(sum_square / self.num_valid)

        std_array[np.isnan(std_array)] = 0
        return std_array.astype(np.float32)

    def decay(self) -> np.ndarray:
        """Calculate expected count of each interaction across a certain distance."""
        return self.mean(balance=self.config.get("balance", True))

    def expected(self) -> Expected:
        """Calculate expected matrix that pixels in a certain diagonal have the same value."""
        return Expected(self.decay())

    @suppress_warning
    def oe(self,
           sparse: bool = False) -> Array:
        """Calculate expected-matrix-corrected matrix to reduce distance bias in hic experiments.

        Parameters
        ----------
        sparse : bool
            bool. If return sparsed version of oe matrix.
        Returns
        -------

        """
        spoe = self.ob(sparse=True).copy()
        decay = self.decay()[self._dis]
        if self.config.get('oetype', 'zscore') == "zscore":
            std = self.std()[self._dis]
            spoe.data -= decay
            spoe.data /= std

        else:
            spoe.data /= decay

        if sparse:
            return spoe
        else:
            dsoe = (spoe + spoe.T).toarray()
            get_diag(dsoe)[:] /= 2
            return dsoe

    @suppress_warning
    def corr(self) -> np.ndarray:
        """Calculate pearson correlation matrix based on oe matrix.
        """
        dsoe = self.oe(sparse=False, full=False)
        dsoe[np.isnan(dsoe)] = 0
        dsoe = fill_diags(
            mat=dsoe,
            diags=self.config.get("ignore_ndiags", 1),
            fill_values=0
        )
        return self.handle_mask(np.corrcoef(dsoe).astype(dsoe.dtype), full=True)


def filter_fn(attr, attr_obj):
    return not attr.startswith("_") and attr != 'handle_mask'


def output_handler(raw_fn, args, kwargs, extra_kwargs, output):
    if not isinstance(output, (np.ndarray, sparse.spmatrix)):
        return output

    import inspect
    chrom_ma = args[0] if inspect.isfunction(raw_fn) else None

    if chrom_ma is not None and 'full' in extra_kwargs:
        output = chrom_ma.handle_mask(output, full=extra_kwargs['full'])

    return output


MethodWrapper.wrap_attr(ChromMatrix, filter_fn, MethodWrapper(output_handler=output_handler))

if __name__ == "__main__":
    pass
