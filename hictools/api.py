"""
Interface.
"""
import functools
import inspect
from typing import Union
from functools import partial
from contextlib import redirect_stderr

import ray
import cooler
import numpy as np
from scipy import sparse
import pandas as pd

from compartment import corr_sorter, linear_bins, get_decay, get_pca_compartment, get_eigen_compartment, Expected
from peaks import hiccups, get_chunk_slices, fetch_regions
from tad import split_diarray, call_domain, insulation_score, di_score, train_hmm, hidden_path
from utils import remove_small_gap, is_symmetric, suppress_warning, fill_diag


def infer_mat(mat,
              mask: np.ndarray = None,
              mask_ratio: float = 0.2,
              span_fn: callable = linear_bins,
              check_symmetric: bool = False,
              copy: bool = False) -> tuple:
    """Maintain non-zero contacts outside bad regions in a triangular sparse matrix.\n
    When calculating decay, always keep contacts outside bad regions to non-nan, and
    keep contacs within bad regions to nan.\n
    This step could take considerable time as dense matrix enable the fast computaion of decay whereas sparse matrix
    can reduce space occupancy and speed up the calculation of OE matrix.\n

    :param mat: np.ndarray/scipy.sparse.sparse_matrix.
    :param mask: np.ndarray.
    :param mask_ratio: float.
    :param span_fn: callable.
    :param check_symmetric: bool.
    :param copy: bool.
    :return: tuple(scipy.sparse.coo_matrix, np.ndarray, np.ndarray).
    """

    def find_mask(nan_mat: np.ndarray):
        last = None
        last_row = -1
        while 1:
            row = np.random.randint(mat.shape[0])
            if row != last_row and not np.alltrue(nan_mat[row]):
                if last is None:
                    last = nan_mat[row]
                elif np.all(last == nan_mat[row]):
                    return ~last
                else:
                    return None
            last_row = row

    def mask_by_ratio(mat: np.ndarray) -> np.ndarray:
        col_mean = np.nanmean(mat, axis=0)
        return col_mean > (np.mean(col_mean) * mask_ratio)

    if check_symmetric and not is_symmetric(mat):
        raise ValueError('Matrix is not symmetric.')

    if copy:
        mat = mat.copy()

    if not isinstance(mat, np.ndarray) and not isinstance(mat, sparse.coo_matrix):
        mat = mat.tocoo(copy=False)

    if mask is None:
        if not isinstance(mat, np.ndarray):
            mat_cache = mat
            mat = mat.toarray()

        nan_mat = np.isnan(mat)
        contain_nan = nan_mat.any()
        if contain_nan:
            mask = find_mask(nan_mat)
            if mask is None:
                mask = mask_by_ratio(mat)
        else:
            mask = mask_by_ratio(mat)
        nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
        if contain_nan and nan_mat[~nan_mask].any():
            mat[nan_mat] = 0
        mat[nan_mask] = np.nan
        decay = get_decay(mat, span_fn)

        if not isinstance(mat, np.ndarray):
            mat = sparse.triu(mat_cache)
            mat.eliminate_zeros()
            mat.data[np.isnan(nan_mask[mat.nonzero()])] = 0
            mat.data[np.isnan(mat.data)] = 0
            mat.eliminate_zeros()
        else:
            mat[nan_mask] = 0
            mat = sparse.triu(mat, format='coo'), mask, decay

    else:
        if not isinstance(mat, np.ndarray):
            nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
            mat.data[np.isnan(mat.data)] = 0

            dense_mat = mat.toarray()
            dense_mat[nan_mask] = np.nan
            decay = get_decay(dense_mat, span_fn)

            mat = sparse.triu(mat)
            mat.eliminate_zeros()
            mat.data[np.isnan(nan_mask[mat.nonzero()])] = 0
            mat.eliminate_zeros()
        else:
            nan_mat = np.isnan(mat)
            contain_nan = nan_mat.any()
            nan_mask = ~(mask[:, np.newaxis] * mask[np.newaxis, :])
            if contain_nan & nan_mat[~nan_mask].any():
                mat[nan_mat] = 0
            mat[nan_mask] = np.nan
            decay = get_decay(mat, span_fn)
            mat[nan_mask] = 0
            mat = sparse.triu(mat, format='coo')

    return mat, mask, decay


def handle_mask(func):
    """

    :param func:
    :return:
    """
    sig = inspect.signature(func)
    if 'full' in sig.parameters.keys():
        raise ValueError("full arguiment already defined.")

    @functools.wraps(func)
    def inner(self, *args, full=True, **kwargs):
        result = func(self, *args, **kwargs)
        if not isinstance(result, np.ndarray):
            return result
        if (len(result.shape) == 2) and not np.equal(*result.shape):
            nan_mat = np.full([result.shape[0], self.shape[0]],
                              np.nan,
                              result.dtype)
            nan_mat[self.mask[None, :]] = result
            return result

        if len(result.shape) == 1:
            mask = self.mask
            shape = self.shape[0]
            _full = result.size == self.shape[0]
        else:
            mask = self.mask_index
            shape = self.shape
            _full = result.shape == self.shape

        if not _full and full:
            nan_mat = np.full(shape, np.nan, dtype=result.dtype)
            nan_mat[mask] = result
            return nan_mat
        elif _full and not full:
            return result[mask]
        else:
            return result

    params = list(sig.parameters.values())
    params.append(inspect.Parameter('full',
                                    inspect.Parameter.KEYWORD_ONLY,
                                    default=True))
    inner.__signature__ = sig.replace(parameters=params)

    return inner


@ray.remote
class ChromMatrix(object):
    """

    """
    HMM_MODELS = {}

    def __init__(self, cool: cooler.Cooler, chrom: str, span_fn: callable = linear_bins):
        """

        :param cool:
        :param chrom:
        :param span_fn:
        """
        self.chrom = chrom
        self._observed = self._get_sparse(cool, chrom)
        bins = cool.bins().fetch(chrom)
        self._weights = np.array(bins['weight'])
        self._start = bins.index.min()
        self._mask = ~np.isnan(self._weights)
        self._decay = {}
        self._cool = cool
        self._binsize = cool.binsize
        self._span_fn = span_fn
        self.shape = self._observed.shape

    @staticmethod
    def _get_sparse(cool: cooler.Cooler, chrom: str) -> np.ndarray:
        """

        :param cool:
        :param chrom:
        :return:
        """
        mat = sparse.triu(cool.matrix(sparse=True, balance=True).fetch(chrom), format='coo')
        mat.data[np.isnan(mat.data)] = 0
        mat.eliminate_zeros()

        return mat

    @property
    def mask(self) -> np.ndarray:
        """

        :return:
        """
        return self._mask

    @property
    def mask2d(self) -> np.ndarray:
        """

        :return:
        """
        return self._mask[:, np.newaxis] * self.mask[np.newaxis, :]

    @property
    def mask_index(self) -> tuple:
        return np.ix_(self.mask, self.mask)

    def observed(self,
                 balance: bool = True,
                 sparse=False, copy=False) -> Union[np.ndarray, sparse.spmatrix]:
        """

        :param balance:
        :param sparse:
        :param copy:
        :return:
        """
        observed = self._observed
        if not balance:
            observed = observed.copy()
            observed.data = observed.data \
                            / self._weights[observed.row] \
                            / self._weights[observed.col]
        if sparse:
            observed = observed.copy() if (copy and balance) else observed
        else:
            tmp_observed = observed.copy()
            x, y = tmp_observed.nonzero()
            tmp_observed.data[np.where(y == x)] = 0
            observed = (observed + tmp_observed.T).toarray()

        return observed

    def decay(self, balance: bool = True, ndiags: int = None) -> np.ndarray:
        """

        :param balance:
        :param ndiags:
        :return:
        """
        if self._decay.get(balance, None) is None:
            self._decay[balance] = get_decay(self.observed(balance=balance,
                                                           sparse=True,
                                                           copy=False)
                                             .toarray(),
                                             span_fn=self._span_fn,
                                             ndiags=ndiags)
        return self._decay[balance]

    def expected(self, balance: bool = True) -> Expected:
        """Caluculate expected count of each interaction across a certain distance.

        :param balance:
        :return:
        """
        return Expected(self.decay(balance=balance))

    @suppress_warning
    def oe(self,
           balance: bool = True,
           ignore_diags: int = 1,
           diag_value: int = 1,
           sparse: bool = False,
           full: bool = True) -> np.ndarray:
        """

        :param full:
        :param balance:
        :param ignore_diags:
        :param diag_value:
        :param sparse:
        :return:
        """
        oe = self.observed(balance, sparse=True, copy=True)
        x, y = oe.nonzero()
        oe.data /= self.decay()[y - x]
        if not sparse:
            oe += oe.T
            oe = oe.toarray()
            for diag_index in range(-ignore_diags + 1, ignore_diags):
                fill_diag(oe, diag_index, diag_value)

        if not full and not sparse:
            return oe[self.mask_index]
        else:
            return oe

    @suppress_warning
    def corr(self,
             balance: bool = True,
             ignore_diags: int = 1,
             diag_value: int = 1,
             full: bool = True) -> np.ndarray:
        """

        :param full:
        :param balance:
        :param ignore_diags:
        :param diag_value:
        :return:
        """

        oe = self.oe(balance, ignore_diags, diag_value, full=False)
        oe[np.isnan(oe)] = 0
        corr = np.corrcoef(np.nan_to_num(oe))
        for diag_index in range(-ignore_diags + 1, ignore_diags):
            fill_diag(corr, diag_index, diag_value)

        if not full:
            return corr
        else:
            nan_mat = np.full(self.shape, np.nan, dtype=corr.dtype)
            nan_mat[self.mask_index] = corr
            return nan_mat

    def insu_score(self,
                   balance: bool = True,
                   window_size: int = 20,
                   ignore_diags: int = 1,
                   normalize: bool = True,
                   full: bool = True) -> np.ndarray:
        """

        :param full:
        :param balance:
        :param window_size:
        :param ignore_diags:
        :param normalize:
        :return:
        """

        insu_score = insulation_score(self.observed(balance,
                                                    sparse=True,
                                                    copy=False)
                                      .tocsr(copy=False),
                                      window_size=window_size,
                                      ignore_diags=ignore_diags,
                                      normalize=normalize)

        if not full:
            return insu_score[self.mask]
        else:
            return insu_score

    def di_score(self,
                 balance: bool = True,
                 windowsize: int = 20,
                 ignore_diags: int = 1,
                 fetch_window: bool = False,
                 method: str = 'standard',
                 full: bool = True) -> np.ndarray:
        """

        :param full:
        :param fetch_window:
        :param balance:
        :param windowsize:
        :param ignore_diags:
        :param method:
        :return:
        """

        score = di_score(self.observed(balance,
                                       sparse=True,
                                       copy=False)
                         .tocsr(copy=False),
                         window_size=windowsize,
                         ignore_diags=ignore_diags,
                         fetch_window=fetch_window,
                         method=method)
        if not full:
            return score[self.mask]
        else:
            return score

    def fetch_hmm_model(self,
                        num_mix: int = 3,
                        window_size: int = 10,
                        ignore_diags: int = 1,
                        method: str = 'standard'):
        """

        :param num_mix:
        :param window_size:
        :param ignore_diags:
        :param method:
        :return:
        """
        key = (str(self._cool), window_size, method)
        if self.HMM_MODELS.get(key, None) is None:
            self.HMM_MODELS[key] = train_hmm(self._cool,
                                             num_mix,
                                             partial(di_score,
                                                     window_size=window_size,
                                                     ignore_diags=ignore_diags,
                                                     method=method))

        return self.HMM_MODELS[key]

    def peaks(self,
              max_dis: int = 5000000,
              p: int = 2,
              w: int = 5,
              fdrs: tuple = (0.01, 0.01, 0.01, 0.01),
              sigs: tuple = (0.01, 0.01, 0.01, 0.01),
              single_fcs: tuple = (1.75, 1.5, 1.5, 1.75),
              double_fcs: tuple = (2.5, 0, 0, 2.5),
              ignore_single_gap: bool = True,
              chunk_size: int = 500,
              bin_index: bool = True,
              num_cpus: int = 1) -> pd.DataFrame:
        """

        :param max_dis:
        :param p:
        :param w:
        :param fdrs:
        :param sigs:
        :param single_fcs:
        :param double_fcs:
        :param ignore_single_gap:
        :param chunk_size:
        :param bin_index:
        :param num_cpus:
        :return:
        """

        def expected_fetcher(key, slices, expected=self.expected()):
            return expected[slices]

        def observed_fetcher(key, slices, cool=self._cool):
            row_st, row_ed = slices[0].start + self._start, slices[0].stop + self._start
            col_st, col_ed = slices[1].start + self._start, slices[1].stop + self._start
            return cool.matrix()[slice(row_st, row_ed), slice(col_st, col_ed)]

        def factors_fetcher(key, slices, factors=self._weights):
            return factors[slices[0]], factors[slices[1]]

        band_width = max_dis // self._binsize
        if chunk_size is None:
            chunk_size = band_width
        chunks = get_chunk_slices(length=self.shape[0],
                                  band_width=band_width,
                                  height=chunk_size,
                                  ov_length=2 * w)
        chunks = ((self.chrom, chunk) for chunk in chunks)
        kernels = fetch_regions(p, w, kernel=True)
        with open('ray.log', 'w') as f:
            with redirect_stderr(f):
                peaks = hiccups(expected_fetcher=expected_fetcher,
                                observed_fetcher=observed_fetcher,
                                factors_fetcher=factors_fetcher,
                                chunks=chunks,
                                kernels=kernels,
                                num_cpus=num_cpus,
                                max_dis=max_dis,
                                outer_radius=w,
                                resolution=self._binsize,
                                fdrs=fdrs,
                                sigs=sigs,
                                single_fcs=single_fcs,
                                double_fcs=double_fcs,
                                ignore_single_gap=ignore_single_gap,
                                bin_index=bin_index)
        return peaks

    def tads(self,
             hmm_model=None,
             window_size=10,
             ignore_diags: int = 1,
             method='standard',
             num_mix: int = 3,
             calldomain_fn=call_domain) -> pd.DataFrame:
        """

        :param hmm_model:
        :param window_size:
        :param ignore_diags:
        :param method:
        :param num_mix:
        :param calldomain_fn:
        :return:
        """
        if hmm_model is None:
            hmm_model = self.fetch_hmm_model(num_mix=num_mix,
                                             window_size=window_size,
                                             ignore_diags=ignore_diags,
                                             method=method)

        tads = []
        for start, diarray in split_diarray(self.di_score(), remove_small_gap(~self.mask)):
            domain = calldomain_fn(hidden_path(diarray, hmm_model).path)
            tads.extend((st + start, ed + start) for st, ed in domain)

        return pd.DataFrame(tads)

    def compartments(self,
                     method: str = 'pca',
                     vec_range: Union[slice, int] = slice(0, 3),
                     com_range: Union[slice, int] = slice(0, 1),
                     sort_fn: callable = corr_sorter,
                     full: bool = True) -> np.ndarray:
        """

        :param full:
        :param method:
        :param vec_range:
        :param com_range:
        :param sort_fn:
        :return:
        """
        vecnum = vec_range.stop if isinstance(vec_range, slice) else vec_range
        if method == 'pca':
            vecs = get_pca_compartment(mat=self.corr(full=False),
                                       vecnum=vecnum)
        elif method == 'eigen':
            vecs = get_eigen_compartment(mat=self.oe(full=False) - 1,
                                         vecnum=vecnum)
        else:
            raise ValueError("Only support for 'eigrn' or 'pca' ")
        vecs = vecs[vec_range]
        if sort_fn is not None:
            vecs = sort_fn(self, vecs)
        vecs = vecs[com_range]
        if vecs.shape[0] == 1:
            vecs = vecs.ravel()

        if not full:
            return vecs
        else:
            if len(vecs.shape) == 2:
                nan_mat = np.full((vecs.shape[0], self.shape[1]), np.nan, dtype=vecs.dtype)
                nan_mat[: self.mask] = vecs
            else:
                nan_mat = np.full(self.shape[1], np.nan, dtype=vecs.dtype)
                nan_mat[self.mask] = vecs
            return nan_mat


if __name__ == "__main__":
    pass
