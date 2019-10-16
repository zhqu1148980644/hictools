"""Interface."""
from functools import partial, lru_cache
from typing import Union, Tuple, Sequence

import cooler
import numpy as np
import pandas as pd
from cooler.core import CSRReader
from cooler.util import open_hdf5
from scipy import sparse

from .compartment import (
    corr_sorter,
    get_pca_compartment,
    get_eigen_compartment,
    Expected
)
from .peaks import (
    hiccups,
    get_chunk_slices,
    fetch_kernels
)
from .tad import (
    insulation_score,
    di_score,
    split_diarray,
    call_domain,
    hidden_path
)
from .utils.numtools import (
    get_diag,
    fill_diags
)
from .utils.utils import (
    remove_small_gap,
    suppress_warning,
    LazyProperty,
    multi_methods
)


class ChromMatrix(object):
    # TODO support for initializing from file or 2d-matrix.
    def __init__(self, co: Union[cooler.Cooler, str],
                 chrom: str,
                 bin_span: Sequence = None,
                 mean_nonzero: bool = True,
                 std_nonzero: bool = False):
        """

        :param co: Union[cooler.Cooler, str]. Path of .cool file or cooler.Cooler object.
        :param chrom: str: Chromosome name.
        :param bin_span: Sequence.
        """
        if isinstance(co, str):
            co = cooler.Cooler(co)
        self._observed, self._weights, self._start = self._get_info(co, chrom)
        self._mask = ~np.isnan(self._weights)
        self._dis = self._observed.col - self._observed.row
        self._binsize = co.binsize
        self._bin_span = bin_span
        self.mean_nonzero = mean_nonzero
        self.std_nonzero = std_nonzero
        self.cool = co
        self.chrom = chrom
        self.shape = self._observed.shape

    @staticmethod
    def _get_info(co: cooler.Cooler,
                  chrom: str) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
        """Fetch a triangular sparse matrix with no value in gap regions(The original data
        in hdf5)
        """
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

        return mat.astype(np.float32), weights.astype(np.float32), row_st

    def handle_mask(self, array: np.ndarray, full: bool):
        """Automatically handle mask."""
        full_length = self.shape[0]
        is_matrix, shape = (len(array.shape) == 2), array.shape
        intact = shape[0] == full_length
        if len(shape) == 2:
            intact = intact or (shape[1] == full_length)

        if intact == full:
            return array
        elif intact and not full:
            return array[self.mask if (len(shape)) != 2 else self.mask_index]
        else:
            predict_length = self.mask.sum()
            fit = shape[0] == predict_length
            if is_matrix:
                fit = fit or (shape[1] == predict_length)
            if not fit:
                raise ValueError(f"Array of shape {shape} can't be unmasked."
                                 f"Original shape is {(full_length, full_length)}")
            nan_mat_fn = partial(np.full, fill_value=np.nan, dtype=array.dtype)

            if is_matrix and (shape[0] != shape[1]):
                array = array.T if (max(shape) == shape[0]) else array
                nan_mat = nan_mat_fn(shape=(min(shape), full_length))
                nan_mat[:, self.mask] = array

            elif is_matrix and (shape[0] == shape[1]):
                nan_mat = nan_mat_fn(shape=(full_length, full_length))
                nan_mat[self.mask_index] = array

            else:
                nan_mat = nan_mat_fn(shape=full_length)
                nan_mat[self.mask] = array

            return nan_mat

    @property
    def mask(self) -> np.ndarray:
        """1-d ndarray with gap regions set to False.
        """
        return self._mask

    @LazyProperty
    def mask2d(self) -> np.ndarray:
        """2-d ndarray with gap regions set to False.
        """
        return self.mask[:, np.newaxis] * self.mask[np.newaxis, :]

    @LazyProperty
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

    @LazyProperty
    def num_valid(self):
        return np.array([self.diag_mask(i).sum()
                         for i in range(self.shape[0])])

    @LazyProperty
    def num_nonzero(self) -> np.ndarray:
        nonzero_num = np.bincount(self._dis)
        if nonzero_num.size != self.shape[0]:
            nonzero_num = self._pad1d(nonzero_num)

        return nonzero_num.astype(np.int32)

    def observed(self,
                 balance: bool = True,
                 sparse: bool = True,
                 copy: bool = False) -> Union[np.ndarray, sparse.coo_matrix]:
        """Return observed matrix.

        :param balance: bool. If use factors to normalize the observed contacts matrix.
        :param sparse: bool. If return sprase version of matrix in scipy.sparse.coo_matrix
        format.
        :param copy: bool. If return the copy of original matrix.
        :return: Union[np.ndarray, sparse.spmatrix].
        """
        observed = self._observed
        if not balance:
            observed = observed.copy()
            observed.data = (observed.data
                             / self._weights[observed.row]
                             / self._weights[observed.col])
        if sparse:
            observed = observed.copy() if (copy and balance) else observed
        else:
            observed = (observed + observed.T).toarray()
            get_diag(observed, 0)[:] /= 2
        return observed

    @lru_cache(maxsize=1)
    @suppress_warning
    def mean(self,
             balance: bool = True) -> np.ndarray:
        ob = self.observed(balance=balance, sparse=True, copy=False)
        sum_counts = np.bincount(ob.col - ob.row, weights=ob.data)
        if sum_counts.size != self.shape[0]:
            sum_counts = self._pad1d(sum_counts)

        if self.mean_nonzero:
            mean_array = sum_counts / self.num_nonzero
        else:
            mean_array = sum_counts / self.num_valid

        mean_array[np.isnan(mean_array)] = 0

        if self._bin_span is not None:
            for st, ed in zip(self._bin_span[:-1], self._bin_span[1:]):
                mean_array[st: ed] = np.mean(mean_array[st: ed])

        return mean_array.astype(np.float32)

    @lru_cache(maxsize=1)
    @suppress_warning
    def std(self,
            balance: bool = True) -> np.ndarray:
        ob = self.observed(balance=balance, sparse=True, copy=False)
        mean_array = self.mean(balance=balance)
        sum_square = np.bincount(self._dis,
                                 weights=(ob.data - mean_array[self._dis]) ** 2)

        if sum_square.size != self.shape[0]:
            sum_square = self._pad1d(sum_square)

        if self.std_nonzero:
            std_array = np.sqrt(sum_square / self.num_nonzero)

        else:
            sum_square += ((mean_array ** 2) *
                           (self.num_valid - self.num_nonzero))
            std_array = np.sqrt(sum_square / self.num_valid)

        std_array[np.isnan(std_array)] = 0
        return std_array.astype(np.float32)

    def decay(self,
              balance: bool = True) -> np.ndarray:
        """Calculate expected count of each interaction across a certain distance."""
        return self.mean(balance=balance)

    def expected(self,
                 balance: bool = True) -> Expected:
        """Calculate expected matrix that pixels in a certain diagonal have the same value."""
        return Expected(self.decay(balance=balance))

    @suppress_warning
    def oe(self,
           balance: bool = True,
           zscore: bool = False,
           sparse: bool = False,
           full: bool = True) -> np.ndarray:
        """Calculate expected-matrix-corrected matrix to reduce distance bias in hic experiments.

        :param zscore:
        :param balance: bool. If use factors to normalize the observed contacts matrix
        before calculation.
        :param sparse: bool. If return sparsed version of oe matrix.
        :param full: bool. Return non-gap region of output oe matrix if full set to False.
        :return: np.ndarray. 2-d array representing oe matrix.
        """
        spoe = self.observed(
            balance=balance,
            sparse=True,
            copy=True
        )
        decay = self.decay(balance=balance)[self._dis]
        if zscore:
            std = self.std(balance=balance)[self._dis]
            spoe.data -= decay
            spoe.data /= std

        else:
            spoe.data /= decay

        if sparse:
            return spoe
        else:
            dsoe = (spoe + spoe.T).toarray()
            get_diag(dsoe)[:] /= 2
            return self.handle_mask(dsoe, full)

    @suppress_warning
    def corr(self,
             balance: bool = True,
             zscore: bool = False,
             ignore_diags: int = 1,
             fill_value: float = 1.,
             full: bool = True) -> np.ndarray:
        """Calculate correlation matrix based on matrix..

        :param zscore:
        :param balance: bool. If use factors to normalize the observed contacts matrix
        before calculation.
        :param ignore_diags: int. Number of diagonals to ignore. Values in these ignored
        diagonals will set to 'fill_value'.
        :param fill_value: float. Value to fill ignored diagonals of OE mattrix.
        :param full: bool. Return non-gap region of output corr matrix if full set to False.
        :return: np.ndarray. 2-d array representing corr matrix.
        """

        dsoe = self.oe(balance=balance,
                       zscore=zscore,
                       sparse=False,
                       full=False)
        dsoe[np.isnan(dsoe)] = 0
        dsoe = fill_diags(
            mat=dsoe,
            diags=ignore_diags,
            fill_values=fill_value
        )
        corr = np.corrcoef(dsoe).astype(dsoe.dtype)
        return self.handle_mask(corr, full)

    def insu_score(self,
                   balance: bool = True,
                   window_size: int = 20,
                   ignore_diags: int = 1,
                   normalize: bool = True,
                   full: bool = True) -> np.ndarray:
        """Calculate insulation score.

        :param balance: bool. If use factors to normalize the observed contacts matrix
        before calculation.
        :param window_size: int. Diameter of square in which contacts are summed along
        the diagonal.
        :param ignore_diags: int. Number of diagonals to ignore.
        :param normalize: bool. If normalize the insulation score with log2 ratio of
        insu_score and mean
        insu_score.
        :param full: bool. Return non-gap region of output insulation score if full set
        to False.
        :return: np.ndarray.
        """

        score = insulation_score(
            self.observed(
                balance=balance,
                sparse=True,
                copy=False
            ).tocsr(copy=False),
            window_size=window_size,
            ignore_diags=ignore_diags,
            normalize=normalize
        )

        return self.handle_mask(score, full)

    def di_score(self,
                 balance: bool = True,
                 window_size: int = 20,
                 ignore_diags: int = 1,
                 fetch_window: bool = False,
                 method: str = 'standard',
                 full: bool = True) -> np.ndarray:
        """Calculate directionality index.

        :param balance: bool. If use factors to normalize the observed contacts matrix
        before calculation.
        :param window_size: int. Length of upstream array and downstream array.
        :param ignore_diags: int. Number of diagonals to ignore.
        :param fetch_window: bool. If set to True, return np.hstack([contacts_up. contacts_down])
        :param method: str. Method for computing directionality index. 'standard' and
        'adptive' are supported by now.
        :param full: Return non-gap region of output directionality index if full set to False.
        :return: np.ndarray.
        """

        score = di_score(
            self.observed(
                balance=balance,
                sparse=True,
                copy=False
            ).tocsr(copy=False),
            window_size=window_size,
            ignore_diags=ignore_diags,
            fetch_window=fetch_window,
            method=method
        )

        return self.handle_mask(score, full)

    @multi_methods
    def peaks(self):
        """Currently support {num} methods for calling peaks: {methods}."""

    @peaks
    def _hiccups(self,
                 max_dis: int = 5000000,
                 p: int = 2,
                 w: int = 5,
                 fdrs: tuple = (0.1, 0.1, 0.1, 0.1),
                 sigs: tuple = (0.1, 0.1, 0.1, 0.1),
                 fold_changes: tuple = (1.5, 1.5, 1.5, 1.5),
                 ignore_single_gap: bool = True,
                 chunk_size: int = 500,
                 num_cpus: int = 1,
                 bin_index: bool = True,
                 **kwargs) -> pd.DataFrame:
        """Using hiccups algorithm to detect loops in a band regions close to the main
        diagonal of hic contact matrix.

        :param max_dis: int. Max distance of loops. Due to the natural property of
        loops(distance less than 8M) and computation bound of hiccups algorithm, hiccups
        algorithm are applied only in a band region over the main diagonal to speed up the
        whole process.
        :param p: int. Radius of innner square in hiccups. Pixels in this region will not
        be counted in the calcualtion of background.
        :param w: int. Radius of outer square in hiccps. Only pixels in this region will
        be counted in the calculation of background.
        :param fdrs: tuple. Tuple of fdrs to control the false discovery rate for each
        background.
        :param sigs: tuple. Tuple of padjs thresholds for each background.
        :param fold_changes: tuple. Padjs threshold for each region. Valid peak's padjs
        should pass all four fold-hange thresholds.
        :param ignore_single_gap: bool. If ignore small gaps when filtering peaks close
        to gap regions.
        :param chunk_size: int. Height of each chunk(submatrix).
        :param num_cpus: int. Number of cores to call peaks. Calculation based on chunks
        and the detection of peaks based on different chromosome will run in parallel.
        :return: pd.Dataframe.
        """

        def expected_fetcher(key, slices, expected=self.expected(**kwargs)):
            return expected[slices]

        def observed_fetcher(key, slices, cool=self.cool):
            row_st, row_ed = slices[0].start + \
                self._start, slices[0].stop + self._start
            col_st, col_ed = slices[1].start + \
                self._start, slices[1].stop + self._start
            return cool.matrix()[slice(row_st, row_ed), slice(col_st, col_ed)]

        def factors_fetcher(key, slices, factors=self._weights):
            return factors[slices[0]], factors[slices[1]]

        band_width = max_dis // self._binsize
        if chunk_size is None:
            chunk_size = band_width
        chunks = get_chunk_slices(
            length=self.shape[0],
            band_width=band_width,
            height=chunk_size,
            ov_length=2 * w
        )
        chunks = ((self.chrom, chunk) for chunk in chunks)
        kernels = fetch_kernels(p, w)

        peaks_df = hiccups(
            observed_fetcher=observed_fetcher,
            expected_fetcher=expected_fetcher,
            factors_fetcher=factors_fetcher,
            chunks=chunks,
            kernels=kernels,
            inner_radius=p,
            outer_radius=w,
            num_cpus=num_cpus,
            max_dis=max_dis,
            resolution=self._binsize,
            fdrs=fdrs,
            sigs=sigs,
            fold_changes=fold_changes,
            ignore_single_gap=ignore_single_gap,
            bin_index=bin_index
        )

        return peaks_df

    @peaks
    def _peaks2d(self):
        return NotImplemented

    @multi_methods
    def compartments(self):
        pass

    @compartments
    @lru_cache(maxsize=3)
    @suppress_warning(warning_msg=RuntimeWarning)
    def _decomposition(self,
                       method: str = 'pca',
                       balance: bool = True,
                       zscore: bool = False,
                       ignore_diags: int = 3,
                       fill_value: float = 1.,
                       numvecs: int = 3,
                       sort_fn: callable = corr_sorter,
                       full: bool = True) -> np.ndarray:
        """Calculate A/B compartments based on decomposition of intra-chromosomal 
        interaction matrix. Currently, two methods are supported for detecting A/B 
        compatements. 'pca' uses principle component analysis based on corr matrix 
        and 'eigen' uses eigen value decomposition based on OE-1 matrix.

        :return: np.ndarray. Array representing the A/B seperation of compartment.
        Negative value denotes B compartment.
        """

        if method in ('pca', 'eigen'):
            corr = self.corr(
                balance=balance,
                zscore=zscore,
                full=False,
                ignore_diags=ignore_diags,
                fill_value=fill_value
            )
        else:
            raise ValueError("Only support for 'eigrn' or 'pca' ")

        if method == 'pca':
            vecs = get_pca_compartment(mat=corr, vecnum=numvecs)

        else:
            dsoe = self.oe(balance=balance,
                           zscore=zscore,
                           sparse=False,
                           full=False)
            dsoe = fill_diags(dsoe, diags=ignore_diags, fill_values=fill_value)
            vecs = get_eigen_compartment(mat=dsoe - 1, vecnum=numvecs)

        vecs = np.array(vecs)
        if sort_fn is not None:
            vecs = sort_fn(self, eigvecs=vecs, corr=corr)

        return self.handle_mask(vecs, full)

    @compartments
    @lru_cache(maxsize=3)
    def _clustering(self):
        return NotImplemented

    @multi_methods
    def tads(self):
        """Calling peaks"""

    @tads
    @lru_cache(maxsize=2)
    def _di_hmm(self):
        return NotImplemented

    @tads
    @lru_cache(maxsize=2)
    def _rw_tad(self):
        return NotImplemented


if __name__ == "__main__":
    pass
