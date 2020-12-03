"""Tools for Loop-detection analysis."""
from multiprocessing import Pool
from typing import Tuple, Sequence, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage, stats, sparse
from sklearn.cluster import DBSCAN
from statsmodels.stats import multitest

from .utils.utils import CPU_CORE, suppress_warning
from .utils.numtools import mask_array, index_array, Toeplitz
from .chrommatrix import ChromMatrix, Array

HKernels = Tuple[Sequence[np.ndarray], Tuple[int, int]]


@dataclass
class HiccupsPeaksFinder(object):
    chrom_ma: ChromMatrix
    inner_radius: int = 2
    outer_radius: int = 5
    band_width: int = 600
    fdrs: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1)
    sigs: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1)
    fold_changes: Tuple[float, float, float, float] = (1.5, 1.5, 1.5, 1.5)
    num_cpus: int = max(1, CPU_CORE - 2)

    def __post_init__(self):
        self.kernels: HKernels = self.fetch_kernels(self.inner_radius, self.outer_radius)

    def __call__(self) -> pd.DataFrame:
        observed = sparse.csr_matrix(self.chrom_ma.ob(sparse=True))
        decay = self.chrom_ma.decay()
        weights = self.chrom_ma.weights
        # fetch chunk slices
        chunks: Iterator[Tuple[slice, slice]] = self.get_chunk_slices(
            length=self.chrom_ma.shape[0],
            band_width=self.band_width,
            height=self.band_width,
            ov_length=2 * self.outer_radius
        )

        # fetching backgrounds model for nonzero pixles for each chunk for 4 kernels
        with Pool(processes=self.num_cpus) as pool:
            params = (
                (observed[s1, s2], (decay[s1], decay[s2]), (1 / weights[s1], 1 / weights[s2]),
                 self.kernels, self.band_width)
                for s1, s2 in chunks
            )
            backgounds = pool.starmap(self.calculate_chunk, params)

        # indices are 0-based, plus onto the start index in the original matrix
        for (indices, *_), chunk in zip(backgounds, chunks):
            x_st, y_st = chunk[0].start, chunk[1].start
            indices += np.array([[x_st], [y_st]])

        # 1. gathering backgrounds info of all nonzero pixels
        indices = np.concatenate([b[0] for b in backgounds], axis=1)
        contacts_array = np.concatenate([b[1] for b in backgounds])
        lambda_array = np.concatenate([b[2] for b in backgounds], axis=1)
        enrich_ratio = np.concatenate([b[3] for b in backgounds])
        # print(f'Before multiple test: {indices[0].size}')

        # 2. Multiple test. Filtering insignificant point after calculating padj using fdr_bh multiple test method.
        pvals, padjs, rejects = self.multiple_test(contacts_array, lambda_array, fdrs=self.fdrs, sigs=self.sigs)
        peaks = (indices, contacts_array, lambda_array, enrich_ratio, pvals, padjs)
        peaks = tuple(mask_array(np.all(rejects, axis=0), *peaks))
        # print(f'After multiple test: {peaks[0][0].size}')

        # 3. Apply greedy clustering to merge  points into confidant peaks.
        peak_indexs, shapes = self.cluster(peaks[0], peaks[1], peaks[2])
        peaks = (*tuple(index_array(peak_indexs, *peaks)), shapes)
        # print(f'After cluster: {peaks[0][0].size}')

        # 4. Filter by gap_region, fold changes(enrichment) and singlet peak's sum-qvalue.
        valid_mask = self.filter(peaks, gap_mask=~self.chrom_ma.mask, fold_changes=self.fold_changes)
        peaks = tuple(mask_array(valid_mask, *peaks))
        # indices, contacts_array, lambda_array, enrich_ratio, pvals, padjs, shape = peaks
        # print(f'After filter: {peaks[0][0].size}')

        peask_df = self.build_results(peaks, binsize=self.chrom_ma.binsize)

        return peask_df

    @staticmethod
    def fetch_kernels(p: int, w: int) -> HKernels:
        """Return kernels of four regions: donut region, vertical, horizontal, lower_left region.
        """

        def region_to_kernel(*regions) -> np.ndarray:
            for region in regions:
                kernel = np.full((2 * w + 1, 2 * w + 1), 0, dtype=np.int)
                for i, j in region:
                    kernel[i + w, j + w] = 1
                yield kernel

        def rect(x_start, x_len, y_start, y_len):
            return set((i, j)
                       for i in range(x_start, x_start + x_len)
                       for j in range(y_start, y_start + y_len))

        length = 2 * w + 1
        center = rect(-p, 2 * p + 1, -p, 2 * p + 1)
        strips = rect(-w, length, 0, 1) | rect(0, 1, -w, length)

        donut = rect(-w, length, -w, length) - (center | strips)
        vertical = rect(-w, length, -1, 3) - center
        horizontal = rect(-1, 3, -w, length) - center
        lower_left = rect(1, w, -w, w) - center

        return tuple(region_to_kernel(donut, vertical, horizontal, lower_left)), (p, w)

    @staticmethod
    def get_chunk_slices(length: int,
                         band_width: int,
                         height: int,
                         ov_length: int) -> Iterator[Tuple[slice, slice]]:
        """Return slices of all chunks along the digonal that ensure the band region with specified width is fully covered.\n
        Band region's left border is the main diagonal.
        """
        band_width *= 2
        start = 0
        while 1:
            y_end = start + band_width
            x_end = start + height
            if (y_end < length) and (x_end < length):
                yield slice(start, x_end), slice(start, y_end)
                start += height - ov_length
            else:
                yield slice(start, length), slice(start, length)
                break

    @staticmethod
    @suppress_warning
    def calculate_chunk(observed: Array,
                        exps: Tuple[np.ndarray, np.ndarray],
                        factors: Tuple[np.ndarray, np.ndarray],
                        kernels: HKernels,
                        band_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For a given chunk, calculate lambda values and contact(true counts) values of each pixel in regions specified in kernels.
        """
        ks, (r1, r2) = kernels
        num_kernels = len(ks)
        try:
            if isinstance(observed, sparse.spmatrix):
                observed = observed.toarray()
            expected = Toeplitz(*exps)[:]
            observed[np.isnan(observed)] = 0
            zero_region = observed == 0
            expected[zero_region] = 0

            # calculate lambda array for all nonzero pixels in valid region under each kernel
            x, y = observed.nonzero()
            dis = y - x
            mask = ((dis <= (band_width - 2 * r2))
                    & (x < (observed.shape[0] - r2))
                    & (dis >= r2)
                    & (x >= r2))
            x, y = x[mask], y[mask]

            if x.size == 0:
                return np.empty((2, 0)), np.empty(0), np.empty((num_kernels, 0)), np.empty(0)

            ratio_array = np.full((num_kernels, x.size), 0, dtype=np.float)

            oe_matrix = observed / expected
            for index, kernel in enumerate(ks):
                # ob_sum = ndimage.convolve(observed, kernel)
                # ex_sum = ndimage.convolve(expected, kernel)
                # ratio_array[index] = (ob_sum / ex_sum)[(x, y)]

                # Another option
                # counts = ndimage.convolve(valid_mat, kernel)
                ratio = ndimage.convolve(oe_matrix, kernel) / kernel.sum()
                ratio_array[index] = ratio[x, y]

            lambda_array = (ratio_array
                            * expected[x, y]
                            * factors[0][x]
                            * factors[1][y])

            inner_len = 2 * r1 + 1
            outer_len = 2 * r2 + 1
            inner_num = inner_len ** 2
            percentage = (inner_num / outer_len ** 2)
            plateau_ma = oe_matrix - ndimage.percentile_filter(
                oe_matrix,
                int((1 - percentage) * 100),
                (outer_len, outer_len)
            )
            plateau_region = (plateau_ma > 0).astype(np.int16)
            enrich_ratio = ndimage.convolve(
                plateau_region,
                np.ones((inner_len, inner_len))
            )[x, y] / inner_num

            nan_mask = np.isnan(lambda_array)
            lambda_array[nan_mask] = 0
            contacts_array = observed[x, y] * factors[0][x] * factors[1][y]
            non_nan_mask = ~(np.any(nan_mask, axis=0) | np.isnan(contacts_array))
            indices = np.vstack((x, y))
            # Another option is to prefilter by fold changes
            return (indices[:, non_nan_mask],
                    contacts_array[non_nan_mask],
                    lambda_array[:, non_nan_mask],
                    enrich_ratio[non_nan_mask])
        except Exception as e:
            return np.empty((2, 0)), np.empty(0), np.empty((num_kernels, 0)), np.empty(0)

    @staticmethod
    def multiple_test(contact_array: np.ndarray,
                      lambda_array: np.ndarray,
                      fdrs: Tuple[float, float, float, float],
                      sigs: Tuple[float, float, float, float],
                      method: str = "fdr_bh") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conduct poisson test on each pixel and multiple test correction for all tests.
        """

        def lambda_chunks(lambda_array: np.ndarray,
                          full: bool = False,
                          base: float = 2,
                          exponent: float = 1 / 3) -> Iterator[Tuple[float, float, np.ndarray]]:
            """Assign values in lambda_array to logarithmically spaced chunks of every base**exponent range.
            """
            min_value = np.min(lambda_array)
            num = int(np.ceil(np.log2(np.max(lambda_array)) / exponent) + 1)
            lambda_values = np.logspace(
                start=0,
                stop=(num - 1) * exponent,
                num=num,
                base=base
            )

            for start, end in zip(lambda_values[:-1], lambda_values[1:]):
                if not full and min_value > end:
                    continue
                mask = (start < lambda_array) & (lambda_array <= end)
                yield start, end, mask

        num_test, len_test = lambda_array.shape
        pvals = np.full((num_test, len_test), 1, np.float)
        padjs = np.full((num_test, len_test), 1, np.float)
        rejects = np.full((num_test, len_test), False, np.bool)

        for test_i in range(num_test):
            for _, end, lambda_mask in lambda_chunks(lambda_array[test_i]):
                chunk_size = lambda_mask.sum()
                if chunk_size == 0:
                    continue
                # poisson_model = stats.poisson(np.ones(chunk_size) * end)
                poisson_model = stats.poisson(lambda_array[test_i, lambda_mask])
                _pvals = 1 - poisson_model.cdf(contact_array[lambda_mask])
                reject, _padjs, _, _ = multitest.multipletests(
                    pvals=_pvals,
                    alpha=fdrs[test_i],
                    method=method
                )
                rejects[test_i][lambda_mask] = reject
                padjs[test_i][lambda_mask] = _padjs
                pvals[test_i][lambda_mask] = _pvals

        rejects = rejects & (padjs < np.array(sigs)[:, None])

        return pvals, padjs, rejects

    @staticmethod
    def cluster(indices: np.ndarray,
                contacts: np.ndarray,
                lambda_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dbscan = DBSCAN(2)
        dbscan.fit(indices.T)

        peak_indexs, shapes = [], []
        for cluster_id in set(dbscan.labels_) - {-1}:
            point_indexs = np.where(dbscan.labels_ == cluster_id)[0]
            points = indices[:, point_indexs]
            center_index = np.argmax(
                (contacts[point_indexs] / lambda_array[:, point_indexs]).sum(axis=0)
            )
            center = points[:, center_index]
            width = np.abs(points[1] - center[1]).max() * 2 + 1
            height = np.abs(points[0] - center[0]).max() * 2 + 1
            peak_indexs.append(point_indexs[center_index])
            if height >= 2 * width:
                height = width
            elif width >= 2 * height:
                width = height
            shapes.append([width, height])

        for singlet_index in np.where(dbscan.labels_ == -1)[0]:
            peak_indexs.append(singlet_index)
            shapes.append([1, 1])

        return np.array(peak_indexs), np.array(shapes).T

    @staticmethod
    def filter(peaks: tuple,
               gap_mask: np.ndarray,
               fold_changes: Tuple[float, float, float, float] = (2, 1.5, 1.5, 2)) -> np.ndarray:
        """Post-filtering peaks after filtered by mulitple test and megred by clustering:\n
            1. Remove peaks close to gap region(bad bins).\n
            3. Retain peaks with fold changes over a given threshold in four regions.\n
        """

        def enrich_mask(contact_array: np.ndarray,
                        lambda_array: np.ndarray,
                        enrich_ratio: np.ndarray) -> np.ndarray:
            """Return mask of valid peaks passed the enrichment fold changes filtering."""
            fc_mask = np.all(contact_array
                             >= lambda_array * np.array(fold_changes)[:, None], axis=0)
            ec_mask = enrich_ratio > 0.4

            return fc_mask & ec_mask

        def away_gap_mask(indices, gap_mask, extend_width) -> np.ndarray:
            """Return mask of valid peaks away from gap regions."""
            for i in range(extend_width):
                gap_mask |= np.r_[gap_mask[1:], [False]]
                gap_mask |= np.r_[[False], gap_mask[: -1]]
            gap_region = set(np.where(gap_mask)[0])

            return ~np.array([i in gap_region or j in gap_region
                              for i, j in zip(*indices)])

        indices, contact_array, lambda_array, enrich_ratio, pvals, padjs, shapes = peaks

        return away_gap_mask(indices, gap_mask, 1) & enrich_mask(contact_array, lambda_array, enrich_ratio)

    @staticmethod
    def build_results(peaks_tuple: tuple, binsize: int = 1) -> pd.DataFrame:
        """Aggregate peak-infos into a pd.DataFrame object.
        """
        region_names = ['donut', 'horizontal', 'vertical', 'lower_left']
        num_region = len(region_names)
        col_names = (['i', 'j', 'ob']
                     + ['ex_' + region for region in region_names]
                     + ['pval_' + region for region in region_names]
                     + ['padj_' + region for region in region_names]
                     + ['enrich_ratio', 'width', 'height'])
        dtypes = [np.int] * 3 + [np.float] * (len(col_names) - 3)

        if peaks_tuple:
            indices, contacts_array, lambda_array, enrich_ratio, pvals, padjs, shape = peaks_tuple
            peaks: np.ndarray = np.zeros(shape=contacts_array.size,
                                         dtype=[(col_name, dtype) for col_name, dtype in zip(col_names, dtypes)])
            fields_name = list(peaks.dtype.names)
            peaks['i'], peaks['j'], peaks['ob'] = indices[0], indices[1], contacts_array
            peaks[fields_name[3: 3 + num_region]] = list(zip(*lambda_array))
            peaks[fields_name[3 + num_region: 3 + 2 * num_region]] = list(zip(*pvals))
            peaks[fields_name[3 + 2 * num_region: 3 + 3 * num_region]] = list(zip(*padjs))
            peaks[fields_name[-3]] = enrich_ratio
            peaks[fields_name[-2:]] = list(zip(*shape))
            peaks_df = pd.DataFrame(peaks)
            if binsize is not None and binsize > 1:
                peaks_df[['i', 'j', 'width', 'height']] *= binsize
            return peaks_df
        else:
            return pd.DataFrame(columns=col_names).astype(
                {name: t for name, t in zip(col_names, dtypes)}
            )


@dataclass
class ImagePeaksFinder(object):
    """Idea:
    1: Use maximum filter to find local maximum for each region. -> points.
    2: Use percentage filter to find enriched regions. -> regions.
    3: Design a statitical test method for the assessment pf  randomness and enrichment. -> filter points.
    """
    pass


if __name__ == "__main__":
    pass
