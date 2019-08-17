"""
Tools for Loop-detection analysis.
"""
# TODO(zhongquan789@gmail.com) implement APA and multi-resolution combination methods.
# TODO(zhongquan789@126.com) find a way to automatically choose p and w used in hiccups.
# Possible way:
#    find highly enriched pixels(Need carefully consideration for the effect of decay.)
#    Use aggregate analysis to find a decaying pattern as a reference for automatically chossing p and w.
#    This should be fast.
# TODO(zhongquan789@126.com) implement cloops algorithm.

from collections import OrderedDict
from collections import defaultdict
from functools import partial
from typing import Tuple, Iterable, List, Callable

import numpy as np
import pandas as pd
import ray
from scipy import ndimage, stats
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from statsmodels.stats import multitest

from .utils import remove_small_gap, suppress_warning, mask_array, index_array


def extend_gap(gap_mask: np.ndarray, extend_width: int, remove_single=True) -> np.ndarray:
    """Create indexs of new mask which are extended from original gap mask for a certain width.

    :param gap_mask: np.ndarray. True values represent gap regions.
    :param extend_width: int. Extending width in both ends of each gap region.
    :param remove_single: bool. Remove singlet gaps.
    :return: set. indexs of extended gap regions.
    """
    if remove_single:
        gap_mask = remove_small_gap(gap_mask)

    for i in range(extend_width):
        gap_mask |= np.r_[gap_mask[1:], [False]]
        gap_mask |= np.r_[[False], gap_mask[: -1]]

    return gap_mask


def diff_kernel(new_kernels: Tuple[np.ndarray], old_kernels: Tuple[np.ndarray]):
    """Fetch the difference between two kernels to avoid the redundant computation.\n
    Shape of old_kernel must smaller than that of new_kernel, thus old_kernel should be subset of new_kernel.

    :param new_kernels: tuple. tuple of kernels.
    :param old_kernels: tuple. tuple of kernels.
    :return: tuple.
    """
    shape = new_kernels[0].shape
    diff_len = int((new_kernels[0].shape[0] - old_kernels[0].shape[0]) / 2)
    diff_kernels = []
    for old, new in zip(old_kernels, new_kernels):
        old = old.astype(np.bool)
        tmp = np.full(shape, True)
        tmp[diff_len: -diff_len, diff_len: -diff_len] = ~old
        diff_kernels.append((new & tmp).astype(np.int))
    return tuple(diff_kernels)


def lambda_chunks(lambda_array: np.ndarray,
                  full: bool = False,
                  base: float = 2,
                  exponent: float = 1 / 3) -> Tuple[float, float, np.ndarray]:
    """Assign values in lambda_array to logarithmically spaced chunks of every base**exponent range.

    :param lambda_array:
    :param full: bool. If return the full lambda chunks that start from zero(otherwise from min number).
    :param base:
    :param exponent:
    :return: list. containing tuples of (start, end, indexs) of each chunks.
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


def fetch_regions(p: int,
                  w: int,
                  kernel: bool = False) -> Tuple[np.ndarray]:
    """Return coordinates of pixels whithin four regions: donut region, vertical, horizontal, lower_left region.

    :param p: int. radius of center square.
    :param w: int. radius of outer square.
    :param kernel: bool. if return mask format
    :return: tuple.
    """

    def region_to_kernel(*regions):
        kernels = []
        for region in regions:
            kernel = np.full((2 * w + 1, 2 * w + 1), 0, dtype=np.int)
            for i, j in region:
                kernel[i + w, j + w] = 1
            kernels.append(kernel)
        return tuple(kernels)

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

    regions = donut, vertical, horizontal, lower_left

    if kernel:
        return region_to_kernel(*regions)
    else:
        return regions


def cluster(indices: Tuple[np.ndarray, np.ndarray], contacts: np.ndarray, initial_dis: float):
    """Use DBSCAN followed by greedy clustering to merge pixels into confidant peaks.

    :param indices: np.ndarray. Coordinates of all pixels. The first(second) array represents x(y) coordinate of each pixel.
    :param contacts: np.ndarray. Contacts(unnormed) counts of each pixel.
    :param initial_dis: float. Initial distance used for DBSCAN and greedy clustering.
    :return: tuple. return indexes and infos(center coordinates and radius) of each merged peak.
    """

    def create_subpeaks(indexs: np.ndarray, peaks: List[Tuple[int, int]]):
        sub_contacts = [(contacts[i], i) for i in indexs.tolist()]
        sub_contacts.sort(reverse=True, key=lambda v: v[0])
        sub_peaks = [peaks[contact[1]] for contact in sub_contacts]
        return sub_peaks

    def default_info(peak: Tuple[int, int]):
        return peak[0], peak[1], initial_dis

    peaks = list(zip(*indices))
    peak_to_index = {peaks[i]: i for i in range(len(peaks))}
    db = DBSCAN(eps=initial_dis, min_samples=2).fit(np.array(peaks))
    labels = db.labels_
    clusters = []

    for label in set(labels):
        if label == -1:
            continue
        indexs = np.where(labels == label)[0]

        clusters.append(create_subpeaks(indexs, peaks))

    peaks_index = []
    peaks_info = []
    for sub_peaks in clusters:
        merged_peaks, infos = greedy_cluster(sub_peaks, initial_dis)
        peaks_index.extend(peak_to_index[sub_peaks[center_index]]
                           for center_index in merged_peaks)
        peaks_info.extend(infos)

    for index in np.where(labels == -1)[0]:
        peaks_index.append(index)
        peaks_info.append(default_info(peaks[index]))

    return peaks_index, np.array(peaks_info).T


def greedy_cluster(peaks: List[Tuple[int, int]], initial_dis: float) -> Tuple[list, list]:
    """Cluster pixels which are densely distributed in a certain region into a peak by using a greedy clustering method.

    :param peaks:  list. coordinates of all peaks. coordinates are represented by tuples of x and y. e.g (x ,y)
    :param initial_dis: Initital distance for consider pixels as neighbors of a peak center.
    :return: tuple. return indexs and infos(center coordinates and radius) of all peaks.
    """
    record_dis = initial_dis
    initial_dis = max(initial_dis, 1.5)

    def update_center_radius(indexs: List[int], new_index: List[int]):
        """

        :param indexs:
        :param new_index:
        :return:
        """
        peaks_num = len(indexs)
        x, y = 0, 0
        for i in range(peaks_num):
            x += peaks[indexs[i]][0]
            y += peaks[indexs[i]][1]
        _center = (x / peaks_num, y / peaks_num)
        max_radius = max(euclidean(center, peaks[i]) for i in new_index)
        return _center, max_radius + initial_dis

    left_mask = np.full(len(peaks), True)
    final_peaks = []
    infos = []

    while np.any(left_mask):
        left_index = np.where(left_mask)[0]
        center_index, other_indexs = left_index[0], left_index[1:]
        center = peaks[center_index]
        radius = initial_dis
        local = [center_index]
        visited = [center_index]
        merged_nums = None
        while merged_nums is None or merged_nums:
            merged_nums = 0
            remains = []
            new_local = []
            for index in other_indexs:
                if euclidean(center, peaks[index]) <= radius:
                    local.append(index)
                    new_local.append(index)
                    visited.append(index)
                    merged_nums += 1
                else:
                    remains.append(index)
            if new_local:
                center, radius = update_center_radius(local, new_local)
            other_indexs = remains

        center_x = round(center[0])
        center_y = round(center[1])
        final_peaks.append(center_index)
        if radius == 1.5:
            radius = record_dis
        infos.append((center_x, center_y, radius))
        left_mask[visited] = False

    return final_peaks, infos


def get_chunk_slices(length: int,
                     band_width: int,
                     height: int,
                     ov_length: int) -> Tuple[slice, slice]:
    """Return slices of all chunks along the digonal that ensure the band region with specified width is fully covered.\n
    Band region's left border is the main diagonal.

    :param length: int. The length of input square matrix.
    :param band_width: int. Width of the band region.
    :param height: int. Height of each chunk(submatrix).
    :param ov_length: int. Number of overlapping bins of two adjacent chunk(submatrix) along the main diagonal.
    :return: Tuple[slice, slice]. x,y slice of each chunk(submatrix).
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


def multiple_test(contact_array: np.ndarray,
                  lambda_array: np.ndarray,
                  fdrs: Tuple[float],
                  sigs: Tuple[float],
                  method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conduct poisson test on each pixel and multiple test correction for all tests.

    :param contact_array: np.ndarray. 1-d array contains observed contacts(number of counts) of all pixels.
    :param lambda_array: np.ndarray. 2-d array contains lambdas(background) of all pixels.
    'lambda_array.shape[0]' denotes the number of backgrounds and 'lambda_array.shape[1]' denotes the number of pixles.
    :param fdrs: tuple. Tuple of fdrs to control the false discovery rate for each background.
    Length of this tuple must equal to 'lambda_array.shape[0]'
    :param sigs: tuple. Tuple of padjs threshold. Length of this tuple must equal to 'lambda_array.shape[0]'
    :param method: str. Method used for multiple test. Supported methods see: statsmodels.stats.multitest.
    :return: Tuple[pvals, padjs, rejects]. pvals and padjs are both 2-d ndarrays with the same shape as 'lambda_array'.
    rejects is a 1-d Boolean array with shape of 'lambda_array.shape[1]', each value represents whether reject the null hypothesis.
    """
    num_test, len_test = lambda_array.shape
    pvals = np.full((num_test, len_test), 1, np.float)
    padjs = np.full((num_test, len_test), 1, np.float)
    rejects = np.full((num_test, len_test), False, np.bool)

    for test_i in range(num_test):
        for _, end, lamda_mask in lambda_chunks(lambda_array[test_i]):
            chunk_size = lamda_mask.sum()
            if chunk_size == 0:
                continue
            poisson_model = stats.poisson(np.ones(chunk_size) * end)
            _pvals = 1 - poisson_model.cdf(contact_array[lamda_mask])
            reject, _padjs, _, _ = multitest.multipletests(
                pvals=_pvals,
                alpha=fdrs[test_i],
                method=method
            )
            rejects[test_i][lamda_mask] = reject
            padjs[test_i][lamda_mask] = _padjs
            pvals[test_i][lamda_mask] = _pvals

    rejects = rejects & (padjs < np.array(sigs)[:, None])

    return pvals, padjs, rejects


def additional_filtering(peaks: tuple,
                         factors: np.ndarray,
                         sum_qvalue: float = 0.02,
                         single_fcs: tuple = (1.7, 1.5, 1.5, 1.75),
                         double_fcs: tuple = (2, 0, 0, 2),
                         ignore_single_gap: bool = True):
    """Post-filtering peaks after filtered by mulitple test and megred by clustering:
        1. Remove peaks close to gap region(bad bins).\n
        2. Remove singlet peaks with sums of four padjs lesser than 'sum_qvalues' .\n
        3. Remove peaks with fold changes less than a given threshold in either one of four regions.\n
        4. Retain peaks with fold changes over a given threshold in either one of four regions.\n

    :param peaks: Tuple[np.ndarray, np.ndarray]. Tuple contains corrdinates of all peak.
    :param factors: np.ndarray. ICE factors for detecting gaps(bad bins.). Bad bins should bed marked with np.nan.
    :param sum_qvalue: float. Threshold for filtering those peaks with sum padjs of four regions lesser than a certain value.
    :param single_fcs: tuple. Padjs threshold for each region. Valid peak's padjs should pass all four fold-change thresholds.
    :param double_fcs: tuple. Padjs threshold for each region. Valid peak's padjs should pass either one of four fold-change thresholds.
    :param ignore_single_gap: bool. If ignore small gaps when filtering peaks close to gap regions.
    :return: np.ndarray. Boolean array of valid peaks passed all filtering.
    """

    def fold_change_mask(contact_array: np.ndarray,
                         lambda_array: np.ndarray) -> np.ndarray:
        """Return mask of valid peaks passed the enrichment fold changes filtering.
        """
        single_fc_mask = np.all(contact_array
                                >= lambda_array * np.array(single_fcs)[:, None], axis=0)
        double_fc_mask = np.any(contact_array
                                >= lambda_array * np.array(double_fcs)[:, None], axis=0)

        return single_fc_mask & double_fc_mask

    def gap_mask(indices: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return mask of valid peaks away from gap regions.
        """
        gap_region = set(np.where(extend_gap(np.array(np.isnan(factors)), 1, ignore_single_gap))[0])

        return ~np.array([i in gap_region or j in gap_region
                          for i, j in zip(*indices)])

    def singlet_mask(peak_info: np.ndarray,
                     padjs: np.ndarray,
                     initial_dis: float):
        """Return mask of valid peaks with sum padjs of four regions greater than sum_qvalue threshold.
        """
        mask = peak_info[-1] == initial_dis
        mask[mask] &= (padjs.sum(axis=0)[mask] < sum_qvalue)
        return ~mask

    indices, contact_array, lambda_array, pvals, padjs, peak_info = peaks

    return (gap_mask(indices)
            & singlet_mask(peak_info, padjs, peak_info[-1].min())
            & fold_change_mask(contact_array, lambda_array))


def build_results(peaks: tuple) -> pd.DataFrame:
    """Aggregate peak-infos into a pd.DataFrame object.

    :param peaks: tuple. Tuple containing all infos related to peaks returned by find_peaks.
    :return: pd.DataFrame.
    """
    region_names = ['donut', 'horizontal', 'vertical', 'lower_left']
    num_region = len(region_names)
    col_names = (['i', 'j', 'ob']
                 + ['ex_' + region for region in region_names]
                 + ['pval_' + region for region in region_names]
                 + ['padj_' + region for region in region_names]
                 + ['center_i', 'center_j', 'radius'])

    dtypes = [np.int] * 3 + [np.float] * (len(col_names) - 3)

    if peaks:
        indices, contacts_array, lambda_array, pvals, padjs, peak_info = peaks
        peaks = np.zeros(shape=contacts_array.size,
                         dtype=[(col_name, dtype) for col_name, dtype in zip(col_names, dtypes)])
        fields_name = list(peaks.dtype.names)
        peaks['i'], peaks['j'], peaks['ob'] = indices[0], indices[1], contacts_array
        peaks[fields_name[3: 3 + num_region]] = list(zip(*lambda_array))
        peaks[fields_name[3 + num_region: 3 + 2 * num_region]] = list(zip(*pvals))
        peaks[fields_name[3 + 2 * num_region: 3 + 3 * num_region]] = list(zip(*padjs))
        peaks[fields_name[-3:]] = list(zip(*peak_info))

        return pd.DataFrame(peaks)
    else:
        peaks = pd.DataFrame(columns=col_names)
        peaks = peaks.astype({name: t for name, t in zip(col_names, dtypes)})
        return peaks


@suppress_warning
def calculate_lambda(expected: np.ndarray,
                     observed: np.ndarray,
                     row_factors: np.ndarray,
                     col_factors: np.ndarray,
                     kernels: Tuple[np.ndarray],
                     band_width: int,
                     outer_radius: int,
                     ignore_diags: int = 3) -> Tuple[tuple, np.ndarray]:
    """Calculate lambda values(background) for each pixel in regions sepcified in kernels.

    :param expected: np.ndarray. 2-d ndarray represents expeceted(normed) matrix.
    :param observed: np.ndarray. 2-d ndarray represents observed(normed) matrix.
    :param row_factors: np.ndarray. 1-d ndarray represents ICE factors of each row in expected.
    :param col_factors: np.ndarray. 1-d ndarray represents ICE factors of each column in expected.
    :param kernels: Tuple[np.ndarray]. Each array(mask) represents a certain region that is used for computing
    lambda(background) by summing all values within this region for each pixel.
    :param band_width: int. Width of the band region.
    :param outer_radius: int. The maximum radius among all kernels.
    :param ignore_diags: int. Number of diagonals to ignore. Pixles within this region will not be counted in available contacts.
    :return: Tuple[tuple, np.ndarray]. The first tuple contains indices of all available pixels, and the second ndarray
    contains the corresponding lambdas in all regions specified in kernels.
    """
    try:
        x, y = observed.nonzero()
        dis = y - x
        mask = ((dis <= (band_width - 2 * outer_radius))
                & (x < (band_width - outer_radius))
                & (dis >= ignore_diags - 1)
                & (x >= outer_radius))
        x, y = x[mask], y[mask]

        ratio_array = np.full((len(kernels), x.size), 0, dtype=np.float)
        for index, kernel in enumerate(kernels):
            ob_sum = ndimage.convolve(observed, kernel)
            ex_sum = ndimage.convolve(expected, kernel)
            ratio_array[index] = (ob_sum / ex_sum)[(x, y)]

        lambda_array = (ratio_array
                        * expected[(x, y)]
                        * row_factors[x]
                        * col_factors[y])

        return (x, y), lambda_array

    except Exception as e:

        return (np.array([]), np.array([])), np.array([])


@ray.remote
def calculate_chunk(expected_fetcher: Callable[[str, tuple], np.ndarray],
                    observed_fetcher: Callable[[str, tuple], np.ndarray],
                    factors_fetcher: Callable[[str, tuple], tuple],
                    chunk: Tuple[str, Tuple[slice, slice]],
                    kernels: Tuple[np.ndarray],
                    band_width: int,
                    outer_radius: int) -> Tuple[tuple, np.ndarray, np.ndarray]:
    """For a given chunk, calculate lambda values and contact(true counts) values of each pixel in regions specified in kernels.

    :param expected_fetcher: Callable[[str, tuple], np.ndarray]. Callable object that return a 2-d submatrix
    representing expected matrix in a certain region.
    :param observed_fetcher: Callable[[str, tuple], np.ndarray]. Callable object that return a 2-d submatrix
    representing observed matrix in a certain region.
    :param factors_fetcher: Callable[[str, tuple], tuple]. Callable object that return a tuple of 1-d ndarray
    representing ICE factors of row and column in a certain region.
    :param chunk: Tuple[str, Tuple[slice, slice]]. Tuple used as arguments for fetching data by using fetcher functions.
    :param kernels: Tuple[np.ndarray]. Each array(mask) represents a certain region that is used for computing
    lambda(background) by summing all values within this region for each pixel.
    :param band_width: int. Width of the band region.
    :param outer_radius: int. The maximum radius among all kernels.
    :return: Tuple[indices, lambdas, contacts]. coordinates, lambda values and contact values of all valid pixels.
    """
    key, slices = chunk
    observed = observed_fetcher(key, slices)
    expected = expected_fetcher(key, slices)
    observed[np.isnan(observed)] = 0
    expected[np.isnan(expected)] = 0
    row_factors, col_factors = factors_fetcher(key, slices)
    row_factors = 1 / row_factors
    col_factors = 1 / col_factors

    indices, lambda_array = calculate_lambda(
        expected=expected,
        observed=observed,
        row_factors=row_factors,
        col_factors=col_factors,
        kernels=kernels,
        band_width=band_width,
        outer_radius=outer_radius
    )

    if indices[0].size == 0:
        return (np.array([]), np.array([])), np.array([]), np.array([])
    else:
        contacts_array = (observed[indices]
                          * row_factors[indices[0]]
                          * col_factors[indices[1]])

        true_indices = (indices[0] + slices[0].start, indices[1] + slices[1].start)

        return true_indices, lambda_array, contacts_array


@ray.remote
@suppress_warning
def find_peaks(backgrounds: list,
               test_fn: callable,
               cluster_fn: callable,
               filter_fn: callable) -> tuple:
    """Find peaks from pixels with pre-computed lambdas and contacts by applying multiple test, clustering
    and additional filtering.

    :param backgrounds: list. Each element is a tuple containing indices, lambdas and contacts of pixels.
    :param test_fn: callable. Used for applying statistical test for all pixels.
    :param cluster_fn: callable. Used for merging(clustering pixels into peaks.
    :param filter_fn: callable. Used for filtering peaks find by clustering.
    :return: tuple. Tuple of arrays.
    """
    # load data from ray store
    try:
        x_indice = []
        y_indice = []
        lambda_array = []
        contacts_array = []
        for ray_id in backgrounds:
            (_x_indice, _y_indice), _lambda_array, _contacts_array = ray.get(ray_id)
            if len(_x_indice) == 0:
                continue
            x_indice.append(_x_indice)
            y_indice.append(_y_indice)
            lambda_array.append(_lambda_array)
            contacts_array.append(_contacts_array)

        indices = (np.concatenate(x_indice), np.concatenate(y_indice))
        lambda_array = np.concatenate(lambda_array, axis=1)
        contacts_array = np.concatenate(contacts_array)
        # multiple test
        pvals, padjs, rejects = test_fn(contacts_array, lambda_array)
        peaks = indices, contacts_array, lambda_array, pvals, padjs

        # Filtering insignificant point after calculating padj using fdr_bh multiple test method.
        reject = np.all(rejects, axis=0)
        peaks = tuple(mask_array(reject, *peaks))

        # Apply greedy clustering to merge  points into confidant peaks.
        peak_index, peak_info = cluster_fn(peaks[0], peaks[1])
        peaks = tuple(index_array(peak_index, *peaks))
        peaks = (*peaks, peak_info)

        # Filter by gap_region, fold changes(enrichment) and singlet peak's sum-qvalue.
        valid_mask = filter_fn(peaks)
        peaks = tuple(mask_array(valid_mask, *peaks))

        return peaks

    except Exception as e:
        return tuple()


@suppress_warning(warning_msg=ResourceWarning)
def hiccups(expected_fetcher: Callable[[str, tuple], np.ndarray],
            observed_fetcher: Callable[[str, tuple], np.ndarray],
            factors_fetcher: Callable[[str, tuple], np.ndarray],
            chunks: Iterable[Tuple[str, tuple]],
            kernels: Tuple[np.ndarray],
            num_cpus: int = 20,
            max_dis: int = 1000000,
            resolution: int = 10000,
            method: str = 'fdr_bh',
            fdrs: tuple = (0.01, 0.01, 0.01, 0.01),
            sigs: tuple = (0.01, 0.01, 0.01, 0.01),
            single_fcs: tuple = (1.7, 1.5, 1.5, 1.75),
            double_fcs: tuple = (2, 0, 0, 2),
            ignore_single_gap: bool = True,
            bin_index: bool = True) -> pd.DataFrame:
    """Call peaks using hiccups algorithm.

    :param expected_fetcher: Callable[[str, tuple], np.ndarray]. Callable object that return a 2-d submatrix
    representing expected matrix for each chunk(region).
    :param observed_fetcher: Callable[[str, tuple], np.ndarray]. Callable object that return a 2-d submatrix
    representing observed matrix for each chunk(region).
    :param factors_fetcher: Callable[[str, tuple], tuple]. Callable object that return a tuple of 1-d ndarray
    representing ICE factors of row and column for each chunk(region).
    :param chunks: Iterable[Tuple[str, tuple]]. Only chunk regions in chunks will be examined for detecting peaks.
    Elements of chunks will send to fetcher functions as arguments to fetch corresponding datas required for detecing peaks.
    :param kernels: Tuple[np.ndarray]. Each array(mask) represents a certain region that is used for computing
    lambda(background) by summing all values within this region for each pixel.
    :param num_cpus: int. Number of cores to call peaks. Calculation based on chunks and process of find peaks based
    on different chromosome will run in parallel.
    :param max_dis: int. Max distance of loops to find. Due to the natural property of loops(distance less than 8M) and
    computation bound of hiccups algorithm, hiccups algorithm are applied only in a band region over the main digonal
    to speed up the whole process.
    :param resolution: int. Resolution of input data to find peaks. This is used for setting initial distance in clustering step.
    :param method: str. Method used for multiple test. Supported methods see: statsmodels.stats.multitest.
    :param fdrs: tuple. Tuple of fdrs to control the false discovery rate for each background.
    :param sigs: tuple. Tuple of padjs thresholds for each background.
    :param single_fcs: tuple. Padjs threshold for each region. Valid peak's padjs should pass all four fold-hange thresholds.
    :param double_fcs: tuple. Padjs threshold for each region. Valid peak's padjs should pass either one of four fold-change thresholds.
    :param ignore_single_gap: bool. If ignore small gaps when filtering peaks close to gap regions.
    :param bin_index: bool. Return actual genomic positions of peaks if set to False.
    :return: pd.DataFrame.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    initial_dis = max(20000 // resolution, 1)
    outer_radius = (max(kernel.shape[0] for kernel in kernels) - 1) // 2
    backgrounds_dict = defaultdict(list)
    for chunk in chunks:
        key = chunk[0]
        backgrounds_dict[key].append(
            calculate_chunk.remote(
                expected_fetcher=expected_fetcher,
                observed_fetcher=observed_fetcher,
                factors_fetcher=factors_fetcher,
                chunk=chunk,
                kernels=kernels,
                band_width=max_dis // resolution,
                outer_radius=outer_radius
            )
        )

    test_fn = partial(
        multiple_test,
        fdrs=fdrs,
        sigs=sigs,
        method=method
    )

    cluster_fn = partial(
        cluster,
        initial_dis=initial_dis
    )

    filter_fn = partial(
        additional_filtering,
        double_fcs=double_fcs,
        single_fcs=single_fcs,
        ignore_single_gap=ignore_single_gap
    )

    peaks_dict = OrderedDict()
    full_slices = (slice(0, None), slice(0, None))
    for key, backgrounds in backgrounds_dict.items():
        _filter_fn = partial(filter_fn, factors=factors_fetcher(key, full_slices)[0])
        peaks_dict[key] = find_peaks.remote(
            backgrounds=backgrounds,
            test_fn=test_fn,
            cluster_fn=cluster_fn,
            filter_fn=_filter_fn
        )
    chroms_list = []
    for key in peaks_dict.keys():
        peaks_dict[key] = build_results(ray.get(peaks_dict[key]))
        chroms_list.extend(key for i in range(peaks_dict[key].shape[0]))

    chroms_list = np.array(chroms_list)
    peaks_df = pd.concat([peaks_dict[chrom] for chrom in peaks_dict.keys()])
    peaks_df.insert(0, 'chrom', chroms_list)

    if not bin_index:
        for col in ['i', 'j', 'center_i', 'center_j', 'radius']:
            peaks_df[col] *= resolution

    return peaks_df


def expected_fetcher(key, slices, expected_dict):
    if isinstance(expected_dict[key], ray._raylet.ObjectID):
        expected_dict[key] = ray.get(expected_dict[key])
    return expected_dict[key][slices]


def observed_fetcher(key, slices, cool, start_dict):
    row_st, row_ed = slices[0].start + start_dict[key], slices[0].stop + start_dict[key]
    col_st, col_ed = slices[1].start + start_dict[key], slices[1].stop + start_dict[key]
    return cool.matrix()[slice(row_st, row_ed), slice(col_st, col_ed)]


def factors_fetcher(key, slices, factor_dict):
    return factor_dict[key][slices[0]], factor_dict[key][slices[1]]


def chunks_gen(chromsizes, band_width, height, ov_length):
    for chrom, size in chromsizes.items():
        for slices in get_chunk_slices(size, band_width, height, ov_length):
            yield chrom, slices


if __name__ == "__main__":
    pass
