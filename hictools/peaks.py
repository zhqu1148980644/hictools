"""
Tools for Loop-detection analysis.
"""
# TODO(zhongquan789@gmail.com) implement APA and multi-resolution combination methods.
# Possible way:
#    find highly enriched pixels(Need carefully consideration for the effect of decay.)
#    Use aggregate analysis to find a decaying pattern as a reference for automatically chossing p and w.
#    This should be fast.

from collections import defaultdict
from functools import partial
from typing import Tuple, Union, Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd
import ray
from scipy import ndimage, stats
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from statsmodels.stats import multitest

from utils import remove_small_gap, suppress_warning, mask_array, index_array


def extend_gap(gap_mask: np.ndarray, extend_width: int, remove_single=True) -> np.ndarray:
    """Create indexs of new mask which is  extended from original gap mask for a certain width.

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


def diff_kernel(new_kernels, old_kernels):
    """Fetch the difference between two kernels to avoid the redundant computation.\n
    Shape of old_kernel must smaller than that of new_kernel and old_kernel should be subset of new_kernel.

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
    lambda_values = np.logspace(0, (num - 1) * exponent, num, base=base)

    for start, end in zip(lambda_values[:-1], lambda_values[1:]):
        if not full and min_value > end:
            continue
        mask = (start < lambda_array) & (lambda_array <= end)
        yield start, end, mask


def fetch_regions(p: int,
                  w: int,
                  kernel: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def cluster(indices: np.ndarray, contacts: np.ndarray, initial_dis: float):
    """Use DBSCAN followed by greedy clustering to merge pixels into confidant peaks.

    :param indices: np.ndarray. coordinates of all peaks. coordinates are represented by tuple of x and y. e.g (x ,y)
    :param contacts: np.ndarray.
    :param initial_dis: float. Initial distance used for DBSCAN and greedy clustering.
    :return: tuple. return merged peak indexs and infos(center coordinates and radius) of each peaks.
    """

    def create_subpeaks(indexs, peaks):
        sub_contacts = [(contacts[i], i) for i in indexs]
        sub_contacts.sort(reverse=True, key=lambda v: v[0])
        sub_peaks = [peaks[contact[1]] for contact in sub_contacts]
        return sub_peaks

    def default_info(peak):
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
        peaks_index.extend(peak_to_index[sub_peaks[center_index]] for center_index in merged_peaks)
        peaks_info.extend(infos)

    for index in np.where(labels == -1)[0]:
        peaks_index.append(index)
        peaks_info.append(default_info(peaks[index]))

    return peaks_index, np.array(peaks_info).T


def greedy_cluster(peaks: list, initial_dis: float) -> Tuple[list, list]:
    """Cluster pixels which are densely distributed in a certain region into a peak by using a greedy clustering method.

    :param peaks:  list. coordinates of all peaks. coordinates are represented by tuple of x and y. e.g (x ,y)
    :param initial_dis: Initital distance for consider pixels as neighbors of a peak center.
    :return:  tuple. return merged peak indexs and infos(center coordinates and radius) of each peaks.
    """
    record_dis = initial_dis
    initial_dis = max(initial_dis, 1.5)

    def update_center_radius(indexs, new_index):
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
    """

    :param length:
    :param band_width:
    :param height:
    :param ov_length:
    :return:
    """
    band_width *= 2
    start = 0
    while 1:
        y_end = start + band_width
        if y_end < length:
            yield slice(start, start + height), slice(start, y_end)
            start += height - ov_length
        else:
            yield slice(start, length), slice(start, length)
            break


def multiple_test(contact_array: np.ndarray,
                  lambda_array: np.ndarray,
                  fdrs: tuple,
                  sigs: tuple,
                  method: str):
    """

    :param contact_array:
    :param lambda_array:
    :param fdrs:
    :param sigs:
    :param method:
    :return:
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
            reject, _padjs, _, _ = multitest.multipletests(pvals=_pvals,
                                                           alpha=fdrs[test_i],
                                                           method=method)
            rejects[test_i][lamda_mask] = reject & (_padjs <= sigs[test_i])
            padjs[test_i][lamda_mask] = _padjs
            pvals[test_i][lamda_mask] = _pvals

    return pvals, padjs, rejects


def additional_filtering(peaks,
                         factors,
                         sum_qvalue: float = 0.02,
                         single_fcs: Union[float, tuple] = (1.7, 1.5, 1.5, 1.75),
                         double_fcs: Union[float, tuple] = (2, 0, 0, 2),
                         ignore_single_gap: bool = True):
    """

    :param peaks:
    :param factors:
    :param sum_qvalue:
    :param single_fcs:
    :param double_fcs:
    :param ignore_single_gap:
    :return:
    """

    def fold_change_mask(contacts_array: np.ndarray,
                         lambda_array: np.ndarray,
                         single_fcs,
                         double_fcs) -> np.ndarray:
        """

        :param contacts_array:
        :param lambda_array:
        :param single_fcs:
        :param double_fcs:
        :return:
        """
        single_fc_mask = np.all(contacts_array
                                >= lambda_array * np.array(single_fcs)[:, None], axis=0)
        double_fc_mask = np.any(contacts_array
                                >= lambda_array * np.array(double_fcs)[:, None], axis=0)

        return single_fc_mask & double_fc_mask

    def gap_mask(indices: tuple, factors) -> np.ndarray:
        """

        :param indices:
        :param factors:
        :return:
        """
        gap_region = set(np.where(extend_gap(np.array(np.isnan(factors)), 1, ignore_single_gap))[0])

        return ~np.array([i in gap_region or j in gap_region
                          for i, j in zip(*indices)])

    def singlet_mask(peak_info: np.ndarray,
                     padjs: np.ndarray,
                     initial_dis: float,
                     sum_qvalue):
        """

        :param peak_info:
        :param padjs:
        :param initial_dis:
        :param sum_qvalue:
        :return:
        """
        mask = peak_info[-1] == initial_dis
        mask[mask] &= (padjs[-1][mask].sum() < sum_qvalue)
        return ~mask

    indices, contact_array, lambda_array, pvals, padjs, peak_info = peaks

    return gap_mask(indices, factors) \
           & singlet_mask(peak_info, padjs, peak_info[-1].min(), sum_qvalue) \
           & fold_change_mask(contact_array, lambda_array, double_fcs, single_fcs)


def build_results(peaks) -> pd.DataFrame:
    """

    :param peaks:
    :return:
    """
    indices, contacts_array, lambda_array, pvals, padjs, peak_info = peaks
    region_names = ['donut', 'horizontal', 'vertical', 'lower_left']
    num_region = len(region_names)
    col_names = ['i', 'j', 'ob'] \
                + ['ex_' + region for region in region_names] \
                + ['pval_' + region for region in region_names] \
                + ['padj_' + region for region in region_names] \
                + ['center_i', 'center_j', 'radius']

    dtypes = [np.int] * 3 + [np.float] * (len(col_names) - 3)
    peaks = np.zeros(shape=contacts_array.size,
                     dtype=[(col_name, dtype) for col_name, dtype in zip(col_names, dtypes)])
    fields_name = list(peaks.dtype.names)
    peaks['i'], peaks['j'], peaks['ob'] = indices[0], indices[1], contacts_array
    peaks[fields_name[3: 3 + num_region]] = list(zip(*lambda_array))
    peaks[fields_name[3 + num_region: 3 + 2 * num_region]] = list(zip(*pvals))
    peaks[fields_name[3 + 2 * num_region: 3 + 3 * num_region]] = list(zip(*padjs))
    peaks[fields_name[-3:]] = list(zip(*peak_info))

    return pd.DataFrame(peaks)


def calculate_lambda(expected,
                     observed,
                     row_factors,
                     col_factors,
                     kernels,
                     band_width,
                     outer_radius):
    """

    :param expected:
    :param observed:
    :param row_factors:
    :param col_factors:
    :param kernels:
    :param band_width:
    :param outer_radius:
    :return:
    """
    x, y = observed.nonzero()
    dis = y - x
    mask = (dis <= (band_width - 2 * outer_radius)) \
           & (x < (band_width - outer_radius)) \
           & (dis >= 2) \
           & (x >= outer_radius)
    x, y = x[mask], y[mask]

    ratio_array = np.full((len(kernels), x.size), 0, dtype=np.float)
    for index, kernel in enumerate(kernels):
        ob_sum = ndimage.convolve(observed, kernel)
        ex_sum = ndimage.convolve(expected, kernel)
        ratio_array[index] = (ob_sum / ex_sum)[(x, y)]

    lambda_array = ratio_array \
                   * expected[(x, y)] \
                   * row_factors[x] \
                   * col_factors[y]

    return (x, y), lambda_array


@ray.remote
def calculate_chunk(expected_fetcher,
                    observed_fetcher,
                    factors_fetcher,
                    chunk,
                    kernels,
                    band_width,
                    outer_radius):
    """

    :param expected_fetcher:
    :param observed_fetcher:
    :param factors_fetcher:
    :param chunk:
    :param kernels:
    :param band_width:
    :param outer_radius:
    :return:
    """
    key, slices = chunk
    observed = observed_fetcher(key, slices)
    expected = expected_fetcher(key, slices)
    observed[np.isnan(observed)] = 0
    expected[np.isnan(expected)] = 0
    row_factors, col_factors = factors_fetcher(key, slices)
    row_factors = 1 / row_factors
    col_factors = 1 / col_factors

    indices, lambda_array = calculate_lambda(expected,
                                             observed,
                                             row_factors,
                                             col_factors,
                                             kernels,
                                             band_width,
                                             outer_radius)

    if indices[0].size == 0:
        return (np.array([]), np.array([])), np.array([]), np.array([])
    else:
        contacts_array = observed[indices] \
                         * row_factors[indices[0]] \
                         * col_factors[indices[1]]

        true_indices = (indices[0] + slices[0].start, indices[1] + slices[1].start)

        return true_indices, lambda_array, contacts_array


@ray.remote
@suppress_warning
def find_peaks(backgrounds,
               test_fn,
               cluster_fn,
               filter_fn) -> tuple:
    """

    :param backgrounds:
    :param test_fn:
    :param cluster_fn:
    :param filter_fn:
    :return:
    """
    # load data from ray store
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


@suppress_warning(warning_msg=ResourceWarning)
def hiccups(expected_fetcher: callable,
            observed_fetcher: callable,
            factors_fetcher: callable,
            chunks: Iterable,
            kernels: tuple,
            num_cpus: int = 20,
            max_dis: int = 1000000,
            outer_radius: int = 5,
            resolution: int = 10000,
            method: str = 'fdr_bh',
            fdrs: tuple = (0.01, 0.01, 0.01, 0.01),
            sigs: tuple = (0.01, 0.01, 0.01, 0.01),
            single_fcs: tuple = (1.7, 1.5, 1.5, 1.75),
            double_fcs: tuple = (2, 0, 0, 2),
            ignore_single_gap: bool = True,
            bin_index: bool = True):
    """

    :param expected_fetcher:
    :param observed_fetcher:
    :param factors_fetcher:
    :param chunks:
    :param kernels:
    :param num_cpus:
    :param max_dis:
    :param outer_radius:
    :param resolution:
    :param method:
    :param fdrs:
    :param sigs:
    :param single_fcs:
    :param double_fcs:
    :param ignore_single_gap:
    :param bin_index:
    :return:
    """
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    initial_dis = max(20000 // resolution, 1)
    backgrounds_dict = defaultdict(list)
    for chunk in chunks:
        key = chunk[0]
        backgrounds_dict[key].append(calculate_chunk.remote(expected_fetcher,
                                                            observed_fetcher,
                                                            factors_fetcher,
                                                            chunk,
                                                            kernels,
                                                            max_dis // resolution,
                                                            outer_radius))

    test_fn = partial(multiple_test,
                      fdrs=fdrs,
                      sigs=sigs,
                      method=method)

    cluster_fn = partial(cluster,
                         initial_dis=initial_dis)

    filter_fn = partial(additional_filtering,
                        double_fcs=double_fcs,
                        single_fcs=single_fcs,
                        ignore_single_gap=ignore_single_gap)

    peaks_dict = OrderedDict()
    full_slices = (slice(0, None), slice(0, None))
    for key, backgounds in backgrounds_dict.items():
        _filter_fn = partial(filter_fn, factors=factors_fetcher(key, full_slices)[0])
        peaks_dict[key] = find_peaks.remote(backgounds,
                                            test_fn=test_fn,
                                            cluster_fn=cluster_fn,
                                            filter_fn=_filter_fn)
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

    ray.shutdown()

    return peaks_df


if __name__ == "__main__":
    pass
