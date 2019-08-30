"""
Tools for Loop-detection analysis.
"""
# TODO implement APA.
# TODO Refine hiccupps method(clustering step). Peaks are not as good as that produced in the original paper(Rao 2014).
# TODO implement cloops algorithm.

from collections import OrderedDict
from collections import defaultdict
from functools import partial
from typing import Tuple, Iterable, Callable

import numpy as np
import pandas as pd
import ray
from scipy import ndimage, stats
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


def fetch_kernels(p: int, w: int) -> tuple:
    """Return kernels of four regions: donut region, vertical, horizontal, lower_left region.

    :param p: int. radius of center square.
    :param w: int. radius of outer square.
    :return: tuple.
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

    return tuple(region_to_kernel(donut, vertical, horizontal, lower_left))


def cluster(indices: np.ndarray,
            contacts: np.ndarray,
            lambda_array: np.ndarray,
            initial_dis: float) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param indices:
    :param contacts:
    :param lambda_array:
    :param initial_dis:
    :return:
    """
    dbscan = DBSCAN(max(1.5, initial_dis), 2)
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


def multiple_test(indices: np.ndarray,
                  contact_array: np.ndarray,
                  lambda_array: np.ndarray,
                  fdrs: Tuple[float],
                  sigs: Tuple[float],
                  method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conduct poisson test on each pixel and multiple test correction for all tests.

    :param indices:
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


def additional_filtering(peaks: tuple,
                         gap_mask: np.ndarray,
                         sum_qvalue: float = 0.02,
                         fold_changes: tuple = (2, 1.5, 1.5, 2),
                         ignore_single_gap: bool = True,
                         singlet_only: bool = False):
    """Post-filtering peaks after filtered by mulitple test and megred by clustering:
        1. Remove peaks close to gap region(bad bins).\n
        2. Remove singlet peaks with sums of four padjs lesser than 'sum_qvalues' .\n
        3. Remove peaks with fold changes less than a given threshold in either one of four regions.\n
        4. Retain peaks with fold changes over a given threshold in either one of four regions.\n

    :param peaks: Tuple[np.ndarray, np.ndarray]. Tuple contains corrdinates of all peak.
    :param gap_mask: np.ndarray. Bad bins should be marked with True.
    :param sum_qvalue: float. Threshold for filtering those peaks with sum padjs of four regions lesser than a certain value.
    :param fold_changes: tuple. Padjs threshold for each region. Valid peak's padjs should pass all four fold-change thresholds.
    :param singlet_only:
    :param ignore_single_gap: bool. If ignore small gaps when filtering peaks close to gap regions.
    :return: np.ndarray. Boolean array of valid peaks passed all filtering.
    """

    def valid_mask(shapes: np.ndarray) -> np.ndarray:
        if singlet_only:
            return ~np.all(shapes == 0, axis=0)
        else:
            return np.full(shapes.shape[1], False)

    def enrich_mask(contact_array: np.ndarray,
                    lambda_array: np.ndarray,
                    enrich_ratio: np.ndarray) -> np.ndarray:
        """Return mask of valid peaks passed the enrichment fold changes filtering."""
        fc_mask = np.all(contact_array
                         >= lambda_array * np.array(fold_changes)[:, None], axis=0)
        ec_mask = enrich_ratio > 0.4

        return fc_mask & ec_mask

    def away_gap_mask(indices) -> np.ndarray:
        """Return mask of valid peaks away from gap regions."""
        gap_region = set(np.where(extend_gap(gap_mask, 1, ignore_single_gap))[0])

        return ~np.array([i in gap_region or j in gap_region
                          for i, j in zip(*indices)])

    def qvalue_mask(padjs: np.ndarray):
        """Return mask of valid peaks passed the sum-qvalue threshold."""
        return padjs.sum(axis=0) < sum_qvalue

    indices, contact_array, lambda_array, enrich_ratio, pvals, padjs, shapes = peaks

    return valid_mask(shapes) | (qvalue_mask(padjs)
                                 & away_gap_mask(indices)
                                 & enrich_mask(contact_array, lambda_array, enrich_ratio))


def build_results(peaks: tuple, resolution=None) -> pd.DataFrame:
    """Aggregate peak-infos into a pd.DataFrame object.

    :param resolution:
    :param peaks: tuple. Tuple containing all infos related to peaks returned by find_peaks.
    :return: pd.DataFrame.
    """
    region_names = ['donut', 'horizontal', 'vertical', 'lower_left']
    num_region = len(region_names)
    col_names = (['i', 'j', 'ob']
                 + ['ex_' + region for region in region_names]
                 + ['pval_' + region for region in region_names]
                 + ['padj_' + region for region in region_names]
                 + ['enrich_ratio', 'width', 'height'])
    dtypes = [np.int] * 3 + [np.float] * (len(col_names) - 3)

    if peaks:
        indices, contacts_array, lambda_array, enrich_ratio, pvals, padjs, shape = peaks
        peaks = np.zeros(shape=contacts_array.size,
                         dtype=[(col_name, dtype) for col_name, dtype in zip(col_names, dtypes)])
        fields_name = list(peaks.dtype.names)
        peaks['i'], peaks['j'], peaks['ob'] = indices[0], indices[1], contacts_array
        peaks[fields_name[3: 3 + num_region]] = list(zip(*lambda_array))
        peaks[fields_name[3 + num_region: 3 + 2 * num_region]] = list(zip(*pvals))
        peaks[fields_name[3 + 2 * num_region: 3 + 3 * num_region]] = list(zip(*padjs))
        peaks[fields_name[-3]] = enrich_ratio
        peaks[fields_name[-2:]] = list(zip(*shape))
        if resolution is not None:
            peaks[['i', 'j', 'width', 'height']] *= resolution
        return pd.DataFrame(peaks)
    else:
        return pd.DataFrame(columns=col_names).astype(
            {name: t for name, t in zip(col_names, dtypes)}
        )


@suppress_warning
def calculate_lambda(observed: np.ndarray,
                     expected: np.ndarray,
                     valid_mat: np.ndarray,
                     row_factors: np.ndarray,
                     col_factors: np.ndarray,
                     kernels: Tuple[np.ndarray],
                     band_width: int,
                     inner_radius: int,
                     outer_radius: int,
                     ignore_diags: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate lambda values(background) for each pixel in regions sepcified in kernels.

    :param valid_mat:
    :param inner_radius:
    :param expected: np.ndarray. 2-d ndarray represents expeceted(normed) matrix.
    :param observed: np.ndarray. 2-d ndarray represents observed(normed) matrix.
    :param row_factors: np.ndarray. 1-d ndarray represents ICE factors of each row in expected.
    :param col_factors: np.ndarray. 1-d ndarray represents ICE factors of each column in expected.
    :param kernels: Tuple[np.ndarray]. Each array(mask) represents a certain region that is used for computing
    lambda(background) by summing all values within this region for each pixel.
    :param band_width: int. Width of the band region.
    :param outer_radius: int. The maximum radius among all kernels.
    :param ignore_diags: int. Number of diagonals to ignore. Pixles within this region will not be counted in available contacts.
    :return: Tuple[np.ndarray, np.ndarray]. The first ndarray contains indices of all available pixels, and the second ndarray
    contains the corresponding lambdas in all regions specified in kernels.
    """
    if ignore_diags is None:
        ignore_diags = 2 * outer_radius
    x, y = observed.nonzero()
    dis = y - x
    mask = ((dis <= (band_width - 2 * outer_radius))
            & (x < (observed.shape[0] - outer_radius))
            & (dis >= ignore_diags)
            & (x >= outer_radius))
    x, y = x[mask], y[mask]
    num_kernels = len(kernels)

    if x.size == 0:
        return np.empty((2, 0)), np.empty((num_kernels, 0)), np.empty(0)
    else:

        ratio_array = np.full((num_kernels, x.size), 0, dtype=np.float)
        oe_matrix = observed / expected
        for index, kernel in enumerate(kernels):
            # ob_sum = ndimage.convolve(observed, kernel)
            # ex_sum = ndimage.convolve(expected, kernel)
            # ratio_array[index] = (ob_sum / ex_sum)[(x, y)]

            # Another option
            # counts = ndimage.convolve(valid_mat, kernel)
            ratio = ndimage.convolve(oe_matrix, kernel) / kernel.sum()
            ratio_array[index] = ratio[x, y]

        lambda_array = (ratio_array
                        * expected[x, y]
                        * row_factors[x]
                        * col_factors[y])

        inner_len = 2 * inner_radius + 1
        outer_len = 2 * outer_radius + 1
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

        return np.vstack((x, y)), lambda_array, enrich_ratio


def calculate_chunk(observed_fetcher: Callable[[str, tuple], np.ndarray],
                    expected_fetcher: Callable[[str, tuple], np.ndarray],
                    factors_fetcher: Callable[[str, tuple], tuple],
                    chunk: Tuple[str, Tuple[slice, slice]],
                    kernels: Tuple[np.ndarray],
                    band_width: int,
                    inner_radius: int,
                    outer_radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For a given chunk, calculate lambda values and contact(true counts) values of each pixel in regions specified in kernels.

    :param inner_radius:
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
    zero_region = observed == 0
    expected[zero_region] = 0
    row_factors, col_factors = factors_fetcher(key, slices)
    row_factors, col_factors = 1 / row_factors, 1 / col_factors
    indices, lambda_array, enrich_ratio = calculate_lambda(
        observed=observed,
        expected=expected,
        valid_mat=(~zero_region).astype(np.int16),
        row_factors=row_factors,
        col_factors=col_factors,
        kernels=kernels,
        band_width=band_width,
        inner_radius=inner_radius,
        outer_radius=outer_radius
    )

    if indices[0].size == 0:
        return (indices,
                np.empty(0),
                lambda_array,
                enrich_ratio)
    else:
        contacts_array = (observed[(indices[0], indices[1])]
                          * row_factors[indices[0]]
                          * col_factors[indices[1]])

        nan_mask = np.isnan(lambda_array)
        lambda_array[nan_mask] = 0
        non_nan_mask = ~(np.all(nan_mask, axis=0) | np.isnan(contacts_array))
        start_indice = np.array([[slices[0].start],
                                 [slices[1].start]])
        # Another option
        # enrich_mask = np.all(contacts_array > lambda_array * np.array(fillter_fcs)[:, None], axis=0)
        # return (indices[:, enrich_mask] + start_indice,
        #         contacts_array[enrich_mask],
        #         lambda_array[:, enrich_mask])

        return (indices[:, non_nan_mask] + start_indice,
                contacts_array[non_nan_mask],
                lambda_array[:, non_nan_mask],
                enrich_ratio[non_nan_mask])


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

    def append_array(targets_list, items: Tuple[np.ndarray]):
        if isinstance(items, ray.ObjectID):
            items = ray.get(items)

        if items[0].size != 0:
            for i, item in enumerate(items):
                targets_list[i].append(item)

    indices, contacts_array, lambda_array, enrich_ratio = [], [], [], []

    for background in backgrounds:
        append_array((indices,
                      contacts_array,
                      lambda_array,
                      enrich_ratio), background)

    indices = np.concatenate(indices, axis=1)
    contacts_array = np.concatenate(contacts_array)
    lambda_array = np.concatenate(lambda_array, axis=1)
    enrich_ratio = np.concatenate(enrich_ratio)

    # Multiple test. Filtering insignificant point after calculating padj using fdr_bh multiple test method.
    pvals, padjs, rejects = test_fn(indices, contacts_array, lambda_array)
    peaks = (indices, contacts_array, lambda_array, enrich_ratio, pvals, padjs)

    reject = np.all(rejects, axis=0)
    peaks = tuple(mask_array(reject, *peaks))

    # Apply greedy clustering to merge  points into confidant peaks.
    peak_indexs, shapes = cluster_fn(peaks[0], peaks[1], peaks[2])
    peaks = (*tuple(index_array(peak_indexs, *peaks)), shapes)

    # Filter by gap_region, fold changes(enrichment) and singlet peak's sum-qvalue.
    valid_mask = filter_fn(peaks)
    peaks = tuple(mask_array(valid_mask, *peaks))

    return peaks


@suppress_warning(warning_msg=ResourceWarning)
def hiccups(observed_fetcher: Callable[[str, tuple], np.ndarray],
            expected_fetcher: Callable[[str, tuple], np.ndarray],
            factors_fetcher: Callable[[str, tuple], np.ndarray],
            chunks: Iterable[Tuple[str, Tuple[slice, slice]]],
            kernels: Tuple[np.ndarray],
            inner_radius: int = 2,
            outer_radius: int = 5,
            num_cpus: int = 20,
            max_dis: int = 5000000,
            resolution: int = 10000,
            method: str = 'fdr_bh',
            fdrs: tuple = (0.1, 0.1, 0.1, 0.1),
            sigs: tuple = (0.1, 0.1, 0.1, 0.1),
            fold_changes: tuple = (1.7, 1.5, 1.5, 1.75),
            ignore_single_gap: bool = True,
            bin_index: bool = True) -> pd.DataFrame:
    """Call peaks using hiccups algorithm.

    :param outer_radius:
    :param inner_radius:
    :param bin_index:
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
    :param fold_changes: tuple. Padjs threshold for each region. Valid peak's padjs should pass all four fold-hange thresholds.
    :param ignore_single_gap: bool. If ignore small gaps when filtering peaks close to gap regions.
    :return: pd.DataFrame.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
    _calculate_chunk = ray.remote(calculate_chunk)
    _find_peaks = ray.remote(find_peaks)

    initial_dis = max(20000 // resolution, 1)
    initial_dis += initial_dis / 4
    backgrounds_dict = defaultdict(list)
    for chunk in chunks:
        key = chunk[0]
        backgrounds_dict[key].append(
            _calculate_chunk.remote(
                observed_fetcher=observed_fetcher,
                expected_fetcher=expected_fetcher,
                factors_fetcher=factors_fetcher,
                chunk=chunk,
                kernels=kernels,
                band_width=max_dis // resolution,
                inner_radius=inner_radius,
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
        fold_changes=fold_changes,
        ignore_single_gap=ignore_single_gap,
        singlet_only=False
    )

    peaks_dict = OrderedDict()
    full_slices = (slice(0, None), slice(0, None))
    for key, backgrounds in backgrounds_dict.items():
        gap_mask = np.isnan(factors_fetcher(key, full_slices)[0])
        _filter_fn = partial(filter_fn, gap_mask=gap_mask)
        peaks_dict[key] = _find_peaks.remote(
            backgrounds=backgrounds,
            test_fn=test_fn,
            cluster_fn=cluster_fn,
            filter_fn=_filter_fn
        )

    chroms_list = []
    for key in peaks_dict.keys():
        peaks_dict[key] = build_results(
            ray.get(peaks_dict[key]),
            resolution=resolution if not bin_index else None
        )
        chroms_list.extend(key for i in range(peaks_dict[key].shape[0]))

    peaks_df = pd.concat([peaks_dict[chrom] for chrom in peaks_dict.keys()])
    peaks_df.insert(0, 'chrom', np.array(chroms_list))

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


def detect_peaks2d(img: np.ndarray):
    """Idea:
    1: Use maximum filter to find local maximum for each region. -> points.
    2: Use percentage filter to find enriched regions. -> regions.
    3: Design a statitical test method for the assessment pf  randomness and enrichment. -> filter points.
    """
    pass


if __name__ == "__main__":
    pass
