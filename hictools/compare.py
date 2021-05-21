import gc
from functools import partial
from math import floor
from typing import Union

import numpy as np
from scipy import stats, sparse
from scipy.ndimage.filters import gaussian_filter
from statsmodels.stats import multitest

from .utils.numtools import apply_along_diags


def selfish(mat1: Union[np.ndarray, sparse.spmatrix],
            mat2: Union[np.ndarray, sparse.spmatrix],
            sigma: float = 1.6,
            k_num: int = 12,
            max_bin: int = None):
    def kernel_width(_sigma):
        return floor(4.0 * _sigma + 0.5)

    if mat1.shape != mat2.shape:
        raise ValueError('Unequal shape.')

    sigmas = [sigma * (2 ** (i / (k_num - 2))) for i in range(1, k_num)]
    max_bin = mat1.shape[0] if max_bin is None else max_bin
    nonzero_fn = partial(np.not_equal, 0.)
    mat_len = mat1.shape[0]
    ranges = range(max_bin)
    diags_gen = zip(
        apply_along_diags(stats.zscore, mat1, ranges, nonzero_fn),
        apply_along_diags(stats.zscore, mat2, ranges, nonzero_fn)
    )

    diags, masks = [], []
    for (zscore1, mask1), (zscore2, mask2) in diags_gen:
        diags.append(zscore1 - zscore2)
        masks.append(mask1 & mask2)

    diff_ma = sparse.diags(diags, ranges, dtype=np.float32)
    indices = np.vstack(
        sparse.diags(masks, ranges, dtype=np.bool).nonzero()
    ).astype(np.int32)
    gc.collect()
    max_width = kernel_width(sigmas[-1])
    valid_mask = indices[0] > max_width
    valid_mask &= indices[1] < max_width
    valid_mask &= indices[0] < mat_len - max_width
    valid_mask &= indices[1] < mat_len - max_width
    indices = indices[:, valid_mask]

    pvals = np.full(indices[0].size, 1, dtype=np.float32)
    radius = np.full_like(pvals)
    pre_disk = gaussian_filter(diff_ma, sigmas[0])
    for sigma in sigmas[1:]:
        cur_disk = gaussian_filter(diff_ma, sigma)
        dev_diff = (cur_disk - pre_disk)[indices[0], indices[1]]
        mean, var, *_ = stats.norm.fit(dev_diff)
        cur_pvals = stats.norm.cdf(dev_diff, loc=mean, scale=var)
        r_mask = dev_diff >= 0.5
        cur_pvals[r_mask] = 1 - cur_pvals[r_mask]
        cur_pvals *= 2
        cur_pvals[~r_mask] *= -1
        sig_mask = np.abs(cur_pvals) < np.abs(pvals)
        pvals[sig_mask] = cur_pvals[sig_mask]
        radius[sig_mask] = kernel_width(sigma) + 1
    padjs = multitest.multipletests(np.abs(pvals), method='fdr_bh')
    padjs *= np.sign(pvals)

    return indices, np.vstack([padjs, pvals, radius])
