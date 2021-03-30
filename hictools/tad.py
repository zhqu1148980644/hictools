"""Tools for topological associated domain analysis."""

from typing import Union, Tuple

import numpy as np
from scipy import sparse

from .utils.numtools import mask_array, get_diag, cumsum2d
from .utils.utils import suppress_warning


@suppress_warning
def di_score(matrix: Union[np.ndarray, sparse.csr_matrix],
             window_size: int = 20,
             ignore_diags: int = 3,
             method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
    """Compute directionality index of a given 2d ndarray.\n
    For each bin in the main digonal, directionality index is calculated based on two arrays with length of windowsize: \n
    The upwards(upstream) vertical array start from this bin and the eastwards(downstream) horizontal array start from this bin.\n
    See function listed in 'DI_METHOD_MAP' for detailed description.

    :param matrix: np.ndarray/sparse.csr_matrix. The input matrix to calculate di score.
    :param window_size: int. length of upstream array and downstream array.
    :param ignore_diags: int. The number of diagonals to ignore.
    :param method: str. Method for computing directionality index. 'standard' and 'adptive' are supported by now.
    :param fetch_window: bool. If set to True, return np.hstack([contacts_up. contacts_down])
    :return: np.ndarray. Rerturn directionality index array if 'fetch_window' is False else return array of up/down stream contacts.
    """

    def standard_di(up, down):
        """Compute directionality index described in:\n
        Jesse R.Dixon 2012. Topological domains in mammalian genomes identified by analysis of chromatin interactions.
        """
        up = up.sum(axis=1)
        down = down.sum(axis=1)
        expected = (up + down) / 2.0
        return (np.sign(down - up) *
                    ((up - expected) ** 2 + (down - expected) ** 2)
                    / expected)

    def adap_di(up, down):
        """Compute directionality index described in:\n
        Xiao-Tao Wang 2017.HiTAD: Detecting the structural and functional hierarchies of topologically associating
        domains from chromatin interactions.
        """
        window_size = up.shape[1]
        mean_up = up.mean(axis=1)
        mean_down = down.mean(axis=1)
        var_up = np.square(up - mean_up[:, None]).sum(axis=1)
        var_down = np.square(down - mean_down[:, None]).sum(axis=1)
        return ((mean_down - mean_up) /
                    np.sqrt((var_up + var_down) / (window_size * (window_size - 1))))

    method_map = {
        'standard': standard_di,
        'adaptive': adap_di
    }
    chrom_len = matrix.shape[0]
    x, y = matrix.nonzero()
    dis = y - x
    if isinstance(window_size, int):
        max_len = ignore_diags + window_size
        available = ((dis >= ignore_diags)
                     & (dis < max_len)
                     & (x >= max_len - 1)
                     & (y <= chrom_len - max_len))
        x, y = mask_array(available, x, y)
        values = np.array(matrix[x, y]).ravel()
        x, y, values = mask_array(~np.isnan(values), x, y, values)
        dis = y - x

        contacts_up = np.zeros((chrom_len, window_size), dtype=matrix.dtype)
        contacts_down = np.zeros((chrom_len, window_size), dtype=matrix.dtype)
        for shift in range(ignore_diags, max_len):
            window_pos = shift - ignore_diags
            mask = dis == shift
            tmp_x, tmp_values = mask_array(mask, x, values)
            contacts_down[tmp_x, window_pos] = tmp_values
            contacts_up[tmp_x + shift, window_pos] = tmp_values
        contacts_up[:max_len, :] = 0
        contacts_down[:max_len, :] = 0

    elif isinstance(window_size, np.ndarray) \
            and (window_size.size == chrom_len):
        # TODO(zhongquan789@gmail.com) Suits for multi-windowsize(used for di in tadlib).
        contacts_up = None
        contacts_down = None

    else:
        raise ValueError(
            'window_size should either be an integer or a np.ndarray.')

    return method_map[method](contacts_up, contacts_down), np.hstack([contacts_up, contacts_down])


@suppress_warning
def insu_score(matrix: Union[np.ndarray, sparse.csr_matrix],
               window_size: int = 20,
               ignore_diags: int = 1,
               normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate insulation score of a given 2d ndarray(chromosome) described in:\n
    Emily Crane 2015. Condensin-driven remodelling of X chromosome topology during dosage compensation.\n
    :param matrix: np.ndarray/scipy.sparse.csr_matrix. Interaction matrix representing a hic contacts.
    :param window_size: int. Diameter of square in which contacts are summed along the diagonal.
    :param ignore_diags: int. Number of diagonal to ignore, This values should be >= 1 which means ignore main diagonal.
    :param normalize: bool. If normalize the insulation score with log2 ratio of insu_score and mean insu_score.
    :param count: bool. If return number of valid contacts in each square region for each bin respectively.
    :return: np.ndarray. Return (chrom_len,) array if count set to False otherwise return (2, chrom_len) array.
    """
    chrom_len = matrix.shape[0]

    if isinstance(matrix, sparse.csr_matrix):
        x, y = matrix.nonzero()
        dis = y - x
        available = (dis >= ignore_diags) & (dis <= 2 * window_size)
        x, y = mask_array(available, x, y)
        values = np.array(matrix[x, y]).ravel()
        x, y, values = mask_array(~np.isnan(values), x, y, values)
        dis = y - x

        insu = np.zeros(chrom_len, dtype=matrix.dtype)
        insu[:window_size] = np.nan
        insu[chrom_len - window_size:] = np.nan
        counts = np.zeros(chrom_len, dtype=np.int)
        diag_dict: dict = {}
        for i in range(window_size):
            for j in range(window_size):
                _dis = (abs(j - i) + (min(j, i) + 1) * 2)
                if diag_dict.get(_dis) is None:
                    mask = dis == (abs(j - i) + (min(j, i) + 1) * 2)
                    tmp_x, tmp_value = mask_array(mask, x, values)
                    diag_dict[_dis] = (tmp_x, tmp_value)
                else:
                    tmp_x, tmp_value = diag_dict[_dis]
                x_index = tmp_x + j + 1
                insu[x_index] += tmp_value
                counts[x_index] += 1
        counts[:window_size] = 0
        counts[chrom_len - window_size:] = 0
        insu /= counts
    # Store counts only when parameter count is set to True.
    elif isinstance(matrix, np.ndarray):
        insu = np.full(chrom_len, np.nan, dtype=matrix.dtype)
        counts = np.zeros(chrom_len, dtype=np.int)
        diamond_mask = np.full((window_size, window_size), True)
        diamond_mask = np.triu(diamond_mask, ignore_diags - window_size)
        for row in range(window_size, chrom_len - window_size):
            sub_mat = matrix[row - window_size: row, row +
                                                     1: row + window_size + 1][diamond_mask]
            insu[row] = np.nanmean(sub_mat)
            counts[row] = np.sum(~np.isnan(sub_mat))

    else:
        raise ValueError(
            "Only support for scipy.sparse.csr_matrix and np.ndarray")

    if normalize:
        insu = np.log2(insu / np.nanmean(insu))

    return insu, counts


@suppress_warning
def rinsu_score(matrix: np.ndarray, width_range: slice = slice(4, 15), add: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Relative insulation score.

    Parameters
    ----------
    matrix
    width_range
    add

    Returns
    -------

    """

    def fetch_sum(r1, r2, c1, c2):
        return sumob[r2][c2] - sumob[r1][c2] - sumob[r2][c1] + sumob[r1][c1]

    ob = matrix.copy()
    get_diag(ob, 0)[:] = 0

    shape, n = ob.shape, ob.shape[0]
    sumob = cumsum2d(ob)
    st, ed = width_range.start, width_range.stop
    ris = np.zeros((ed - st, n))

    for i, w in enumerate(range(st, ed)):
        for x in range(n):
            if x - w < 0 or x + w > n:
                continue
            suma = fetch_sum(x - w, x, x - w, x)
            sumb = fetch_sum(x, x + w, x, x + w)
            sumc = fetch_sum(x - w, x, x, x + w)
            if add:
                ris[i][x] = (suma + sumb - sumc) / (suma + sumb + sumc)
            else:
                ris[i][x] = (suma + sumb - sumc) / (suma + sumb)

    num_nonzero = (ris != 0).sum(axis=0)
    num_nonzero[num_nonzero == 0] = 1

    return ris.sum(axis=0) / num_nonzero, ris


pass
