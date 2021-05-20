"""Tools for topological associated domain analysis."""

from copy import copy
from functools import total_ordering
from typing import Union, Tuple, List, Optional

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


@total_ordering
class Score(object):
    def __init__(self, st=None, ed=None, left=None, right=None, is_tad=False, parent=None, **kwargs):
        self.st = st
        self.ed = ed
        self.left: Optional[Score] = left
        self.right: Optional[Score] = right
        self.is_tad: bool = is_tad
        self.parent: Optional[Score] = parent
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return f"{self.st}-{self.ed}"

    def __repr__(self):
        return f"{self.st}-{self.ed}"

    @property
    def inf(self) -> "Score":
        raise NotImplementedError

    @property
    def zero(self) -> "Score":
        raise NotImplementedError

    def __call__(self, st, ed, left=None, right=None, **kwargs) -> 'Score':
        raise NotImplementedError

    def __add__(self, other) -> 'Score':
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def extract_tads(self: 'Score'):
        from collections import defaultdict
        res = defaultdict(list)

        def traverse(score: Optional[Score], level=0, parent: Optional[Score] = None):
            if not score:
                return

            if score.is_tad:
                score.parent = parent
                res[level].append(score)
                parent = score

            next_level = level + 1 if score.is_tad else level
            for child in (score.left, score.right):
                traverse(child, next_level, parent=parent)

        traverse(self)

        return res


class TadScore(Score):
    def __init__(self, info: dict, score=float("-inf"), **kwargs):
        super().__init__(**kwargs)
        self.info: dict = info
        self.score = score

    def __call__(self, st, ed, left=None, right=None, **kwargs):
        # return self.__call_oe_neighbor_diff(st, ed, left, right, **kwargs)

        return self.__call_raw_diamond_diff(st, ed, left, right, **kwargs)

    def __add__(self: "TadScore", s2: "TadScore") -> "Score":
        st = self.st
        ed = s2.ed

        return self(st, ed, left=self, right=s2)

    def __eq__(self: "TadScore", s2: "TadScore") -> bool:
        return self.score == s2.score

    def __lt__(self: "TadScore", s2: "TadScore"):
        return self.score < s2.score

    @property
    def inf(self):
        new = copy(self)
        new.score = float("-inf")
        return new

    @property
    def zero(self):
        new = copy(self)
        new.score = 0
        return new

    def __call_oe_neighbor_diff(self, st, ed, left: "TadScore" = None, right: "TadScore" = None,
                                **kwargs) -> "TadScore":
        sumoe = self.info['sumoe']
        s_sum, *_ = self.sum(st, ed, st, ed, sumoe)
        c_sum = (ed - st) * (ed - st)

        if ed - st < 6 or ed - st > 400:
            s_sum_back = s_sum
            c_sum_back = c_sum
            inside_mean = outside_mean = None
            is_tad = False
            mean_ratio = 1
            score = 0
        else:
            # calculate inside_mean/mean_ratio excluded inner tads
            outside_mean = self.get_outside_mean(st, ed, sumoe)
            if left or right:
                assert left and right
                s_sum -= left.s_sum + right.s_sum
                c_sum -= left.c_sum + right.c_sum

            s_sum_back = s_sum
            c_sum_back = c_sum
            inside_mean = s_sum / c_sum

            mean_ratio = inside_mean / (max(outside_mean[0], outside_mean[1]) * 1)
            is_tad = mean_ratio > 1
            #             score = max(mean_ratio - 1, 0)
            #             score = inside_mean - (max(outside_mean[0], outside_mean[1]))
            score = mean_ratio - 1

        # add up score of inner score
        if left or right:
            assert left and right
            score += left.score + right.score

        # recalculate s_sum and c_sum
        if is_tad:
            s_sum, *_ = self.sum(st, ed, st, ed, sumoe)
            c_sum = (ed - st) * (ed - st)
        elif left or right:
            assert left and right
            s_sum = left.s_sum + right.s_sum
            c_sum = left.c_sum + right.c_sum
        else:
            s_sum = 0
            c_sum = 0
        return TadScore(info=self.info, st=st, ed=ed, score=score,
                        is_tad=is_tad, left=left, right=right, mean_ratio=mean_ratio,
                        s_sum=s_sum, c_sum=c_sum, inside_mean=inside_mean, outside_mean=outside_mean,
                        s_sum_back=s_sum_back, c_sum_back=c_sum_back)

    def __call_raw_diamond_diff(self, st, ed, left=None, right=None, **kwargs):
        score = 0
        if ed - st < 4 or ed - st > 400:
            is_tad = False
            cur_score = 0
            un_score = 0
            outside_left = inside = outside_right = left_ratio = right_ratio = None
        else:
            outside_left, inside, outside_right = self.diamond(st, ed, self.info['ob'])
            left_ratio = (inside - outside_left) / (inside + outside_left)
            right_ratio = (inside - outside_right) / (inside + outside_right)
            left_mean = np.nanmean(left_ratio)
            right_mean = np.nanmean(right_ratio)
            un_score = np.mean([left_mean, right_mean])
            score = cur_score = un_score / (ed - st)
            is_tad = cur_score > 0 and left_mean > 0 and right_mean > 0
        if not is_tad:
            score = 0

        if left or right:
            assert left and right
            score += left.score + right.score

        return TadScore(info=self.info, is_tad=is_tad, st=st, ed=ed, score=score, cur_score=cur_score,
                        un_score=un_score,
                        inside=inside, left=left, right=right,
                        outside_left=outside_left, outside_right=outside_right,
                        left_ratio=left_ratio, right_ratio=right_ratio)

    def sum(self, x1, x2, y1, y2, ma: np.ndarray):
        x1 = max(0, x1)
        x2 = min(len(ma) - 1, x2)
        y1 = max(0, y1)
        y2 = min(len(ma) - 1, y2)
        size1 = x2 - x1
        size2 = y2 - y1
        try:
            s = ma[x2][y2] - ma[x1][y2] - ma[x2][y1] + ma[x1][y1]
        except Exception as e:
            print(f"Error in sum: {x1}-{x2}-{y1}-{y2} with error: {e}")
            raise e
        return s, (size1, size2)

    def mean(self, x1, x2, y1, y2, ma: np.ndarray):
        s, (size1, size2) = self.sum(x1, x2, y1, y2, ma)
        return s / (size1 * size2)

    def diamond(self, st, ed, ma: np.ndarray):
        w = ed - st
        hw = w // 2
        # inside
        inside = ma[st: st + hw, st + hw: ed]

        try:
            sst = max(st - hw, 0)
            out_left = ma[sst: st, st: st + (inside.shape[1])]
        except:
            out_left = inside
        try:
            eed = min(len(ma) - 1, ed + hw)
            out_right = ma[ed - (inside.shape[0]): ed, ed: eed]
        except:
            out_right = inside

        if out_left.shape != inside.shape:
            left = inside.shape[0] - out_left.shape[0]
            out_left = np.concatenate([inside[:left, :], out_left], axis=0)
        if out_right.shape != inside.shape:
            left = inside.shape[1] - out_right.shape[1]
            out_right = np.concatenate([inside[:, :left], out_right], axis=1)

        assert out_left.shape == inside.shape == out_right.shape

        return out_left, inside, out_right

    def get_outside_mean(self, st, ed, ma: np.ndarray):
        # width of neighbor
        w = (ed - st) // 2
        #         w = (ed - st)

        out_left = out_right = -1
        try:
            out_left = self.mean(max(0, st - w), st, st, ed, ma)
        except:
            out_left = 0
        try:
            out_right = self.mean(st, ed, ed, min(ed + w, len(ma) - 1), ma)
        except:
            out_right = 0

        return out_left, out_right


def solve(borders: List[int], score: Score):
    assert isinstance(score, Score)
    n = len(borders)
    dp = [[score.inf for i in range(n)] for i in range(n)]

    for ed in range(1, n):
        for st in range(ed - 1, -1, -1):
            pos_st = borders[st]
            pos_ed = borders[ed]
            max_sc = score(pos_st, pos_ed)
            for mid in range(ed - 1, st, -1):
                dpl = dp[st][mid]
                dpr = dp[mid][ed]
                sum_sc = dpl + dpr
                if sum_sc > max_sc:
                    max_sc = sum_sc
            dp[st][ed] = max_sc

    return dp


def call_tads(ob: np.ndarray):
    from scipy import ndimage, signal
    ob[~np.isfinite(ob)] = 0

    insu = insu_score(ob)
    gau_insu = -ndimage.gaussian_filter1d(insu, 3)
    peaks = signal.find_peaks_cwt(gau_insu, np.arange(2, 5))
    borders = [0] + list(peaks) + [len(insu)]

    info = {
        'ob': ob
    }
    score = TadScore(info=info)
    dp = solve(borders, score)
    return dp[0][-1].extract_tads()
