"""
Tools for topological associated domain analysis.
"""

from collections import namedtuple

import cooler
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from pomegranate import NormalDistribution, HiddenMarkovModel, GeneralMixtureModel, State

from utils import mask_array, remove_small_gap, suppress_warning, CPU_CORE

# background state
STATES = ('start', 'downstream', 'upstream', 'end')
# INIT_PROB = (0.4, 0.0, 0.3, 0.0, 0.0)
INIT_PROB = (1.0, 0.0, 0.0, 0.0)
# END_PROB = (0.0, 0.0, 0.3, 0.0, 0.4)
END_PROB = (0.0, 0.0, 0.0, 1.0)
INIT_TRANSITION = ((0.0, 1.0, 0.0, 0.0),
                   (0.0, 0.7, 0.3, 0.0),
                   (0.0, 0.0, 0.7, 0.3),
                   (1.0, 0.0, 0.0, 0.0))


def plot_tads(domains: list):
    for st, ed in domains:
        plt.plot([st] * (ed - st), np.arange(st, ed), color='yellow')
        plt.plot(np.arange(st, ed), [ed - 1] * (ed - st), color='yellow')
        plt.plot(np.arange(st, ed), np.arange(st, ed), color='yellow')


def standard_di(di_up, di_down):
    """

    :param di_up:
    :param di_down:
    :return:
    """
    di_up = di_up.sum(axis=1)
    di_down = di_down.sum(axis=1)
    expected = (di_up + di_down) / 2.0
    di_array = np.sign(di_down - di_up) * \
               ((di_up - expected) ** 2 + (di_down - expected) ** 2) / expected

    return di_array


def adap_di(di_up, di_down):
    """

    :param di_up:
    :param di_down:
    :return:
    """
    window_size = di_up.shape[1]
    mean_up = di_up.mean(axis=1)
    mean_down = di_down.mean(axis=1)
    var_up = np.square(di_up - mean_up[:, None]).sum(axis=1)
    var_down = np.square(di_down - mean_down[:, None]).sum(axis=1)
    di_array = (mean_down - mean_up) / \
               np.sqrt((var_up + var_down) / (window_size * (window_size - 1)))

    return di_array


DI_METHOD_MAP = {
    'standard': standard_di,
    'adaptive': adap_di
}


@suppress_warning
def di_score(matrix: sparse.csr_matrix,
             window_size: int = 10,
             ignore_diags: int = 3,
             method: str = 'standard',
             fetch_window: bool = False):
    """

    :param matrix:
    :param window_size:
    :param ignore_diags:
    :param method:
    :param fetch_window:
    :return:
    """
    chrom_len = matrix.shape[0]
    x, y = matrix.nonzero()
    dis = y - x
    if isinstance(window_size, int):
        max_len = ignore_diags + window_size
        available = (dis >= ignore_diags) \
                    & (dis < max_len) \
                    & (x >= max_len - 1) \
                    & (y <= chrom_len - max_len)
        x, y = mask_array(available, x, y)
        values = np.array(matrix[x, y]).ravel()
        x, y, values = mask_array(~np.isnan(values), x, y, values)
        dis = y - x

        di_up = np.zeros((chrom_len, window_size))
        di_down = np.zeros((chrom_len, window_size))
        for shift in range(ignore_diags, max_len):
            window_pos = shift - ignore_diags
            mask = dis == shift
            tmp_x, tmp_values = mask_array(mask, x, values)
            di_down[tmp_x, window_pos] = tmp_values
            di_up[tmp_x + shift, window_pos] = tmp_values
        di_up[:max_len, :] = 0
        di_down[:max_len, :] = 0

    elif isinstance(window_size, np.ndarray) \
            and (window_size.size == chrom_len):
        # TODO(zhongquan789@gmail.com) Suits for multi-windowsize
        di_up = None
        di_down = None

    else:
        raise ValueError('window_size should either be an integer or a np.ndarray.')

    if not fetch_window:
        return DI_METHOD_MAP[method](di_up, di_down)
    else:
        results = np.zeros(shape=chrom_len,
                           dtype=[('di_up', np.int), ('di_down', np.int)])
        results['di_up'] = di_up
        results['di_down'] = di_down
        return results


@suppress_warning
def insulation_score(matrix,
                     window_size: int = 50,
                     ignore_diags: int = 1,
                     normalize: bool = True,
                     count: bool = False) -> np.ndarray:
    """
    Calculate insulation score for each chromosome.
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

        insu_score = np.zeros(chrom_len, dtype=np.float)
        insu_score[:window_size] = np.nan
        insu_score[chrom_len - window_size:] = np.nan
        counts = np.zeros(chrom_len, dtype=np.int)
        diag_dict = {}
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
                insu_score[x_index] += tmp_value
                counts[x_index] += 1
        counts[:window_size] = 0
        counts[chrom_len - window_size:] = 0
        insu_score /= counts
    # Store counts only when parameter count is set to True.
    elif isinstance(matrix, np.ndarray):
        insu_score = np.full(chrom_len, np.nan, dtype=np.float)
        if count:
            counts = np.zeros(chrom_len, dtype=np.int)
        diamond_mask = np.full((window_size, window_size), True)
        diamond_mask = np.triu(diamond_mask, ignore_diags - window_size)
        for row in range(window_size, chrom_len - window_size):
            sub_mat = matrix[row - window_size: row, row + 1: row + window_size + 1][diamond_mask]
            insu_score[row] = np.nanmean(sub_mat)
            if count:
                counts[row] = np.sum(~np.isnan(sub_mat))

    else:
        raise ValueError("Only support for scipy.sparse.csr_matrix and np.ndarray")

    if normalize:
        insu_score = np.log2(insu_score / np.nanmean(insu_score))

    if count:
        results = np.zeros(shape=chrom_len,
                           dtype=[('insu_score', np.float), ('counts', np.int)])
        results['insu_score'] = insu_score
        results['counts'] = counts
        return results
    else:
        return insu_score


def boundary_strength(insu_score: np.ndarray) -> np.ndarray:
    # TODO(zhongquan789@gmail.com) implement boundary strength
    pass


def init_mean_fn(mix_num):
    mean_matrix = [[], [], [], []]
    for mix_i in range(mix_num):
        mean_matrix[0].append(mix_i * 7.5 / (mix_num - 1) + 2.5)
        mean_matrix[1].append(mix_i * 7.5 / (mix_num - 1))
        mean_matrix[2].append(-(mix_i * 7.5 / (mix_num - 1)))
        mean_matrix[3].append(-(mix_i * 7.5 / (mix_num - 1)) + 2.5)

    return mean_matrix


def init_var_fn(mix_num):
    var_matrix = [[], [], [], []]
    for state_i in range(len(STATES)):
        for mix_i in range(mix_num):
            var_matrix[state_i].append(7.5 / (mix_num - 1))

    return var_matrix


def ghmm_model(states_labels: tuple,
               transitions: tuple,
               init_prob: tuple,
               end_prob: tuple,
               means: list,
               vars: list) -> HiddenMarkovModel:
    """

    :param states_labels:
    :param transitions:
    :param init_prob:
    :param end_prob:
    :param means:
    :param vars:
    :return:
    """
    hmm_model = HiddenMarkovModel()

    mix_num = len(vars[0])
    states = []
    for state_i, state in enumerate(states_labels):
        mixture = []
        for mix_i in range(mix_num):
            init_mean = means[state_i][mix_i]
            init_var = vars[state_i][mix_i]
            mixture.append(NormalDistribution(init_mean, init_var))
        states.append(State(GeneralMixtureModel(mixture), name=str(state_i)))
    hmm_model.add_states(*tuple(states))

    for row in range(len(states_labels)):
        for col in range(len(states_labels)):
            prob = transitions[row][col]
            if prob != 0.:
                hmm_model.add_transition(states[row], states[col], prob)
    for state_i, prob in enumerate(init_prob):
        if prob != 0.:
            hmm_model.add_transition(hmm_model.start, states[state_i], prob)
    for state_i, prob in enumerate(end_prob):
        if prob != 0.:
            hmm_model.add_transition(states[state_i], hmm_model.end, prob)

    hmm_model.bake()

    return hmm_model


def hidden_path(di_array: np.ndarray, hmm_model) -> tuple:
    """

    :param di_array:
    :param hmm_model:
    :return:
    """
    Path = namedtuple('hidden_path', ['logp', 'path'])
    logp, path = hmm_model.viterbi(di_array)
    return Path(logp, ''.join(state.name for idx, state in path[1: -1]))


def call_domain(path: str, min_size: int = 5) -> tuple:
    """

    :param path:
    :param min_size:
    :return:
    """
    max_len = len(path)
    parts = path.split('30')
    cur_start = 0
    for part in parts:
        part_len = len(part)
        last_start = cur_start
        cur_start += part_len + 2
        if cur_start \
                and (part_len + 2 > min_size) \
                and (cur_start <= max_len):
            yield (last_start - 1, last_start + part_len + 2)


def split_diarray(di_array: np.ndarray,
                  gap_mask: np.ndarray,
                  min_width: int = 15) -> dict:
    """

    :param di_array:
    :param gap_mask:
    :param min_width:
    :return:
    """
    di_dict = {}
    gap_mask = remove_small_gap(gap_mask)
    di_array[np.where(gap_mask)[0]] = np.nan
    nan_index = np.where(np.isnan(di_array))[0]
    nan_index = np.r_[-1, nan_index, di_array.size]
    for index, region_width in enumerate((nan_index[1:] - nan_index[:-1])):
        if region_width < min_width:
            continue
        region_st = nan_index[index] + 1
        region_ed = nan_index[index + 1]
        di_dict[region_st] = di_array[region_st: region_ed]

    return di_dict


def train_hmm(clr: cooler.Cooler, mix_num: int = 3, discore_fn=di_score):
    """

    :param clr:
    :param mix_num:
    :param discore_fn:
    :return:
    """
    model = ghmm_model(STATES,
                       INIT_TRANSITION,
                       INIT_PROB,
                       END_PROB,
                       init_mean_fn(mix_num),
                       init_var_fn(mix_num))
    di_dict = {}
    for chrom in clr.chromnames:
        matrix = clr.matrix(sparse=True).fetch(chrom).tocsr()
        di_array = discore_fn(matrix)
        gap_mask = remove_small_gap(np.isnan(clr.bins().fetch(chrom)['weight'].values))
        di_dict[chrom] = split_diarray(di_array, gap_mask)

    train_data = []
    for chrom_di in di_dict.values():
        train_data.extend(di for di in chrom_di.values())
    model.fit(train_data, algorithm='baum-welch',
              max_iterations=10000, stop_threshold=1e-5, n_jobs=CPU_CORE - 5, verbose=False)

    return model


if __name__ == "__main__":
    pass
