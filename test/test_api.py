import random

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import toeplitz

class TestChromMatrix(object):
    def test_properties(self, get_chrom):
        chrom = get_chrom(100000)
        assert chrom._observed.dtype == np.float32
        assert np.all(chrom.mask_index[0]
                      == np.ix_(chrom.mask, chrom.mask)[0])
        assert np.all(chrom.mask2d
                      == chrom._mask[:, np.newaxis] * chrom._mask[np.newaxis, :])

    def test_handle_mask(self, get_chrom):
        chrom = get_chrom(100000)
        valid_size = chrom.mask.sum()
        chrom_len = chrom.shape[0]
        hm = chrom.handle_mask
        assert hm(np.zeros(valid_size), full=True).shape == (chrom_len,)
        assert hm(np.zeros(chrom_len), full=True).shape == (chrom_len,)
        assert hm(np.zeros((valid_size, valid_size)), full=True).shape == (chrom_len, chrom_len)
        assert hm(np.zeros((chrom_len, chrom_len)), full=True).shape == (chrom_len, chrom_len)
        assert hm(np.zeros((4, valid_size)), full=True).shape == (4, chrom_len)
        assert hm(np.zeros((valid_size, 4)), full=True).shape == (4, chrom_len)
        assert hm(np.ones((chrom_len, chrom_len)), full=False).shape == (valid_size, valid_size)
        assert hm(np.ones(chrom_len), full=False).shape == (valid_size,)
        assert hm(np.ones((100, 1000)), full=False).shape == (100, 1000)

    def test_observed(self, get_chrom):
        chrom = get_chrom(100000)
        shape = chrom.shape
        sp_ob = chrom.observed(sparse=True)
        ds_ob = chrom.observed()
        un_normalized_ob = chrom.observed(balance=False)
        assert isinstance(sp_ob, sparse.coo_matrix)
        assert isinstance(ds_ob, np.ndarray)
        assert sp_ob.shape == shape == ds_ob.shape
        assert un_normalized_ob.shape == shape

    def test_decay(self, get_chrom):
        chrom = get_chrom(100000)
        assert chrom.decay().size == chrom.shape[0]
        assert np.all(chrom.decay(ndiags=100)[100:] == 0)

    def test_expected(self, get_chrom):
        chrom = get_chrom(100000)
        decay = chrom.decay()
        real_expected = toeplitz(decay, decay)
        expected = chrom.expected()
        rst1 = random.randrange(chrom.shape[0] - 100)
        rst2 = random.randrange(chrom.shape[0] - 100)
        outer = chrom.shape[0] + random.randrange(100)
        assert np.all(expected[rst1, rst1 + 100]
                      == real_expected[rst1, rst1 + 100])
        assert np.all(expected[rst1:outer, rst1:outer]
                      == real_expected[rst1:outer, rst1:outer])
        assert np.all(expected[rst1: outer, rst2: outer]
                      == real_expected[rst1: outer, rst2:outer])

    def test_oe_corr(self, get_chrom):
        chrom = get_chrom(100000)
        valid_size = chrom.mask.sum()
        sp_oe = chrom.oe(sparse=True)
        ds_oe = chrom.oe(sparse=False)
        unfull_oe = chrom.oe(full=False)
        corr = chrom.corr()
        unfull_corr = chrom.corr(full=False)
        filled_corr = chrom.corr(ignore_diags=5, fill_value=5)
        assert sparse.isspmatrix(sp_oe)
        assert (sp_oe.shape
                == chrom.shape
                == ds_oe.shape
                == corr.shape
                == filled_corr.shape)
        assert (unfull_oe.shape
                == (valid_size, valid_size)
                == unfull_corr.shape)

    def test_score(self, get_chrom):
        chrom = get_chrom(100000)
        valid_size = chrom.mask.sum()
        insu_score = chrom.insu_score()
        unfull_insu_score = chrom.insu_score(full=False, normalize=False, balance=False)
        di_score_adp = chrom.di_score(method='standard')
        di_score_std = chrom.di_score(method='adaptive')
        unfull_di_score = chrom.di_score(full=False)
        assert (unfull_di_score.shape
                == unfull_insu_score.shape
                == (valid_size,))
        assert (insu_score.shape
                == di_score_adp.shape
                == di_score_std.shape
                == (chrom.shape[0],))

    def test_peaks(self, get_chrom, capsys):
        chrom = get_chrom(10000)
        peaks = chrom.peaks.hiccups(num_cpus=20)
        captured = capsys.readouterr()
        print(f'Number of peaks in {chrom.chrom} '
              f'with resolution {chrom._binsize}: {peaks.shape[0]}')
        assert isinstance(peaks, pd.DataFrame)
        assert peaks.size > 0

    def test_compartments(self, get_chrom):
        chrom = get_chrom(100000)
        coms_pca = chrom.compartments.decomposition(method='pca', numvecs=2)
        coms_eigen = chrom.compartments.decomposition(method='eigen', numvecs=4)
        unfull_coms = chrom.compartments.decomposition(numvecs=1, full=False)
        assert coms_pca.shape == (2, chrom.shape[0])
        assert coms_eigen.shape == (4, chrom.shape[0])
        assert unfull_coms.shape == (1, chrom.mask.sum())

    def test_tads(self, get_chrom):
        pass
