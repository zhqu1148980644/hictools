"""Tools for compartment analysis."""

from dataclasses import dataclass

import numpy as np

from .chrommatrix import ChromMatrix
from .utils.numtools import fill_diags, eig, pca
from .utils.utils import suppress_warning


def get_pca_compartment(mat: np.ndarray, vecnum: int = 3):
    """Fetch A/B compartmentation through principle component analysis(use svd).

    :return type: np.ndarray. positive values represent A compartment and negative
    values represent B compartment.
    """
    _, components = pca(mat, vecnum=vecnum)

    return components


def get_eigen_compartment(mat: np.ndarray, vecnum: int = 3, subtract_mean=False, divide_by_mean=False):
    """Fetch A/B compartmentation through eigen value decompositon.

    :return type: np.ndarray. Positive values represent A compartment and negative values represent B compartment.
    """
    if subtract_mean or divide_by_mean:
        mat = mat.copy()
        mean = np.mean(mat)
        if subtract_mean:
            mat -= mean
        else:
            mat /= mean

    _, eigvecs = eig(mat, vecnum=vecnum)

    return eigvecs


def corr_sorter(eigvecs: np.ndarray, corr: np.ndarray):
    """Choose the most possible vector which may infer the A/B seperation based on pearson correlation matrix.
        1. Choose vector:
            In general, the sums of pearson correlation value within A and B is larger than the sums of pearson
            correlation value across A-B.
        2. Choose AB:
            In general, the interactions within A are stronger than the interactions within B.
    """

    def mean_corr(mat, compartment):

        com_mask1 = compartment > 0
        com_mask2 = compartment < 0
        return (
            np.nanmean(mat[np.ix_(com_mask1, com_mask1)]),
            np.nanmean(mat[np.ix_(com_mask2, com_mask2)]),
            np.nanmean(mat[np.ix_(com_mask1, com_mask2)])
        )

    coms = []
    for i, component in enumerate(eigvecs):
        # In some cases the divergence between max-value in A/B is too large.
        diverse = np.abs(np.min(component)) / np.max(component)
        coma = component > 0
        num_b = len(np.where(~coma)[0])
        if num_b == 0:
            ratio = 1
        else:
            ratio = (coma.size - num_b) / coma.size
        if ((diverse > 10)
                or (diverse < 0.1)
                or (ratio > 15)
                or (ratio < 1 / 15)):
            possible = False
        else:
            possible = True

        mean_aa, mean_bb, mean_ab = mean_corr(mat=corr, compartment=component)

        coms.append(
            (
                component * np.sign(mean_aa - mean_bb),
                mean_aa + mean_bb - 2 * mean_ab,
                possible
            )
        )

    sorted_coms = sorted(coms, key=lambda x: (x[2], x[1]), reverse=True)

    return np.array([com[0] for com in sorted_coms])


@dataclass
class Compartment(object):
    """Calculate A/B compartments based on decomposition of intra-chromosomal
    interaction matrix. Currently, two methods are supported for detecting A/B
    compatements. 'pca' uses principle component analysis based on corr matrix
    and 'eigen' uses eigen value decomposition based on OE-1 matrix.

    :return: np.ndarray. Array representing the A/B seperation of compartment.
    Negative value denotes B compartment.
    """
    chrom_ma: ChromMatrix

    @suppress_warning
    def __call__(self, method: str = "pca", numvecs: int = 3, sort: bool = True, full: bool = True) -> np.ndarray:
        corr = None
        if method == 'pca':
            corr = self.chrom_ma.corr(full=False)
            vecs = get_pca_compartment(mat=corr, vecnum=numvecs)

        elif method == "eigen":
            dsoe = self.chrom_ma.oe(sparse=False, full=False)
            dsoe = fill_diags(dsoe, diags=1, fill_values=0)
            vecs = get_eigen_compartment(mat=dsoe - 1, vecnum=numvecs)
        else:
            raise NotImplementedError("Method should be one of [pca, eigen]")

        vecs = np.array(vecs)
        if sort:
            if corr is None:
                corr = self.chrom_ma.corr(full=False)
            vecs = corr_sorter(eigvecs=vecs, corr=corr)

        return self.chrom_ma.handle_mask(vecs, full=full)

