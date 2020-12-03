import typing as T
from functools import lru_cache
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable

from .utils import get_logger

log = get_logger()

Tensor = T.TypeVar('Tensor', np.ndarray, torch.Tensor)


def random_init(n_beads: int) -> np.ndarray:
    """Init beads position randomly"""
    return np.random.rand(n_beads, 3)


def pairwise_distances(x: Tensor) -> Tensor:
    """Pairwise euclidean distance of all positions"""
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_t = x.T
    y_norm = x_norm.reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * x @ y_t
    return dist


class ReConstruct(ABC):
    """Optimize based 3D model reconstruction
    using Hi-C interaction matrix.
    """

    def __init__(self,
                 matrix: np.ndarray,
                 gpu: bool = False):
        # TODO:
        # How to deal with nan value?
        # Using sparse matrix?
        self.matrix = matrix
        self.n_beads = self.matrix.shape[0]
        self.gpu = gpu and torch.cuda.is_available()
        self._pos = Variable(self.to_torch(self.init_positions()), requires_grad=True)

    def to_torch(self, arr: np.ndarray, **kwargs) -> torch.Tensor:
        tensor = torch.from_numpy(arr, **kwargs)
        if not self.gpu:
            return tensor
        else:
            return tensor.cuda()

    def init_positions(self) -> np.ndarray:
        return random_init(self.n_beads)

    @property
    @abstractmethod
    def loss(self):
        pass

    def train(self, batches=10, steps=100, optim=optim.SGD, **kwargs):
        """Training the model."""
        optimizer = optim([self._pos], **kwargs)
        for b_idx in range(batches):
            optimizer.zero_grad()
            for i in range(steps):
                self.loss.backward(retain_graph=True)
                optimizer.step()
            log.info(f"batch: {b_idx + 1}/{batches}\tloss: {self.loss}")

    @property
    def positions(self) -> np.ndarray:
        self._pos.requires_grad = False
        pos = self._pos
        if self.gpu:
            pos = pos.cpu()
        pos = pos.numpy()
        self._pos.requires_grad = True
        return pos

    @property
    def pairwise_dist(self) -> np.ndarray:
        return pairwise_distances(self.positions)

    def get_positions(self, start: int, end: int,
                      mask: T.Optional[np.ndarray] = None) -> pd.DataFrame:
        """Output the result table"""
        pos = self.positions
        df = pd.DataFrame(pos, columns=['x', 'y', 'z'])
        length = end - start
        n_beads = pos.shape[0]
        len_bead = length // n_beads
        if mask is not None:
            df[~mask][['x', 'y', 'z']] = np.nan
        return df


class MetricMDS(ReConstruct):
    """Multidimensional scaling, according to the method described in:

        Nelle Varoquaux 2014. A statistical approach for inferring the
        3D structure of the genome.

    Expected distances matrix:
        d_{ij} = 1 / (IF_{ij})^(\\alpha)

    The input is the interaction frequency matrix.

    """

    def __init__(self, matrix: np.ndarray,
                 alpha: float = 1.0,
                 **kwargs):
        self.alpha = alpha
        super().__init__(matrix, **kwargs)

    @property
    @lru_cache(1)
    def d_exp(self):
        ifq = self.matrix
        m = ifq ** self.alpha
        # How to deal with zero value?
        m[m == 0] = m[m > 0].min()
        d_exp = 1 / m
        d_exp[np.eye(d_exp.shape[0]) == 1] = 0
        return self.to_torch(d_exp)

    @property
    def loss(self):
        d_exp = self.d_exp
        d = pairwise_distances(self._pos)
        cost = ((d - d_exp) ** 2).mean() / d_exp.max()
        return cost


class Lorentzian(ReConstruct):
    """
        Tuan Trieu 2016. 3D genome structure modeling by Lorentzian
        objective function
    """

    @property
    def loss(self):
        raise NotImplementedError


class MiniMDS(ReConstruct):
    """
        Rieber. miniMDS: 3D structural inference from high-resolution Hi-C data.
    """

    @property
    def loss(self):
        raise NotImplementedError


class GEM(ReConstruct):
    """
        Guangxiang Zhu. Reconstructing spatial organizations of chromosomes
        through manifold learning
    """

    @property
    def loss(self):
        raise NotImplementedError
