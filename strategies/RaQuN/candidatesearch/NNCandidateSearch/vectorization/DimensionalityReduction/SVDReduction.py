import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.DimensionalityReduction.DimensionalityReduction import \
    DimensionalityReduction


class SVDReduction(DimensionalityReduction):
    def __init__(self, n_components = 10):
        self.n_components = n_components

    def reduce(self, in_mat: np.ndarray) -> np.ndarray:
        svd = TruncatedSVD(n_components=self.n_components)
        return svd.fit_transform(csr_matrix(in_mat))

