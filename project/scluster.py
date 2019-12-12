'''
Ethan Lew
(elew@pdx.edu)

Spectral clustering implementation, supporting both graph construction and kernels
for the affinity matrix
'''
import numpy as np

from kkmeans import KKernelClustering, kernel_mat
from graph import construct_graph

class SpectralClustering(KKernelClustering):
    ''' SpectralClustering
    SpectralClustering is a clustering algorithm that performs conventional K means
    clustering on the spectrum of a graph Laplacian constructed from the input data.

    In this object, the graph itself is specified directly. Here, the following graph
    producing strategies can be specified:
        1. K-Nearest Neighbor
        2. Epsilon Ball
        3. Fully Connected
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # override the special kernel
        self._kernel = np.dot

    def train(self, X, w=None, **kwargs):
        W, D, L = construct_graph(X, **kwargs)
        W[W > 1E-50] = 1
        L = D - W
        eD, eV = np.linalg.eig(L)
        eidx = np.argsort(eD)
        Y = np.real(eV[:, eidx[:self._k]].T)
        return super(SpectralClustering, self).train(Y.T, w=w)


class KSpectralClustering(KKernelClustering):
    ''' KSpectralClustering
    Kernel Spectral Clustering utilizes the kernel matrix as the affinity matrix, making
    an explicit graph construction description irrelevant. Here, the kernel needs to be
    passed in and the clustering can be performed with training.
    '''
    def __init__(self, k, max_iter, kernel, **kwargs):
        super().__init__(k, max_iter, np.dot, **kwargs)
        self._skernel = kernel

    def train(self, X, w=None, **kwargs):
        # get kernel matrix
        K = kernel_mat(self._skernel, X)
        # use eigendecomposition to get z
        eD, eV = np.linalg.eig(K)
        eidx = np.argsort(-eD)
        # Z is the first k largest eigenvectors
        Z = np.real(eV[:, eidx[:self._k]])
        self._AD = Z @ Z.T
        # normalize rows
        Z /= np.tile(np.linalg.norm(Z, axis=1), reps=(self._k, 1)).T
        return super(KSpectralClustering, self).train(Z, w=w)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from synthetic import generate_ring_2d
    from kkmeans import k_gaussian, k_polynomial

    M = 200
    N = 2
    k = 3
    data = np.vstack((generate_ring_2d(400, 14, .3), generate_ring_2d(200, 4, .3), generate_ring_2d(200, 10, .3)))
    sc = SpectralClustering(k, 100, lambda x,y: k_gaussian(x, y, .3))
    membership = sc.train(data, type="eball", param=0.6)
    print(membership)

    plt.figure()
    plt.scatter(*data[membership==0, :].T)
    plt.scatter(*data[membership==1, :].T)
    plt.scatter(*data[membership==2, :].T)
    plt.show()

