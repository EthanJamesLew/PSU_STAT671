'''
Ethan Lew
(elew@pdx.edu)

Spectral clustering implementation
'''
import numpy as np

from kkmeans import KKernelClustering, kernel_mat
from graph import construct_graph

class SpectralClustering(KKernelClustering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # override the special kernel
        self._kernel = np.dot

    def train(self, X, w=None, **kwargs):
        W, D, L = construct_graph(X, **kwargs)
        W = (W[W > 1E-12]).astype(np.float32)
        L = D - W
        #self._L = L
        #self._W = (W[W > 1E-10]).astype(np.float32)
        #self._D = D
        #Z = W @ np.diag(np.sqrt(np.diag(D)))
        eD, eV = np.linalg.eig(L)
        eidx = np.argsort(-eD)
        Y = eV[:, eidx[:self._k]].T
        return super(SpectralClustering, self).train(Y.T, w=w)


class KSpectralClustering(KKernelClustering):
    def __init__(self, k, max_iter, kernel, **kwargs):
        # make kkmeans conventional k means by passing a linear kernel
        super().__init__(k, max_iter, np.dot, **kwargs)

        # assign a new kernel to the spectral preprocessing
        self._skernel = kernel

    def train(self, X, w=None, **kwargs):
        # get kernel matrix
        K = kernel_mat(self._skernel, X)

        # use eigendecomposition to get z
        eD, eV = np.linalg.eig(K)
        eidx = np.argsort(-eD)

        # Z is the first k largest eigenvectors
        Z = eV[:, eidx[:self._k]]
        self._W = Z @ Z.T

        # normalize rows
        Z /= np.tile(np.linalg.norm(Z, axis=1), reps=(self._k, 1)).T

        # perform KRR on the rows
        return super(KSpectralClustering, self).train(Z, w=w)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from synthetic import generate_ring_2d
    from kkmeans import k_gaussian, k_polynomial
    from graph import view_graph

    M = 200
    N = 2
    k = 2
    data = np.vstack((generate_ring_2d(20, 0, 1), generate_ring_2d(200, 5, .3)))
    sc = SpectralClustering(k, 100, lambda x,y: k_polynomial(x, y, 2))
    membership = sc.train(data)
    #membership = sc.classify(data)
    print(membership)

    fig, ax = plt.subplots()
    view_graph(ax, data, sc._W)
    plt.show()

    plt.figure()
    plt.scatter(*data[membership==0, :].T)
    plt.scatter(*data[membership==1, :].T)
    plt.scatter(*data[membership==2, :].T)
    plt.scatter(*data[membership==3, :].T)
    plt.scatter(*data[membership==4, :].T)
    plt.show()

