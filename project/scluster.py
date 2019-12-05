'''
Ethan Lew
(elew@pdx.edu)

Spectral clustering implementation
'''
import numpy as np

from kkmeans import KKernelClustering, kernel_mat
from graph import construct_graph

class SpectralClustering(KKernelClustering):
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
        return super(SpectralClustering, self).train(Z, w=w)


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

