'''
Ethan Lew
(elew@pdx.edu)

Contains methods to describe graphs using matrices
'''

import numpy as np
import scipy.spatial as ssp

def get_dist_matrix(X, p=2):
    '''
    Given a collection of M points of N dimensions, calculate a matrix D such that
    D_ij = ||xi - xj||_p, for xi, xj in X
    :param X: [M x N] X
    :param p: p-norm parameter
    :return: [M x M] D
    '''
    return ssp.distance_matrix(X, X, p=p)

def construct_graph(X, p=2, type="knn", param=None):
    '''
    From M data points of dimension N, construct the weighted adjacency matrix W wrt distance p-norm, a degree matrix D
    and graph laplacian L
    :param X:
    :param p:
    :param type: type of graph construction,
        "e-ball": edges are created if they are greater than a distance threshold defined by the parameter
        "k-nn": edges are created if they belong to k-nearest neighbors
        "full": edges are created to every vertex
    :param param: construction parameter
    :return: W, D, L
    '''
    dist = get_dist_matrix(X, p=p)
    if type == "eball":
        e = param if param is not None else 1.0
        dist[np.abs(dist) < e] = 0
        W = dist
    elif type == "knn":
        mask = np.zeros(dist.shape, dtype=np.int32)
        k = param if param is not None else 2
        idx = np.argsort(dist)
        for ii in range(dist.shape[0]):
            for jj in range(k):
                mask[ii, idx[jj, ii]] = 1
                mask[idx[jj, ii], ii] = 1
        W = dist*mask
    elif type == "full":
        W = dist
    D =  np.diag(np.sum(np.abs(W) > 1E-6, axis=0))
    return W, D, D - W

def spectral_clustering(L, k):
    ed, ev = np.eig(L)
    edidx = np.argsort(ed)
    ed = ed[edidx]
    evl = ev[:, edidx]
    ed = ed[:k]
    Y = evl[:, :k].T
    # TODO: k means the spectrum

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.random.rand(6, 3)

    W, D, L = construct_graph(X, param= 2)
    print(L)
