'''
Ethan Lew
(elew@pdx.edu)

Contains methods to describe graphs using matrices
'''

import numpy as np
import scipy.spatial as ssp
import matplotlib.collections as mcl

def view_graph(ax, X, A, color=[0,0,0]):
    ''' view_graph
    Given a graph with vertices at X and edge adjacency A, plot the results on ax
    '''
    M, N = X.shape
    ax.scatter(*X.T, zorder=100)
    ls = []
    alphas = []
    for idx, xi in enumerate(X):
        for jdx, xj in enumerate(X):
            if A[idx, jdx] > 1E-12 and idx < jdx:
                ls.append([xi, xj])
                alphas.append(A[idx, jdx])
    alphas = np.log(1/np.array(alphas)) + 0.1
    alphas /= np.max(alphas)
    print(alphas)
    alphas = np.clip(alphas, 0.0, 1.0)
    print(alphas)
    n_seg = len(ls)
    RGB = np.tile(color, reps=[n_seg, 1])
    RGBA = np.vstack((RGB.T, alphas)).T
    ax.add_collection(mcl.LineCollection(ls, colors=RGBA))
    return ax


def get_dist_matrix(X, dist= lambda x,y: np.linalg.norm(x - y)):
    '''
    Given a collection of M points of N dimensions, calculate a matrix D such that
    D_ij = ||xi - xj||_p, for xi, xj in X
    :param X: [M x N] X
    :return: [M x M] D
    '''
    M, N = X.shape
    D = np.zeros((M, M))
    for idx, xi in enumerate(X):
        for jdx, xj in enumerate(X):
            D[idx, jdx] = dist(xi, xj)
    return D

def construct_graph(X, type="knn", param=None, **kwargs):
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
    dist = get_dist_matrix(X, **kwargs)
    if type == "eball":
        e = param if param is not None else 1.0
        mask = np.abs(dist) < e
        W = dist*mask
    elif type == "knn":
        mask = np.zeros(dist.shape, dtype=np.int32)
        M, _ = dist.shape
        k = param if param is not None else 2
        for ii in range(M):
            di = dist[ii, :]
            idx = np.argsort(di)
            mask[ii, idx[:k]] = 1
            mask[idx[:k], ii] = 1
        W = dist*mask
    elif type == "full":
        W = dist
    D =  np.diag(np.sum(W, axis=0))
    return W, D, D - W

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.random.rand(100, 2)
    W, D, L = construct_graph(X, type='knn', param= 3)
    fig, ax = plt.subplots()
    view_graph(ax, X, W)
    plt.show()
