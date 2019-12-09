'''
Kernalized k-means
@author: Ethan Lew

Implements kernalized k-means
'''

import numpy as np
from inspect import signature
from sklearn.metrics.pairwise import pairwise_kernels

from partition import random_euclid_partition, random_uniform_partition

##KERNELS
def k_polynomial(x, xp, d):
    return (np.dot(x, xp)+1)**d


def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x-xp)**2)/(2*(sigma**2)))


def k_tanh(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

def kernel_mat_pair(f, x, y=None):
    return pairwise_kernels(x, y, f)


def kernel_mat(f, x):
    '''
    :param f: kernel function
    :param x: vector of values
    :return: K symmetric matrix
    '''
    n = len(x)
    K = np.zeros((n, n), dtype=x.dtype)
    for i in range(0, n):
        for j in range(0, i+1):
            v = f(x[i], x[j])
            K[i, j] = v
            K[j, i] = v
    return K

class KKernelClustering:
    def __init__(self, k, max_iter, kernel):
        self._k = k
        self._max_iter = max_iter
        self._kernel = kernel

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, func):
        if len(signature(func).parameters) != 2:
            raise Exception("kernel needs to have two arguments k(x, xp)!")
        else:
            self._kernel = func

    def train(self, X, w=None):
        M, N = X.shape
        w = w if w is not None else np.ones(M)
        self._w = w
        K = kernel_mat_pair(self._kernel, X)
        self._labels = random_uniform_partition(X, self._k)
        #rs = check_random_state(None)
        #self._labels = rs.randint(self._k, size=M)

        dist = np.zeros((M, self._k), dtype=X.dtype)
        self.within_distances = np.zeros(self._k, dtype=X.dtype)

        for idx in range(self._max_iter):
            dist.fill(0)
            self._get_dist(K, dist, self.within_distances, update=True)
            labels_prev = self._labels
            self._labels = dist.argmin(axis=1)

            n_crit = np.sum((self._labels - labels_prev) == 0)
            if 1 - float(n_crit) / M < 1E-20:
                break
        self._err = dist.min(axis=1)
        self._wss = np.sum(self._err)
        self._X = X
        return self._labels

    def _get_dist(self, K, dist, within_dist, update=True):
        for label in range(self._k):
            is_idx = self._labels == label
            if np.sum(is_idx) == 0:
                denom = 1
            else:
                #raise ValueError("No Label")
                denom = self._w[is_idx].sum()
            if update == True:
                KK = K[is_idx][:, is_idx]
                dist_label = np.sum(np.outer(self._w[is_idx], self._w[is_idx])*KK / (denom * denom))
                within_dist[label] = dist_label
                dist[:, label] += dist_label
            else:
                dist[:, label] += within_dist[label]
            dist[:, label] -= 2 * np.sum(self._w[is_idx] * K[:, is_idx], axis=1) / denom
        if update is True:
            dist[:, :] += np.tile(np.diag(K), reps=(self._k, 1)).T

    def classify(self, X, thresh=True):
        M, N = X.shape
        K = kernel_mat_pair(self._kernel, X, self._X)
        dist = np.zeros((M, self._k))
        self._get_dist(K, dist, self.within_distances, update=False)
        if thresh is True:
            return dist.argmin(axis=1)
        else:
            return  dist.min(axis=1)

