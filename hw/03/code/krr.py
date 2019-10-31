''' Kernel Ridge Regression
@author: Ethan Lew
(elew@pdx.edu)

10/30/2019

Implements Kernel Ridge Regression
'''
import numpy as np
from inspect import signature
from label_data import LabelData, PartitionData


def kernel_mat(f, x):
    n = len(x)
    K = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i+1):
            v = f(x[i], x[j])
            K[i, j] = v
            K[j, i] = v
    return K

def lstsq(A, b):
    AA = A.T @ A
    bA = b @ A
    D, U = np.linalg.eigh(AA)
    Ap = (U * np.sqrt(D)).T
    bp = bA @ U / np.sqrt(D)
    return np.linalg.lstsq(Ap, bp, rcond=None)

def alpha(data, k, l):
    n = data.n
    K = kernel_mat(k, data.x)
    G = K + l*n*np.eye(n)
    return np.linalg.solve(G, data.y)
    #return np.linalg.pinv(G) @ data.y


def krr(data, alpha, k, x):
    total  = 0
    for idx, xd in enumerate(data.x):
        total += alpha[idx]*k(xd, x)
    return total


class KernelRidgeRegression:
    def __init__(self, data, k=np.dot, l=1e-3):
        self._data = data
        self._alpha = None
        self._k = k
        self._l = l
        self.train()

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, l):
        self._l = l
        self.train()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, func):
        if len(signature(func).parameters) != 2:
            raise Exception("kernel needs to have two arguments k(x, xp)!")
        else:
            self._k = func
            self.train()

    def train(self):
        self._alpha = alpha(self._data, self._k, self._l)

    def __call__(self, x):
        if type(x) is float:
            return krr(self._data, self._alpha, self._k, x)
        else:
            m = x.shape[0]
            ans = np.zeros((m))
            for idx, xi in enumerate(x):
                ans[idx] = krr(self._data, self._alpha, self._k, xi)
            return ans


if __name__ == '__main__':

    print(kernel_mat(np.dot, [1,2,3]))
    data = np.array([[2], [1], [7], [9]])
    labs = np.array([0.4, 1.2, 3.4, -0.4])
    ld = LabelData()
    ld.add_data(data, labs)

    a = alpha(ld, np.dot, 0.01)

    kregr = KernelRidgeRegression(ld)
    print(kregr([3]))
