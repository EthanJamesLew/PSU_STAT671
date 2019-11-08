from mpl_toolkits.mplot3d import Axes3D
from krr import KernelRidgeRegression, krr
from label_data import LabelData
import numpy as np
import matplotlib.pyplot as plt
from mnist_load import load_mnist_data

import time

def k_polynomial(x, xp, d):
    return (np.dot(x, xp)+1)**d


def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x-xp)**2)/(2*(sigma**2)))


def k_sigmoid(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

def risk(y, yp):
    return 1/(np.size(y))*np.sum(0.5*np.abs(y - yp))

def view_digits(ax, digits, nx, ny):
    width = int(np.sqrt(digits.shape[1]))
    img = np.zeros((nx*width, ny*width))
    idx = 0
    for i in range(0, nx):
        for j in range(0, ny):
            img[i*width:i*width+width, j*width:j*width+width] = digits[idx].reshape((width, width))
            idx += 1
    ax.imshow(img, extent=[0, 1, 0, 1], cmap='Greys')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

def classify_test(s0, s1, usage, ratio, k):
    mnist, names = load_mnist_data(s0, s1, ratio, usage)

    ld = LabelData()
    ld.add_data(mnist.training[0], mnist.training[1])

    t0 = time.time()
    kregr = KernelRidgeRegression(ld, k=k, l=.0001)
    t1 = time.time()
    ttotal = t1 - t0
    #print(mnist.training[1])

    t0 = time.time()
    y_v = np.array([krr(ld, kregr._alpha, k, x) for x in mnist.validation[0]])
    y_v[y_v > 0.5] = 1
    y_v[y_v < 0.5] = 0
    t1 = time.time()
    vtotal = t1 - t0

    t0 = time.time()
    y_t = np.array([krr(ld, kregr._alpha, k, x) for x in mnist.training[0]])
    y_t[y_t > 0.5] = 1
    y_t[y_t < 0.5] = 0
    t1 = time.time()
    rtotal = t1 - t0

    error = risk(mnist.validation[1], y_v)
    erisk = risk(mnist.training[1], y_t)

    return {"error": error, "risk": erisk, "training time": ttotal, "validation time": vtotal, "risk time": rtotal,
            "training size": mnist.training[1].shape[0], "validation size": mnist.validation[1].shape[0]}

if __name__=='__main__':
    cerror = np.zeros((10, 10))
    for s0 in range(0, 10):
        for s1 in range(0, 10):
            res = classify_test(s0, s1, 0.1, 0.5, lambda x, y: k_polynomial(x, y, 2))
            cerror[s0, s1] = res['error']
            print(s0, s1)


    print(cerror)

