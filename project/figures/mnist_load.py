from mnist import MNIST
import numpy as np
from label_data import PartitionData, LabelData
import time

usage = 0.01
mndata = MNIST('../MNIST')
Xt, Yt = mndata.load_training()
Xv, Yv = mndata.load_testing()
Xt, Yt = np.array(Xt), np.array(Yt)
Xv, Yv = np.array(Xv), np.array(Yv)
N = Yv.size
mask = np.zeros(N, dtype=np.bool)
mask[:int(N*usage)] = 1
np.random.shuffle(mask)
Xv, Yv = Xv[mask, :], Yv[mask]

N = Yt.size
mask = np.zeros(N, dtype=np.bool)
mask[:int(N*usage)] = 1
np.random.shuffle(mask)
Xt, Yt = Xt[mask, :], Yt[mask]


def view_digits(ax, digits, nx, ny):
    '''
    :param ax: pyplot axis object
    :param digits: array [N x 784] MNIST data
    :param nx: number of columns
    :param ny: number of rows
    :return: None
    '''
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

def risk(y, yp):
    '''
    :param y: {pm 1} classified values
    :param yp: {pm 1} true values
    :return:
    '''
    return 1/(np.size(y))*np.sum(0.5*np.abs(y - yp))


