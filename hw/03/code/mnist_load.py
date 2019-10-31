from mnist import MNIST
import numpy as np
from label_data import PartitionData, LabelData

mndata = MNIST('mnist')
Xt, Yt = mndata.load_training()
Xv, Yv = mndata.load_testing()

def load_mnist_data(s0, s1, ratio, usage):
    X = np.vstack((Xt, Xv))
    Y = np.hstack((Yt, Yv))

    X = X[(Y == s0) | (Y == s1)].astype(np.float32)
    Y = Y[(Y == s0) | (Y == s1)].astype(np.float32)



    M = int(round(np.shape(X)[0] * usage))
    use = np.zeros(np.shape(X)[0], dtype=np.bool)
    use[0:M] = 1.0
    np.random.shuffle(use)

    X = X[use, :]
    Y = Y[use]

    Y = np.array(Y, dtype=np.float32)
    unique = set(np.array(Y, dtype=np.float32))
    Y[Y == min(unique)] = 0.0
    Y[Y == max(unique)] = 1.0

    md = LabelData()
    md.add_data(X, Y)
    mnist_data = PartitionData(md)
    mnist_data.partition(ratio)

    indices_lp = {1.0: max(unique), 0.0: min(unique)}
    return mnist_data, indices_lp

if __name__ == "__main__":
    usage = 0.1
    ratio = 0.5
    s0, s1 = 0,4
    mnist, names = load_mnist_data(s0, s1, ratio, usage)
