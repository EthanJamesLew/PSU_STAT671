from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from classifier import PartitionData,  SimpleClassifier, risk
from tqdm import tqdm

mndata = MNIST('mnist')
Xt, Yt = mndata.load_training()
Xv, Yv = mndata.load_testing()

def load_mnist_data(s0, s1, ratio, usage):
    X = np.vstack((Xt, Xv))
    Y = np.hstack((Yt, Yv))

    X = X[(Y == s0) | (Y == s1)]
    Y = Y[(Y == s0) | (Y == s1)]

    M = round(np.shape(X)[0] * usage)
    use = np.zeros(np.shape(X)[0], dtype=np.bool)
    use[0:M] = 1
    np.random.shuffle(use)

    X = X[use, :]
    Y = Y[use]

    Y = np.array(Y, dtype=np.int)
    unique = set(np.array(Y, dtype=np.int))
    Y[Y == min(unique)] = -1
    Y[Y == max(unique)] = 1

    mnist_data = PartitionData()
    mnist_data.add_data(X, Y)
    mnist_data.partition(ratio)

    indices_lp = {1: max(unique), -1: min(unique)}
    return mnist_data, indices_lp

def classify_mnist(s0, s1, ratio, usage):
    sc = SimpleClassifier()
    mnist, names  = load_mnist_data(s0, s1, ratio, usage)
    sc.k = lambda x, xp: (np.dot(x, xp) )**1
    sc.add_data(*mnist.training)
    sc.train()

    valr = np.zeros(np.shape(mnist.training[0])[0], dtype=np.int)
    val = np.zeros(np.shape(mnist.validation[0])[0], dtype=np.int)
    total_calcs = np.shape(mnist.training[0])[0] + np.shape(mnist.validation[0])[0] + 1

    with tqdm(total=total_calcs, desc="Running Risk/Error Analysis", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i in range(0, np.shape(mnist.training[0])[0]):
            valr[i] = sc.classify(mnist.training[0][i])
            pbar.update(1)
        for i in range(0, np.shape(mnist.validation[0])[0]):
            val[i] = sc.classify(mnist.validation[0][i])
            pbar.update(1)

    return risk(valr, mnist.training[1]), risk(val, mnist.validation[1])

risk_all = np.zeros((10,10))
error_all = np.zeros((10,10))
for i in range(0, 10):
    for j in range(0, i):
        print("Classifying %d and %d" % (i, j))
        risk_r, error_r = classify_mnist(i, j, 0.5, 0.1)
        print("\tRisk: ", risk_r)
        print("\tError: ", error_r)
        risk_all[i, j] = risk_r
        risk_all[j, i] = risk_r
        error_all[i, j] = error_r
        error_all[j, i] = error_r
#print(labels[20])
#plt.figure()
#plt.imshow(np.reshape(images[20],(28, 28)), extent=[0, 1, 0, 1])
#plt.show()