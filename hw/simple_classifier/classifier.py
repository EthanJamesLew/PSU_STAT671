''' Simple Classifier
Ethan Lew
10/10/19

A simple classifier designed to match the description of STAT 671 lecture 1.
'''
import numpy as np

class Classifier:
    def __init__(self, N):
        '''
        :param N: dimensionality of feature space
        :return: None
        '''
        self.N = N
        self.cn = np.zeros((N))
        self.cp = np.zeros((N))
        self.X = None
        self.Y = None

    @property
    def c(self):
        return (self.cp + self.cn)/2

    def train(self, X, Y):
        ''' train
        Adds inputs X and Y to training set and re-computes the centroids
        :param X: (N x M) N samples of dimensionality M
        :param Y: (N) N labels in {-1, 1}
        :return:
        '''
        unique = list(set(Y))

        if len(unique) != 2:
            return

        Y[Y == unique[0]] = 1
        Y[Y == unique[1]] = -1

        # append the data to the old training data, or create new
        if self.X == None and self.Y == None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack((self.X, X))
            self.Y = np.vstack((self.Y, Y))

        # get where the data is for each label
        idxp = (self.Y == 1)
        idxn = (self.Y == -1)

        # compute the centroids
        self.cn = np.mean(self.X[idxn], axis=0)
        self.cp = np.mean(self.X[idxp], axis=0)

    def classify(self, x):
        ''' classify
        given an observation x, predict the label
        :param x: (N) sample of dimensionality N
        :return: y label in {-1, 1}
        '''
        return int(np.sign(np.dot(self.cp - self.cn, x - self.c)))


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    iris_sk = load_iris()
    species_lp = {0: 'I.setosa', 1: 'I. versicolor', 2: 'I. virginica'}
    iris_df = {'sepal_length': iris_sk['data'][:, 0], 'sepal_width': iris_sk['data'][:, 1],
               'petal_length': iris_sk['data'][:, 2],
               'petal_width': iris_sk['data'][:, 3], 'species': iris_sk['target']}
    iris_df = pd.DataFrame(data=iris_df)

    def partition_df(df, ratio):
        N = df.shape[0]
        M = round(N * ratio)
        train = np.zeros((N), dtype=np.bool)
        train[0:M] = 1
        np.random.shuffle(train)
        train_df = df[train]
        val_df = df[~train]
        return train_df, val_df


    # Get the two species to classify
    test1_df = iris_df.loc[iris_df['species'] != 2]

    # Get training and validation sets
    train1_df, ver1_df = partition_df(test1_df, 0.8)

    iris_classifier = Classifier(4)

    iris_classifier.train(np.array(train1_df)[:, :-1], np.array(train1_df)[:, -1] * 2 - 1)

    val  = np.array([iris_classifier.classify(x) for x in np.array(ver1_df)[:, :-1]])
    print(val)
    print(np.array(ver1_df, dtype=np.int)[:, -1]*2-1 )

    D = np.array(test1_df)[:, :-1]
    y = np.array(test1_df)[:, -1]
    U, S, Vh = np.linalg.svd(D, full_matrices=True)

    U = U[:3, :4]
    Dr = U@D.T
    Dr = Dr.T
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(Dr[:, 0], Dr[:, 1], Dr[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=10)
    cp = iris_classifier.cp
    cp = U@cp
    ax.scatter(cp[0], cp[1], cp[2],
               cmap=plt.cm.Set1, edgecolor='k', s=100)
    cn = iris_classifier.cn
    cn = U @ cn
    ax.scatter(cn[0], cn[1], cn[2],
               cmap=plt.cm.Set1, edgecolor='k', s=100)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()
