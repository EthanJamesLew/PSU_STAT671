import numpy as np

class Perceptron:
    def __init__(self):
        self._X = None
        self._Y = None
        self._w = None
        self._alpha = None
        self._k = np.dot

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, func):
        self._k = func

    @property
    def w(self):
        return self._w

    def train(self, X, Y):
        self._X = X
        self._Y = Y

        self._w = Y[3]*X[3, :]

        for i, xi in enumerate(X):
            if Y[i]*np.dot(self._w, xi) <= 0:
                self._w = self._w + Y[i]*xi

        print(self.w)

    def _kernel_machine(self, x):
        return np.sign(np.sum(self._alpha*np.apply_along_axis(lambda y: self._k(y, x), 1, self._X)))

    def train_kernel(self, X, Y):

        self._X = X
        self._Y = Y

        self._alpha = np.zeros((np.shape(X)[0]))
        for i in range(0, len(Y)):
            print('changed')
            if self._kernel_machine(X[i]) != Y[i]:
                self._alpha[i] += Y[i]

    def classify_kernel(self, x):
        return self._kernel_machine(x)

    def classify(self, x):
        return np.sign(np.dot(self._w, x))

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.decomposition import PCA

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

    iris_classifier = Perceptron()
    iris_classifier.k = lambda x, xp: np.dot(x, xp)

    iris_classifier.train_kernel(np.array(train1_df)[:, :-1], np.array(train1_df)[:, -1] * 2 - 1)

    val  = np.array([iris_classifier.classify_kernel(x) for x in np.array(train1_df)[:, :-1]])

    valt= (np.array(train1_df, dtype=np.int)[:, -1] )
    unique = list(set(valt))
    valt[valt == max(unique)] = 1
    valt[valt == min(unique)] = -1

    Remp = 1.0/len(val) * np.sum(1/2*np.abs(val - valt))
    print(Remp)
    print(val)
    print(valt)

    D = np.array(test1_df)[:, :-1]
    y = np.array(test1_df)[:, -1]