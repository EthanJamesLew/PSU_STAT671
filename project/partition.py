import numpy as np

def random_uniform_partition(data, k):
    ''' randon_uniform partition
    For points belonging to data, assign a uniform number of labels to each one randomly

    :param data: (N x M) N data points on dimension M
    :param k: number of clusters
    :return: (N) membership vector
    '''
    M, N = data.shape
    Mp = int(np.ceil(M / k))
    membership = np.tile(np.arange(0, k, 1), Mp).astype(np.int32)
    np.random.shuffle(membership)
    return membership[:M]


def random_euclid_partition(data, k):
    ''' random_euclid_partition
    For pts belonging to some euclidean space of dimension N, partition in k clusters by using distance
    to randomly generated centroids

    :param data: (N x M) N data points on dimension M
    :param k: number of clusters
    :return: (N) membership vector
    '''
    # initialize dimensions, membership vectors
    M, N = data.shape
    membership = np.zeros(M, dtype=np.int32)

    # generate centroids
    bounds = np.array([[np.min(x), np.max(x)] for x in data.T])
    ranges = np.abs(bounds[:, 0] - bounds[:, 1])
    midpts = np.sum(bounds, axis=1)/2
    centroids = (np.random.rand(k, N)-0.5)*np.tile(ranges, (k, 1)) + np.tile(midpts, (k, 1))

    # get membership
    for idx, pt in enumerate(data):
        # generate (k x m) tiles of pt and get (k) norms
        dist = np.linalg.norm(np.tile(pt, (k, 1)) - centroids, axis=1)
        membership[idx] = np.argmin(dist)
    return membership

