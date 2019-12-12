'''
@author: Ethan Lew
(elew@pdx.edu)

Implement features to generate convenient synthetic data to demonstrate clustering algorithm efficacy
'''

import numpy as np

def generate_ring_2d(N, mu=0, sigma=1):
    ''' generate_ring

    Generates data points along a ring at radius mu and standard deviation sigma
    :param N: Number of points to generate
    :param mu: Average Ring radius
    :param sigma: Ring radius standard deviation
    :return: [N x 2] 2D points
    '''
    X = np.vstack((np.random.randn(N)*sigma+mu, np.random.randn(N)*2*np.pi)).T
    X = np.array([np.array([x[0]*np.cos(x[1]), x[0]*np.sin(x[1])]) for x in X])
    return X

def generate_cluster_2d(N, pos = [0, 0], posstd = [1,1]):
    ''' generate_cluster
    :param N: Number of points to generate
    :param pos: (x, y) Average position of cluster
    :param posstd: (x, y) Position standard deviation
    :return: [N x 2] 2D points
    '''
    return np.vstack((np.random.randn(N)*posstd[0] + pos[0], np.random.randn(N)*posstd[1] + pos[1])).T



