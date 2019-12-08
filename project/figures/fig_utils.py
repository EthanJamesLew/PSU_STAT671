'''
Ethan Lew
(elew@pdx.edu)

Utilities to create figures for the presentations
'''

import numpy as np
from sklearn import datasets

# PARAMS
n_samples = 700
seed_val = 0

# make the datasets
np.random.seed(seed_val)
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

def ax_plot_clusters(ax, X, Y, **kwargs):
    '''
    Given data X and labels Y, plot the scatter plots on axis object as
    :param ax: plt.axis object
    :param X: data for clusters
    :param Y: labels
    :return: plt.axis object
    '''
    unique_labels = list(set(Y))
    for label in unique_labels:
        ax.scatter(*X[Y == label].T, label=str(label), **kwargs)
    return ax