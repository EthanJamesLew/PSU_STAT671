import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from fig_utils import *

from scluster import SpectralClustering
from kkmeans import KKernelClustering, k_polynomial, k_tanh, k_gaussian

# perform the clustering
print("Spectral Clustering...")
sc = SpectralClustering(2, 100, lambda x,y: k_gaussian(x, y, .05))
sc_moon_labels = sc.train(noisy_moons[0])
sc_circle_labels = sc.train(noisy_circles[0])
sc = SpectralClustering(3, 100, lambda x,y: k_gaussian(x, y, .05))
sc_blob_labels = sc.train(blobs[0])

# plot the data
fig, ax = plt.subplots(2, 3)
ax_plot_clusters(ax[0][0], *noisy_circles, s=4)
ax_plot_clusters(ax[0][1], *noisy_moons, s=4)
ax_plot_clusters(ax[0][2], *blobs, s=4)
ax_plot_clusters(ax[1][0], noisy_circles[0], sc_circle_labels, s=4)
ax_plot_clusters(ax[1][1], noisy_moons[0], sc_moon_labels, s=4)
ax_plot_clusters(ax[1][2], blobs[0], sc_blob_labels, s=4)
plt.setp(ax, xticks=[], yticks=[])
plt.show()