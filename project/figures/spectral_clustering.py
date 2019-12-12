'''
Ethan Lew
(elew@pdx.edu)

Shows the spectral clustering
'''

from fig_utils import *

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from scluster import SpectralClustering, KSpectralClustering
from kkmeans import KKernelClustering, k_polynomial, k_tanh, k_gaussian

sc = SpectralClustering(2, 100, lambda x,y: k_gaussian(x, y, .05))
sc_moon_labels = sc.train(noisy_moons[0])