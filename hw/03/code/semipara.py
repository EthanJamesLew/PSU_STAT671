import pandas as pd
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from krr import *

def g(x, xp, alpha, k):
    total = 0
    for xi, ai in zip(xp, alpha):
        total += ai * k(x, xi)
    return total

data_df = pd.read_csv("hmw3-data1.csv")
x = np.array(data_df['x'])
y = np.array(data_df['y'])
pts = np.vstack((x, y)).T
l = 10
k = lambda x, y: min(x,y)
K = kernel_mat(k, x)
X = np.vstack((x*x, x, np.ones(len(x)))).T
theta = np.zeros(3)
alpha = np.zeros(len(x))

theta = theta[:, np.newaxis]
alpha = alpha[:, np.newaxis]
y = y[:, np.newaxis]
Xt = X @ la.inv(X.T @ X) @ X.T
alpha = la.inv(-Xt @ K - Xt * l + K) @ (y - Xt @ y)
theta = la.inv(X.T @ X) @ X.T @ (y - K @ alpha)

x_r =  np.linspace(0, 7, 100)
y_kr = np.array([g(xi, x, alpha, k) for xi in x_r])
y_gr = np.array([xi*xi*theta[0] + xi*theta[1] + theta[2] for xi in x_r])
y_r = np.array([g(xi, x, alpha, k) + xi*xi*theta[0] + xi*theta[1] + theta[2] for xi in x_r])

plt.figure()
plt.scatter(*pts.T, c='k', label='observations')
plt.plot(x_r, y_r, label='k = min(x,y)')
plt.plot(x_r, y_kr, '--', label='kernel regression')
plt.plot(x_r, y_gr, '--', label='parametric regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Semiparametric Regression')
plt.legend()
plt.savefig('semipara.eps')
plt.show()