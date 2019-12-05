import numpy as np
import math as mth
def check_kernel(k, A, X):
    total = 0
    for xi, ai in zip(X, A):
        for xj, aj in zip(X,A):
            total += ai*aj*k(xi, xj)
    return total

def disprove_cos_xpy():
    PI = mth.pi
    k = lambda x,y: mth.cos(x + y)
    A = [-1, 1]
    X = [PI/4, 0]
    return check_kernel(k, A, X)

#k = lambda x,y: min(x, y)/max(x, y)
k = lambda x,y: mth.log(1 + x*y)
A = [0.5, -1]
X = [20, 1]
print(check_kernel(k, A, X))
#print(disprove_cos_xpy())
