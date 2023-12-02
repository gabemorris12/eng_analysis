"""
Submodule for functions that determine eigenvalues and eigenvectors.
"""
import numpy as np
from typing import List, Iterable
from ._handler import handle_arrays


def _threshold(a):
    n = len(a)
    sum_ = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sum_ = sum_ + abs(a[i, j])
    return 0.5*sum_/n/(n - 1)


def _rotate(a, p, k, l):  # Rotate to make a[k,l] = 0
    n = len(a)
    aDiff = a[l, l] - a[k, k]

    if abs(a[k, l]) < abs(aDiff)*1.0e-36:
        t = a[k, l]/aDiff
    else:
        phi = aDiff/(2.0*a[k, l])
        t = 1.0/(abs(phi) + np.sqrt(phi**2 + 1.0))
        if phi < 0.0: t = -t
    c = 1.0/np.sqrt(t**2 + 1.0)
    s = t*c
    tau = s/(1.0 + c)
    temp = a[k, l]
    a[k, l] = 0.0
    a[k, k] = a[k, k] - t*temp
    a[l, l] = a[l, l] + t*temp

    for i in range(k):  # Case of i < k
        temp = a[i, k]
        a[i, k] = temp - s*(a[i, l] + tau*temp)
        a[i, l] = a[i, l] + s*(temp - tau*a[i, l])
    for i in range(k + 1, l):  # Case of k < i < l
        temp = a[k, i]
        a[k, i] = temp - s*(a[i, l] + tau*a[k, i])
        a[i, l] = a[i, l] + s*(temp - tau*a[i, l])
    for i in range(l + 1, n):  # Case of i > l
        temp = a[k, i]
        a[k, i] = temp - s*(a[l, i] + tau*temp)
        a[l, i] = a[l, i] + s*(temp - tau*a[l, i])
    for i in range(n):  # Update transformation matrix
        temp = p[i, k]
        p[i, k] = temp - s*(p[i, l] + tau*p[i, k])
        p[i, l] = p[i, l] + s*(temp - tau*p[i, l])


# Todo: I think this needs some work. Ensure that this works properly, then add docs.
def jacobi(a: List | Iterable, tol=1.0e-8):
    a = handle_arrays(a)
    n = len(a)
    p = np.identity(n, float)
    for k in range(20):
        mu = _threshold(a)  # Compute new threshold
        for i in range(n - 1):  # Sweep through matrix
            for j in range(i + 1, n):
                if abs(a[i, j]) >= mu:
                    _rotate(a, p, i, j)
        if mu <= tol: return np.diagonal(a), p

    raise Exception('Jacobi method did not converge')
