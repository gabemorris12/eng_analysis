"""
Submodule for implementing the different integration methods.
"""
import numpy as np
from typing import Callable


def romberg(f: Callable, a: int | float, b: int | float, tol=1.0e-6) -> tuple:
    """
    Finds the romberg integration of f from a to b.

    :param f: Integrand function
    :param a: Lower limit
    :param b: Upper limit
    :param tol: Minimum tolerance till convergence
    :return: The evaluation and the number of panels used.
    """
    r = np.zeros(21)
    r[1] = _trapezoid(f, a, b, 0.0, 1)
    r_old = r[1]

    for k in range(2, 21):
        r[k] = _trapezoid(f, a, b, r[k - 1], k)
        r = _richardson(r, k)
        if abs(r[1] - r_old) < tol*max(abs(r[1]), 1.0):
            return r[1], 2**(k - 1)
        r_old = r[1]

    raise Exception('Romberg Quadrature failed to converge.')


def _richardson(r, k):
    for j in range(k - 1, 0, -1):
        const = 4.0**(k - j)
        r[j] = (const*r[j + 1] - r[j])/(const - 1.0)
    return r


def _trapezoid(f, a, b, I_old, k):
    if k == 1:
        I_new = (f(a) + f(b))*(b - a)/2.0
    else:
        n = 2**(k - 2)  # Number of new points
        h = (b - a)/n  # Spacing of new points
        x = a + h/2.0  # Coord. of 1st new point
        sum_ = 0.0
        for i in range(n):
            sum_ = sum_ + f(x)
            x = x + h
        I_new = (I_old + h*sum_)/2.0

    return I_new
