"""
Submodule for implementing the different root finding methods.
"""
from typing import Callable, Iterable

import numpy as np

from .linear_solvers import gauss_solve
from ._handler import handle_arrays


def newton_raphson(f: Callable, x: Iterable, tol=1.0e-9, max_iter=30):
    """
    Returns the solution to a system of linear or non-linear functions using the Newton Raphson method.

    :param f: A function that returns the equation(s) that are equal to zero. Must have the call: f(x) where x is an
              array if f is a system of equations and x is a float or int for a single equation.
    :param x: Input guess value. Must match the size of the input "x" to f.
    :param tol: Error tolerance
    :param max_iter: If the error tolerance does not converge due to invalid guess input or no solution, then the
                     iterative process will max out at a value of max_iter and a TooManyIterations error will occur.
    :return:
    """
    x = handle_arrays(x)
    i = 0
    while True:
        jac, f0 = _jacobian(f, x)
        if np.sqrt(np.dot(f0, f0)/len(x)) < tol:
            return x
        dx = gauss_solve(jac, -f0)
        x = x + dx
        if np.sqrt(np.dot(dx, dx)) < tol*max(max(abs(x)), 1.0): return x
        i += 1
        if max_iter < i: raise TooManyIterations('Exceeded the limited amount of iterations.')


def _jacobian(f, x):
    h = 1.0e-4
    n = len(x)
    jac = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        temp = x[i]
        x[i] = temp + h
        f1 = f(x)
        x[i] = temp
        jac[:, i] = (f1 - f0)/h
    return jac, f0


class TooManyIterations(Exception):
    pass
