"""
Submodule for implementing the different linear solver methods. These are the indirect methods.
"""

import numpy as np
from typing import Tuple, Callable, Iterable

from ._handler import handle_arrays


def conj_grad(Av: Callable, x: Iterable, b: Iterable, tol=1.0e-9) -> Tuple[np.array, int]:
    """
    Solves the solution to Ax=b using the conjugate gradient method. The matrix A should be sparse.

    :param Av: A function that returns the matrix product of A and x
    :param x: The initial guess
    :param b: Solution vector
    :param tol: Error tolerance
    :return:
    """
    b = handle_arrays(b)
    n = len(b)
    r = b - Av(x)
    s = r.copy()

    for i in range(n):
        u = Av(s)
        alpha = np.dot(s, r)/np.dot(s, u)
        x = x + alpha*s
        r = b - Av(x)

        if np.sqrt(np.dot(r, r)) < tol:
            break
        else:
            beta = -np.dot(r, u)/np.dot(s, u)
            s = r + beta*s

    # noinspection PyUnboundLocalVariable
    return x, i
