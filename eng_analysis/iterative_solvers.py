"""
Submodule for implementing the different linear solver methods. These are the indirect methods.
"""

import numpy as np
from typing import Tuple, Callable, Iterable


def conj_grad(Av: Callable, x: Iterable, b: Iterable, tol=1.0e-9) -> Tuple[np.array, int]:
    """
    Solves the solution to Ax=b using the conjugate gradient method. The matrix A should be sparse.

    :param Av: A function that returns the matrix product of A and x
    :param x: The initial guess
    :param b: Solution vector
    :param tol: Error tolerance
    :return:
    """
    b = _handle_arrays(b)
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


def _handle_arrays(*args):
    """
    Makes sure that copies and numpy arrays with np.float64 datatypes are being used.

    :param args: Numpy arrays or lists to be handled
    :return: A list of copies of numpy arrays in args with np.float64 datatype
    """
    if len(args) > 1:
        return [np.array(item_, dtype=np.float64, copy=True) for item_ in args]
    else:
        return np.array(args, dtype=np.float64, copy=True)[0]
