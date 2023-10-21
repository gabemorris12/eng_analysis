"""
Submodule for implementing the different linear solver methods. These are the direct methods.
"""
from typing import Union, List, Tuple, Iterable

import numpy as np

from ._handler import handle_arrays


def gauss_solve(A: Union[np.array, List], b: Union[np.array, List]) -> np.array:
    """
    Solves the solution to Ax=b using the gaussian elimination method. This does pivot for the cases where there is a
    0 along the diagonal of the matrix A.

    :param A: Matrix of coefficients
    :param b: Solution vector
    :return: The solution x of the linear system
    """
    A, b = handle_arrays(A, b)
    n = len(b)

    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(A[i, :]))

    for k in range(0, n - 1):
        # If needed, perform row interchange
        p = np.argmax(np.abs(A[k:n, k])/s[k:n]) + k
        if p != k:
            _row_interchange(b, k, p)
            _row_interchange(s, k, p)
            _row_interchange(A, k, p)

        # Elimination
        for i in range(k + 1, n):
            if A[i, k] != 0.0:
                lam = A[i, k]/A[k, k]
                A[i, k + 1:n] = A[i, k + 1:n] - lam*A[k, k + 1:n]
                b[i] = b[i] - lam*b[k]

    # Back Substitution
    b[n - 1] = b[n - 1]/A[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - np.dot(A[k, k + 1:n], b[k + 1:n]))/A[k, k]
    return b


def choleski_decomposition(A: Union[np.ndarray, List]) -> np.array:
    """
    Finds the Choleski decomposition of matrix A, such that [L][L]^T = A.

    Note:
    - A must be symmetric
    - A must be positive definite. If not, then an Assertion error will be raised.

    :param A: Matrix of coefficients
    :return: L, the choleski decomposition of A
    """
    A = handle_arrays(A)
    n = len(A)
    for k in range(n):
        value = A[k, k] - np.dot(A[k, 0:k], A[k, 0:k])
        assert value >= 0, "Matrix is not positive definite."
        A[k, k] = np.sqrt(value)
        for i in range(k + 1, n):
            A[i, k] = (A[i, k] - np.dot(A[i, 0:k], A[k, 0:k]))/A[k, k]
    for k in range(1, n): A[0:k, k] = 0.0
    return A


def choleski_solve(L: Union[np.ndarray, List], b: Union[np.ndarray, List]) -> np.array:
    """
    Solves the solution to LL^Tx=b with L being the Choleski decomposition of some matrix A. Use choleski_decomposition
    to find L prior to using this function.

    :param L: Choleski decomposition of some matrix A
    :param b: Solution vector
    :return: The solution x of the linear system
    """
    L, b = handle_arrays(L, b)
    n = len(b)
    # Solution of [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k, 0:k], b[0:k]))/L[k, k]
    # Solution of [L_transpose]{x} = {y}
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(L[k + 1:n, k], b[k + 1:n]))/L[k, k]
    return b


def lu_decomposition3(c: Iterable, d: Iterable, e: Iterable):
    """
    Performs the LU decomposition of a tridiagonal matrix. More details can be seen on page 60 of the book in the
    readme.

    :param c: Array of the lower diagonal
    :param d: Array of the diagonal
    :param e: Array of the upper diagonal
    :return: c, d, e --> The diagonals of the decomposed matrix
    """
    c, d, e = handle_arrays(c, d, e)
    n = len(d)
    for k in range(1, n):
        lam = c[k - 1]/d[k - 1]
        d[k] = d[k] - lam*e[k - 1]
        c[k - 1] = lam
    return c, d, e


def lu_solve3(c: Union[List, np.ndarray], d: Union[List, np.ndarray], e: Union[List, np.ndarray],
              b: Union[List, np.ndarray]):
    """
    Solves the solution Ax=b where c, d, and e are the vectors returned from lu_decomposition3.

    :param c: c vector from lu_decomposition3
    :param d: d vector from lu_decomposition3
    :param e: e vector from lu_decomposition3
    :param b: Solution vector
    :return: The solution x to Ax=b
    """
    b = handle_arrays(b)
    n = len(d)
    for k in range(1, n):
        b[k] = b[k] - c[k - 1]*b[k - 1]
    b[n - 1] = b[n - 1]/d[n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - e[k]*b[k + 1])/d[k]
    return b


def lu_decomposition(A: Union[np.array, List]) -> Tuple[np.array, np.array]:
    """
    Performs LU decomposition with pivoting to handle zero's on diagonal.

    :param A: Matrix of coefficients
    :return: LU, seq where LU contains U in the upper triangle portion and the non-diagonal terms of L in the lower
             triangle. The permutations are recorded in the vector "seq."
    """
    A = handle_arrays(A)
    n = len(A)
    seq = np.arange(n)

    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(abs(A[i, :]))

    for k in range(0, n - 1):

        # Row interchange, if needed
        p = np.argmax(np.abs(A[k:n, k])/s[k:n]) + k
        if p != k:
            _row_interchange(s, k, p)
            _row_interchange(A, k, p)
            _row_interchange(seq, k, p)

        # Elimination
        for i in range(k + 1, n):
            if A[i, k] != 0.0:
                lam = A[i, k]/A[k, k]
                A[i, k + 1:n] = A[i, k + 1:n] - lam*A[k, k + 1:n]
                A[i, k] = lam
    return A, seq


def lu_solve(LU: Union[np.array, List], b: Union[np.array, List], seq: Union[np.array, List]) -> np.array:
    """
    Solves the solution to LUx=b with LU being the LU decomposition of some matrix A. Use lu_decomposition to find LU
    prior to using this function.

    :param LU: The LU matrix returned from lu_decomposition
    :param b: Solution vector
    :param seq: The recorded permutations from lu_decomposition
    :return: The solution x of the linear system
    """
    LU, b = handle_arrays(LU, b)
    n = len(LU)

    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    # Solution
    for k in range(1, n):
        x[k] = x[k] - np.dot(LU[k, 0:k], x[0:k])
    x[n - 1] = x[n - 1]/LU[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (x[k] - np.dot(LU[k, k + 1:n], x[k + 1:n]))/LU[k, k]
    return x


def _row_interchange(v, i, j):
    """
    Performs a row interchange by taking in a numpy array "v" and swapping rows "i" and "j".

    :param v: Numpy array
    :param i: A row in v
    :param j: A row in v
    """
    if len(v.shape) == 1:
        v[i], v[j] = v[j], v[i]
    else:
        v[[i, j], :] = v[[j, i], :]


if __name__ == '__main__':
    A_ = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 1],
        [0, -1, 2, -1],
        [-1, 2, -1, 0]
    ])
    b_ = np.array([1, 0, 0, 0])
    print(gauss_solve(A_, b_))
    print(np.linalg.solve(A_, b_))
