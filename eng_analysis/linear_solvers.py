"""
Submodule for implementing the different linear solver methods. These are the direct methods.
"""
import numpy as np
from typing import Union, List, Tuple


def gauss_solve(A: Union[np.array, List], b: Union[np.array, List]) -> np.array:
    """
    Solves the solution to Ax=b using the gaussian elimination method. This does pivot for the cases where there is a
    0 along the diagonal of the matrix A.

    :param A: Matrix of coefficients
    :param b: Solution vector
    :return: The solution x of the linear system
    """
    A, b = _handle_arrays(A, b)
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
    A = _handle_arrays(A)
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
    L, b = _handle_arrays(L, b)
    n = len(b)
    # Solution of [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k, 0:k], b[0:k]))/L[k, k]
    # Solution of [L_transpose]{x} = {y}
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(L[k + 1:n, k], b[k + 1:n]))/L[k, k]
    return b


def lu_decomposition(A: Union[np.array, List]) -> Tuple[np.array, np.array]:
    """
    Performs LU decomposition with pivoting to handle zero's on diagonal.

    :param A: Matrix of coefficients
    :return: LU, seq where LU contains U in the upper triangle portion and the non-diagonal terms of L in the lower
             triangle. The permutations are recorded in the vector "seq."
    """
    A = _handle_arrays(A)
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
    LU, b = _handle_arrays(LU, b)
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
