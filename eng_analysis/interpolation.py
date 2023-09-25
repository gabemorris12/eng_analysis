"""
Submodule for implementing the different interpolation methods.
"""
from typing import Iterable, Union, List

import numpy as np

from .linear_solvers import lu_decomposition3, lu_solve3


class NewtonPoly:
    def __init__(self, x: Iterable, y: Iterable):
        """
        Returns a callable object that evaluates the Newton polynomial at the point(s).

        :param x: x data
        :param y: y data

        Attributes:
        - coefficients: The coefficients (a) of the polynomial in the form of a0 + (x − x0)a1 + (x − x0)(x − x1)a2 +···+
                        (x − x0)(x − x1)···(x − x_n−1)a_n
        """
        self.x, self.y = _handle_arrays(x, y)
        self.coefficients = self._get_coefficients()

    def _get_coefficients(self):
        m = len(self.x)  # Number of data points
        a = self.y.copy()
        for k in range(1, m):
            a[k:m] = (a[k:m] - a[k - 1])/(self.x[k:m] - self.x[k - 1])
        return a

    def __call__(self, x: Iterable | int | float) -> np.array:
        x = _handle_arrays(x)
        n = len(self.x) - 1  # Degree of polynomial
        p = self.coefficients[n]
        for k in range(1, n + 1):
            p = self.coefficients[n - k] + (x - self.x[n - k])*p
        return p


class CubicSpline:
    def __init__(self, x: Iterable, y: Iterable):
        """
        Returns a callable object that evaluates the natural cubic spline value at the point(s).

        :param x: x data
        :param y: y data

        Attributes:
        - curvatures: The curvatures of the cubic spline at its knots.
        """
        self.x, self.y = _handle_arrays(x, y)
        self.curvatures = self._curvatures()

    def _curvatures(self):
        n = len(self.x) - 1
        c = np.zeros(n)
        d = np.ones(n + 1)
        e = np.zeros(n)
        k = np.zeros(n + 1)
        c[0:n - 1] = self.x[0:n - 1] - self.x[1:n]
        d[1:n] = 2.0*(self.x[0:n - 1] - self.x[2:n + 1])
        e[1:n] = self.x[1:n] - self.x[2:n + 1]
        k[1:n] = 6.0*(self.y[0:n - 1] - self.y[1:n])/(self.x[0:n - 1] - self.x[1:n]) - 6.0*(
                self.y[1:n] - self.y[2:n + 1])/(self.x[1:n] - self.x[2:n + 1])
        c, d, e = lu_decomposition3(c, d, e)
        k = lu_solve3(c, d, e, k)
        return k

    def __call__(self, x: Union[List, np.ndarray, int, float]):
        i = _find_segment(self.x, x)
        h = self.x[i] - self.x[i + 1]
        y = ((x - self.x[i + 1])**3/h - (x - self.x[i + 1])*h)*self.curvatures[i]/6.0 - (
                    (x - self.x[i])**3/h - (x - self.x[i])*h)*self.curvatures[i + 1]/6.0 + (
                        self.y[i]*(x - self.x[i + 1]) - self.y[i + 1]*(x - self.x[i]))/h
        return y


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


def _find_segment(x_data, x):
    left = 0
    right = len(x_data) - 1
    while True:
        if (right - left) <= 1: return left
        i = int((left + right)/2)
        if x < x_data[i]:
            right = i
        else:
            left = i
