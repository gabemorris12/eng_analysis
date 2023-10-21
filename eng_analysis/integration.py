"""
Submodule for implementing the different integration methods.
"""
from typing import Callable, Iterable, List

import numpy as np


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


def gauss_legendre2(f: Callable, x: Iterable | List, y: Iterable | List, m: int) -> float:
    """
    Computes the Gauss-Legendre integration of a multivariable function, f(x, y), over a quadrilateral of the order "m".

    :param f: Integrand function of two variables
    :param x: The x values of the corner coordinates of the quadrilateral
    :param y: The y values of the corner coordinates of the quadrilateral
    :param m: The order of integration
    :return: The evaluation of the integral
    """
    s, A = _legendre_weights(m)
    sum_ = 0.0
    for i in range(m):
        for j in range(m):
            xCoord, yCoord = _map(x, y, s[i], s[j])
            sum_ = sum_ + A[i]*A[j]*_jac(x, y, s[i], s[j])*f(xCoord, yCoord)
    return sum_


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


def _legendre_weights(m, tol=1e-12):
    A = np.zeros(m)
    x = np.zeros(m)
    nRoots = int((m + 1)/2)  # Number of non-neg. roots
    for i in range(nRoots):
        t = np.cos(np.pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p, dp = _legendre(t, m)  # Newton-Raphson
            dt = -p/dp
            t = t + dt  # method
            if abs(dt) < tol:
                x[i] = t
                x[m - i - 1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2)  # Eq.(6.25)
                A[m - i - 1] = A[i]
                break
    return x, A


def _legendre(t, m):
    p0 = 1.0
    p1 = t
    for k in range(1, m):
        p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k)
        p0 = p1
        p1 = p
    dp = m*(p0 - t*p1)/(1.0 - t**2)
    # noinspection PyUnboundLocalVariable
    return p, dp


def _jac(x, y, s, t):
    J = np.zeros((2, 2))
    J[0, 0] = -(1.0 - t)*x[0] + (1.0 - t)*x[1] + (1.0 + t)*x[2] - (1.0 + t)*x[3]
    J[0, 1] = -(1.0 - t)*y[0] + (1.0 - t)*y[1] + (1.0 + t)*y[2] - (1.0 + t)*y[3]
    J[1, 0] = -(1.0 - s)*x[0] - (1.0 + s)*x[1] + (1.0 + s)*x[2] + (1.0 - s)*x[3]
    J[1, 1] = -(1.0 - s)*y[0] - (1.0 + s)*y[1] + (1.0 + s)*y[2] + (1.0 - s)*y[3]
    return (J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0])/16.0


def _map(x, y, s, t):
    N = np.zeros(4)
    N[0] = (1.0 - s)*(1.0 - t)/4.0
    N[1] = (1.0 + s)*(1.0 - t)/4.0
    N[2] = (1.0 + s)*(1.0 + t)/4.0
    N[3] = (1.0 - s)*(1.0 + t)/4.0
    xCoord = np.dot(N, x)
    yCoord = np.dot(N, y)
    return xCoord, yCoord


if __name__ == '__main__':
    f_lamb = lambda x_, y_: (x_ - 2)**2*(y_ - 2)**2
    x_values = [0, 4, 4, 1]
    y_values = [0, 1, 4, 3]
    print(gauss_legendre2(f_lamb, x_values, y_values, 3))
    print(x_values)
