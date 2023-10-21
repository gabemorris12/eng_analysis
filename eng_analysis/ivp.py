"""
Submodule for implementing the different initial value problems.
"""
from typing import Callable, List, Iterable


def runge_kutta(func: Callable, y0: List | Iterable, t: List | Iterable) -> List:
    """
    Fourth order Runge-Kutta method.

    :param func: func(y, t); A function that returns the first order derivatives of the system.
    :param y0: Initial conditions of y.
    :param t: Time points of y where the first value corresponds to the initial conditions.
    :return: The numerical solution of the ODE.
    """
    n = len(func(y0, t[0]))
    arrays = [[y0_] for y0_ in y0]

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        K1 = func([array[i] for array in arrays], t[i])
        K2 = func([array[i] + 0.5*h*K for array, K in zip(arrays, K1)], t[i] + 0.5*h)
        K3 = func([array[i] + 0.5*h*K for array, K in zip(arrays, K2)], t[i] + 0.5*h)
        K4 = func([array[i] + h*K for array, K in zip(arrays, K3)], t[i] + h)

        for j in range(n):
            arrays[j].append(arrays[j][i] + h/6*(K1[j] + 2*K2[j] + 2*K3[j] + K4[j]))

    return arrays
