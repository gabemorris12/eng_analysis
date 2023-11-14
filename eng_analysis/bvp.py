"""
Submodule for implementing the different boundary value problems.
"""
from scipy.integrate import solve_bvp


def bol_stoer(fun, bc, x, y, **kwargs):
    sol = solve_bvp(fun, bc, x, y, **kwargs)
    return sol.sol(x)
