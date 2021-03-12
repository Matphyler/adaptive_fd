import numpy as np
import sympy as sp
from typing import Optional


def adaptive_fd(obj, x: float, max_iter: int = 30, r_lower: float = 5., r_upper: float = 10.,
                h_init: Optional[float] = None, h_max: float = 1E10, h_min: float = 1E-16):

    eps_f = obj.eps_f
    assert eps_f > 0., "eps_f must be positive"
    assert max_iter > 0, "max iter must be positive"

    if h_init is None:
        h = 2.0 * np.sqrt(eps_f)
    else:
        h = h_init

    u = np.inf
    l = 0.

    flag = 0
    fx = obj.call(x)
    fwd = np.nan
    bwd = np.nan
    n_iter = 0

    for n_iter in range(max_iter):

        fwd = obj.call(x + h) - fx  # forward
        bwd = fx - obj.call(x - h)  # backward

        r = np.abs(fwd - bwd) / eps_f

        if r < r_lower:
            l = h
        elif r > r_upper:
            u = h
        else:
            break

        if np.isinf(u):
            h_new = h * 2
        else:
            h_new = (l + u) / 2.

        if h_new > h_max or h_new < h_min:
            flag = -2  # out of bound
            break
        else:
            h = h_new
    else:
        flag = -1  # max iter reached

    return {'LS_flag': flag, 'h': h, 'fwd_grad': fwd / h, 'bwd_grad': bwd / h, 'n_iter': n_iter + 1}
