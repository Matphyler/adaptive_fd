import numpy as np
import types
from typing import Callable


def _add_noise(obj: Callable, eps_f: float, enable_history=True) -> Callable:
    """
    Inject noise to f.

    Note that this modifies an instance at
    runtime and will override the eps_f, n_eval (and history, if
    enable_history = True) attributes of the object; it will also
    override the call method.

    :param obj:
    :param eps_f:
    :param enable_history:
    :return:
    """
    assert eps_f > 0, "eps_f must be positive"
    obj.eps_f = eps_f
    obj.n_eval = 0
    if enable_history:
        obj.history = []

    def call(self, x):

        self.n_eval += 1

        if enable_history:
            f_true = self.__call__(x)

            noise = self.eps_f * np.random.uniform(low=-1.0, high=1.0)

            f_noise = f_true + noise

            self.history.append({'x': x, 'f_true': f_true, 'f_noise': f_noise})

            return f_noise
        else:
            return self.__call__(x) + self.eps_f * np.random.uniform(low=-1.0, high=1.0)

    obj.call = types.MethodType(call, obj)

    return obj


def add_noise(eps_f: float, enable_history=True, obj=None):
    """
    Can either use as a decorator by calling:

    @add_noise(eps_f=1E-3, enable_history=True)
    def f(x):
        return x + 1

    or use as a function:
    add_noise(eps_f = 1E-3, obj = f)
    :param eps_f:
    :param enable_history:
    :return:
    """

    if obj is not None:
        return _add_noise(obj, eps_f=eps_f, enable_history=enable_history)

    else:

        def wrapper(obj):
            return _add_noise(obj, eps_f=eps_f, enable_history=enable_history)

        return wrapper
