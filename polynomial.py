import numpy as np
from typing import Union, Optional, Tuple
import math
import re
import click
import logging
import sys


class Polynomial:

    def __init__(self, coeff):
        coeff = np.array([i for i in coeff], dtype=np.float)
        self.order: int = coeff.shape[0] - 1
        self.coeff: np.ndarray = coeff
        self._repr_str: Optional[str] = None
        self._d_coeff = {0: self.coeff}

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_ = np.asarray(x)
        return np.power.outer(x_, np.arange(self.order + 1)).dot(self.coeff)

    @staticmethod
    def _monomial_str(o: int, c: float) -> Tuple[bool, str]:
        sign = True if c >= 0 else False
        if c == 0.:
            return True, ""
        else:
            if np.abs(c) > 1E2 or np.abs(c) < 1E-2:  # use E notation
                c_str = "{:.2e}".format(np.abs(c), o)
            else:
                c_str = "{:.2f}".format(np.abs(c))

            if o == 0:
                x_str = ""
            elif o == 1:
                x_str = "x"
            else:
                x_str = f"x^{o}"

            return sign, f"{c_str} {x_str}".strip()

    def d_coeff(self, n: int = 0):
        if not isinstance(n, int) and n >= 0:
            raise ValueError("n must be a non-negative integer!")
        if n == 0:
            return self.coeff
        elif n not in self._d_coeff:
            coeff = self.d_coeff(n - 1)
            coeff = coeff * np.arange(len(coeff))
            self._d_coeff[n] = coeff[1:]
        return self._d_coeff[n]

    def __repr__(self) -> str:
        if self._repr_str is None:
            s = ""
            for o, c in enumerate(self.coeff):
                sign, mono_str = self._monomial_str(o, c)
                if not mono_str:
                    continue
                sign_str = "+" if sign else "-"
                if (not s) and sign:  # at the beginning and positive sign
                    s += f"{mono_str} "  # omit the sign
                else:
                    s += f"{sign_str} {mono_str} " # include the sign

            self._repr_str = s.strip()
        return self._repr_str

    def to_latex(self) -> str:

        sign_pattern = re.compile(r"e(\+0*)")

        e_pattern = re.compile(r"([\d\.]+)e([\-]*\d+)\s")

        s = self.__repr__()

        s = re.sub(
            pattern=sign_pattern,
            repl=r"e",
            string=s
        )

        return re.sub(
            pattern=e_pattern,
            repl=r" \1 \\times 10^{\2} ",
            string=s
        ).strip()

    def D(self, n: int = 1) -> "Polynomial":
        """
        n-th order derivatives
        :param n: order of derivative; must be non-negative integer; if n=0, will return self.
        :return:
        """
        if not isinstance(n, int) and n >= 0:
            raise ValueError("n must be a non-negative integer!")

        if n == 0:
            return self
        else:
            return Polynomial(self.d_coeff(n))

    def _test_derivatives(self, x: float, n: int = 3, eps: float = 1E-8):

        """
        Test derivatives by computing the n-th order taylor expansion at x, evaluate at x + eps, and then compare
        with f(x + eps).
        :param x:
        :param n:
        :param eps:
        :return:
        """

        if not isinstance(n, int) and n >= 1:
            raise ValueError("n must be a positive integer!")

        d_array = np.array([self.D(n=i).__call__(x) for i in range(n+1)]) \
                  / np.array([math.factorial(i) for i in np.arange(n + 1)])

        eps_array = eps ** np.arange(n + 1)

        return self.__call__(x + eps) - d_array.dot(eps_array)

@click.command()
@click.option("-o", "--order", type=int, help="order of polynomial")
@click.option("-l", "--lower_bound", type=float, default=-1.0, help="lower bound of the coefficient of the polynomial")
@click.option("-u", "--upper_bound", type=float, default=1.0, help="upper bound of the coefficient of the polynomial")
@click.option("-n", "--taylor_order", type=int, default=2, help="order of Taylor model")
def main(order: int, lower_bound: float, upper_bound: float, taylor_order: int):

    logger = logging.getLogger("polynomial")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)

    coeff = np.random.uniform(low=lower_bound, high=upper_bound, size=order+1)
    poly = Polynomial(coeff)
    x = np.random.uniform(low=lower_bound, high=upper_bound)
    logger.info(f"f(x) = {poly.__repr__()}")
    logger.info(f"x = {x}")
    for o in reversed(range(-8, 0)):
        logger.info("1E{}\t{:+.3E}\t{:+.3E}\t{:+.3E}\t{:+.3E}".format(o, poly._test_derivatives(x, 1, eps=10 ** o), poly._test_derivatives(x, 2, eps=10 ** o), poly._test_derivatives(x, 3, eps=10 ** o), poly._test_derivatives(x, 4, eps=10 ** o)))


if __name__ == "__main__":
    main()
