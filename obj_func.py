import sympy as sp
import numpy as np
from matplotlib import pyplot as plt, rc_context
from typing import Optional


class ObjFunc:
    def __init__(self, expr: sp.core.expr.Expr, eps_f: float) -> None:

        if len(expr.free_symbols) != 1:
            raise ValueError("expr must be a SymPy expression with only one free symbol.")
        self.x = next(iter(expr.free_symbols))

        if eps_f <= 0:
            raise ValueError("eps must be a positive float.")
        self.eps_f = eps_f

        self.f_expr = expr
        self.f = sp.lambdify(self.x, self.f_expr, "numpy")

        self.g_expr = sp.diff(self.f_expr, self.x)
        self.g = sp.lambdify(self.x, self.g_expr, "numpy")

        self.hess_expr = sp.diff(self.g_expr, self.x)
        self.hess = sp.lambdify(self.x, self.hess_expr, "numpy")

        self.num_eval = 0
        self.call_history = []

        self._last_ls_history = []

    def __call__(self, x: float) -> float:
        """
        Evaluate the function at x, with noise. This will increase the
        counter self.num_eval, and will append the results in self.call_history.
        :param x: float, the value at which to evaluate the function
        :return: noisy function
        """

        self.num_eval += 1

        noise = self.eps_f * np.random.uniform(low=-1.0, high=1.0)
        f_true = self.f(x)
        f_noise = f_true + noise

        self.call_history.append({'x': x, 'noise': noise, 'f_true': f_true, 'f_noise': f_noise})

        return self.f(x) + noise

    def eps_g(self, x: float, h: float, mode='forward'):
        return np.abs((self.f(x + h) - self.f(x)) / h - self.g(x)) + 2 * self.eps_f / h

    def reset(self) -> None:
        self.num_eval = 0
        self.call_history = []

    def adafd(self, x: float,
              mode: str = 'forward',
              max_iter: int = 30,
              r_lower: float = 3.,
              r_upper: float = 7.,
              h_init: Optional[float] = None,
              h_max: float = 1E10,
              h_min: float = 1E-16):

        eps_f = self.eps_f
        assert eps_f > 0., "eps_f must be positive"
        assert max_iter > 0, "max iter must be positive"

        if mode[0] in {'f', 'F'}:
            forward_mode = True
        elif mode[0] in {'c', 'C'}:
            forward_mode = False
        else:
            raise ValueError("only forward and central modes are supported")

        self._last_ls_history = []
        num_eval = self.num_eval

        if h_init is None:
            h = 2.0 * np.sqrt(eps_f)
        else:
            h = h_init

        u = np.inf
        l = 0.

        flag = 0
        n_iter = 0

        fh, f2h, fhf, fhb = np.nan, np.nan, np.nan, np.nan

        f0 = self.__call__(x)  # f(x)

        if forward_mode:
            fh = self.__call__(x + h)  # f(x+h)
            f2h = self.__call__(x + 2 * h)  # f(x+2h)
        else:
            fhf = self.__call__(x + h)
            fhb = self.__call__(x - h)

        for n_iter in range(max_iter):

            if forward_mode:
                r = np.abs(f2h - 2. * fh + f0) / (2. * eps_f)

                self._last_ls_history.append(
                    dict(
                        r=r,
                        h=h,
                        fh=fh,
                        f2h=f2h,
                        u=u,
                        l=l
                    )
                )
            else:
                r = np.abs(fhf - 2. * f0 + fhb) / (2. * eps_f)
                self._last_ls_history.append(
                    dict(
                        r=r,
                        h=h,
                        fhf=fhf,
                        fhb=fhb,
                        u=u,
                        l=l
                    )
                )

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

            if not forward_mode:
                fhf = self.__call__(x + h)
                fhb = self.__call__(x - h)
            else:
                if np.isinf(u):
                    fh = f2h
                    f2h = self.__call__(x + 2*h)
                elif l == 0.:
                    f2h = fh
                    fh = self.__call__(x + h)
                else:
                    fh = self.__call__(x + h)
                    f2h = self.__call__(x + 2*h)

        else:
            flag = -1  # max iter reached

        if forward_mode:
            g_est = (fh - f0) / h
        else:
            g_est = (fhf - f0) / h

        g_acc = self.g(x)
        eps_g = np.abs(g_est - g_acc)

        return {'LS_flag': flag, 'h': h, 'n_iter': n_iter + 1, 'g_est': g_est, 'g_acc': g_acc, 'eps_g': eps_g, 'r': r,
                'num_eval': self.num_eval - num_eval}

    def gen_plot(self, x: float, h_lower=1E-5, h_upper=10., base=1.1, dpi=400, extra_args=None):

        if extra_args is None:
            extra_args = {}

        cmode = self.adafd(x=x, mode='C', **(extra_args.get('forward', {})))
        fmode = self.adafd(x=x, mode='F', **(extra_args.get('central', {})))

        index_lower = int(np.floor(np.log(h_lower) / np.log(base)))
        index_upper = int(np.ceil(np.log(h_upper) / np.log(base)))

        h = base ** np.arange(index_lower, index_upper + 1)
        eps_g = np.array([self.eps_g(x=x, h=i) for i in h])

        h_th = 2 * np.sqrt(self.eps_f / np.abs(self.hess(x)))

        with rc_context(rc={'figure.dpi': dpi}):
            fig, ax = plt.subplots()

            color = next(ax._get_lines.prop_cycler)['color']
            ax.axvline(fmode['h'], linestyle='--', color=color, label="adaFD (forward mode, h={:.2E})".format(fmode['h']))
            ax.scatter([fmode['h']], [self.eps_g(x=x, h=fmode['h'])], color=color)

            color = next(ax._get_lines.prop_cycler)['color']
            ax.axvline(cmode['h'], linestyle='--', color=color, label="adaFD (central mode, h={:.2E})".format(cmode['h']))
            ax.scatter([cmode['h']], [self.eps_g(x=x, h=cmode['h'])], color=color)

            color = next(ax._get_lines.prop_cycler)['color']
            ax.axvline(h_th, linestyle='--', color=color,
                       label="computed by L (h={:.2E})".format(h_th))
            ax.scatter([h_th], [self.eps_g(x=x, h=h_th)], color=color)

            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(h, eps_g, label='worst case', color=color)

            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"$f(x) = {sp.latex(self.f_expr)}, x={x}, \\epsilon_f = {self.eps_f}$")
            plt.xlabel("finite difference interval $h$")
            plt.ylabel("$\\epsilon_g$")
            plt.legend()

        return h, eps_g