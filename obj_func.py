import math
import sys
from typing import Optional

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt, rc_context

from ada_est_res import AdaEstRes
from estimating_scheme import EstimatingScheme


class ObjFunc:

    @staticmethod
    def _deterministic_noise(x: float) -> float:
        """
        Return noise between [-1, 1] that is
        deterministic in x
        :param x:
        :return:
        """
        def _my_hash(z):
            magic_number = 114509728
            z_hash = z.__hash__()
            return (z_hash ** 2 + magic_number) % sys.hash_info.modulus

        x_hash = x
        for i in range(10):
            x_hash = _my_hash(x_hash)

        return 2. * (x_hash / sys.hash_info.modulus) - 1.

    def __init__(self, expr: sp.core.expr.Expr, eps_f: float, enable_cache: bool = True,
                 deterministic: bool = True) -> None:

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
        self.num_eval_no_cache = 0
        # self.call_history = []

        self._last_ls_history = []

        self.enable_cache = enable_cache
        self._cache = {}

        self.deterministic = deterministic

    def _clear_cache(self) -> None:
        self._cache = {}

    def __call__(self, x: float) -> float:
        """
        Evaluate the function at x, with noise. This will increase the
        counter self.num_eval, and will append the results in self.call_history.
        :param x: float, the value at which to evaluate the function
        :return: noisy function
        """

        self.num_eval_no_cache += 1

        if self.enable_cache:
            if x in self._cache:
                return self._cache[x]

        self.num_eval += 1

        if not self.deterministic:
            noise = self.eps_f * np.random.uniform(low=-1.0, high=1.0)
        else:
            # deterministic noise
            raw_noise = self._deterministic_noise(x)
            noise = self.eps_f * raw_noise

        # f_true = self.f(x)
        # f_noise = f_true + noise

        # self.call_history.append({'x': x, 'noise': noise, 'f_true': f_true, 'f_noise': f_noise})

        res = self.f(x) + noise

        if self.enable_cache:
            self._cache[x] = res

        return res

    def ada_est(self,
                x: float,
                scheme: EstimatingScheme,
                max_iter: int = 30,
                eps_f: Optional[float] = None,
                h_max: float = 1E10,
                h_min: float = 1E-16,
                r_lower: Optional[float] = None,
                r_upper: Optional[float] = None,
                save_history: bool = True,
                ) -> AdaEstRes:

        if eps_f is None:
            eps_f = self.eps_f

        if scheme.h_init_func is None:
            h_init = eps_f
        else:
            h_init = scheme.h_init_func(eps_f)

        est_order: int = scheme.est_order
        if r_upper is None:
            r_upper = scheme.r_upper
        if r_lower is None:
            r_lower = scheme.r_lower

        h_opt = self.optimal_h(x=x, scheme=scheme, eps_f=eps_f)

        if save_history:
            history = []
        else:
            history = None

        num_eval = self.num_eval
        num_eval_no_cache = self.num_eval_no_cache

        if eps_f is None:
            eps_f = self.eps_f

        assert eps_f > 0., "eps_f must be positive"
        assert max_iter > 0, "max iter must be positive"

        if est_order is not None:
            acc = sp.lambdify(self.x, sp.diff(self.f_expr, self.x, est_order), "numpy")(x)
        else:
            acc = None

        h = h_init

        u = np.inf
        l = 0.

        flag = 0
        n_iter = 0

        if self.enable_cache:
            self._clear_cache()

        f_vals = np.array([self.__call__(x + s * h) for s in scheme.r_shift])

        r = np.nan

        for n_iter in range(max_iter):

            r = np.abs(f_vals.dot(scheme.r_coeff) / eps_f)

            if save_history:
                history_entry = dict(
                    h=h,
                    r=r,
                    f_vals=f_vals
                )
                history.append(history_entry)

            if np.isnan(r):
                flag = -3  # error, NAN encoutered
                break

            if r > r_upper:
                u = h
            elif r < r_lower:
                l = h
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

            f_vals = np.array([self.__call__(x + s * h) for s in scheme.r_shift])

        else:
            flag = -1  # max iter exceeded

        num_eval = self.num_eval - num_eval
        num_eval_no_cache = self.num_eval_no_cache - num_eval_no_cache

        if scheme.est_coeff is not None and scheme.est_shift is not None and est_order is not None:
            estimated = np.array([self.__call__(x + s * h) for s in scheme.est_shift]).dot(scheme.est_coeff) / (
                        h ** est_order)
        else:
            estimated = None

        if estimated is not None and acc is not None:

            error = np.abs(estimated - acc)
            rel_error = error / np.abs(acc)
        else:
            error = None
            rel_error = None

        result = AdaEstRes(
            scheme_name=scheme.name,
            LS_flag=flag,
            h_init=h_init,
            h=h,
            h_opt=h_opt,
            r=r,
            n_iter=n_iter + 1,
            num_eval=num_eval,
            num_eval_no_cache=num_eval_no_cache,
            estimated=estimated,
            acc=acc,
            error=error,
            rel_error=rel_error,
            eps_f=eps_f,
            r_l=r_lower,
            r_u=r_upper
        )

        if save_history:
            self._last_ls_history = history

        return result

    def ada_est_gen_plot(
            self,
            x: float,
            scheme: EstimatingScheme,
            eps_f: Optional[float] = None,
            h_lower=1E-5,
            h_upper=10.,
            base=1.1,
            dpi=400,
            num_samples=1,
            **extra_args,
    ) -> None:

        ada_res = [self.ada_est(x=x, scheme=scheme, eps_f=eps_f, **extra_args) for _ in range(num_samples)]

        eps_f = ada_res[0].eps_f

        h_ada = [res.h for res in ada_res]

        index_lower = int(np.floor(np.log(h_lower) / np.log(base)))
        index_upper = int(np.ceil(np.log(h_upper) / np.log(base)))

        hs = base ** np.arange(index_lower, index_upper + 1)

        true_vale = ada_res[0].acc

        def worst_case_func(h):
            return (np.abs(scheme.est(obj=self.f, x=x, h=h) - true_vale) + np.abs(scheme.est_coeff).sum() * eps_f / (
                    h ** scheme.est_order)) / np.abs(true_vale)

        worst_case_err = [worst_case_func(h) for h in hs]

        with rc_context(rc={'figure.dpi': dpi}):
            ax = plt.gca()

            color = next(ax._get_lines.prop_cycler)['color']
            if num_samples == 1:
                label_str = "{} (h={:.2E})".format(scheme.name, ada_res[0].h)
                alpha = 1.0
                linestyle = '--'
            else:
                label_str = "{} (h in {:.2E}~{:.2E}".format(scheme.name, min(h_ada), max(h_ada))
                alpha = .3
                linestyle = '-'
            for i, h in enumerate(h_ada):
                if i == 0:
                    ax.axvline(h, linestyle=linestyle, color=color, alpha=alpha, label=label_str)
                else:
                    ax.axvline(h, linestyle=linestyle, color=color, alpha=alpha)
            ax.scatter(h_ada, [worst_case_func(h) for h in h_ada], color=color)

            # color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(hs, worst_case_err,
                    # label='{}: worst case'.format(scheme.name),
                    color=color)

            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"$f(x) = {sp.latex(self.f_expr)}, x={x}, \\epsilon_f = {self.eps_f}$")
            plt.xlabel("scaling factor $h$")
            plt.ylabel("worst case relative error")
            plt.legend()

    def optimal_h(self, x: float, scheme: EstimatingScheme, eps_f: Optional[float] = None) -> float:

        tol = 1E-10

        if eps_f is None:
            eps_f = self.eps_f

        O_R = scheme.err_order
        O_E = scheme.est_order

        L_R = np.abs(sp.lambdify(self.x, sp.diff(self.f_expr, self.x, O_R), "numpy")(x))

        if L_R < tol:
            return np.inf

        C_E = np.abs(scheme.est_coeff.dot(scheme.est_shift ** O_R)) / math.factorial(O_R)

        S_E = np.abs(scheme.est_coeff).sum()

        return ((O_E * S_E * eps_f) / (C_E * (O_R - O_E) * L_R)) ** (1 / O_R)
