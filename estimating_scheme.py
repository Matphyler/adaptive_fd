import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

import numpy as np


@dataclass
class EstimatingScheme:
    est_shift: np.ndarray
    est_coeff: np.ndarray
    est_order: Optional[int] = field(default=None)
    scaling_factor: float = field(default=2)
    r_shift: Optional[np.ndarray] = field(default=None)
    r_coeff: Optional[np.ndarray] = field(default=None)
    r_lower: Optional[float] = field(default=None)
    r_upper: Optional[float] = field(default=None)
    err_order: Optional[int] = field(default=None)
    h_init_func: Optional[Callable[[float], float]] = field(default=None)
    name: Optional[str] = field(default=None)

    def __post_init__(self) -> None:

        if self.est_order is None:
            self.est_order = self._infer_est_order()

        if self.err_order is None:
            self.err_order = self._infer_err_order()

        if self.h_init_func is None and self.err_order is not None:
            O_R = self.err_order
            O_E = self.est_order
            C_E = np.abs(self.est_coeff.dot(self.est_shift ** O_R)) / math.factorial(O_R)
            const = O_E * np.abs(self.est_coeff).sum() / (C_E * (O_R - O_E))

            self.h_init_func = lambda x: ((const * x) ** (1 / self.err_order))

        # if testing ratio details are not provided
        # infer them automatically from estimating scheme
        if self.r_shift is None or self.r_coeff is None:
            coeff_dict = defaultdict(lambda: 0.)

            for s, c in zip(self.est_shift, self.est_coeff):
                coeff_dict[s] += c
                coeff_dict[self.scaling_factor * s] -= c / (self.scaling_factor ** self.est_order)

            coeff_list = list(zip(*sorted(list(coeff_dict.items()), key=lambda x: x[0])))

            r_shift = coeff_list[0]
            r_coeff = coeff_list[1]

            r_coeff /= np.abs(r_coeff).sum()

            self.r_shift = np.array(r_shift)
            self.r_coeff = np.array(r_coeff)

        if self.r_lower is None or self.r_upper is None:
            self.r_lower, self.r_upper = self._infer_ratio_interval()

    @staticmethod
    def generate_scheme(template: "EstimatingScheme", scaling_factor: float,
                        name: Optional[str] = None) -> "EstimatingScheme":
        return EstimatingScheme(
            est_shift=template.est_shift,
            est_coeff=template.est_coeff,
            scaling_factor=scaling_factor,
            name=name
        )

    def __repr__(self) -> str:
        if self.name is not None:
            return f"EstimatingScheme({self.name})"
        else:
            return "EstimatingScheme()"

    def est(self, obj: Callable[[float], float], x: float, h: float) -> float:
        return np.array([obj(x + s * h) for s in self.est_shift]).dot(self.est_coeff) / (h ** self.est_order)

    def _gen_taylor_coeff(self, order: int) -> np.ndarray:
        return self.est_coeff.dot(np.power.outer(self.est_shift, np.arange(order + 1)) / np.array(
            [math.factorial(i) for i in range(order + 1)]).reshape((1, -1)))

    def _infer_est_order(self, tol=None, max_order=50) -> int:

        if tol is None:
            abs_coeff = np.abs(self.est_coeff)
            tol = min(max(1E-16, np.min(abs_coeff[np.nonzero(abs_coeff)])), 1E-5)

        order_upper_bound = 6
        while order_upper_bound < max_order:
            taylor_coeff = self._gen_taylor_coeff(order_upper_bound)
            try:
                return next(i for i in np.where(np.abs(taylor_coeff) >= tol)[0])
            except StopIteration:
                order_upper_bound *= 2

        raise ValueError("Failed; try increase max_order.")

    def _infer_err_order(self, tol=None, max_order=50) -> int:
        """
        This function try to infer the correct error order
        by computing the Taylor expansion of the estimator
        :param tol:
        :param max_order:
        :return:
        """

        if self.est_order is None:
            raise ValueError("no est_order; call _infer_est_order() and set est_order first.")

        if tol is None:
            abs_coeff = np.abs(self.est_coeff)
            tol = min(max(1E-16, np.min(abs_coeff[np.nonzero(abs_coeff)])), 1E-5)

        order_upper_bound = 6
        while order_upper_bound < max_order:
            taylor_coeff = self._gen_taylor_coeff(order_upper_bound)
            try:
                return next(i for i in np.where(np.abs(taylor_coeff) >= tol)[0] if i > self.est_order)
            except StopIteration:
                order_upper_bound *= 2

        raise ValueError("Failed; try increase max_order.")

    def _infer_target_ratio(self) -> float:

        O_R = self.err_order
        O_E = self.est_order

        C_E = np.abs(self.est_coeff.dot(self.est_shift ** O_R)) / math.factorial(O_R)
        C_R = np.abs(self.r_coeff.dot(self.r_shift ** O_R)) / math.factorial(O_R)

        return (O_E / (O_R - O_E)) * (C_R / C_E) * np.abs(self.est_coeff).sum()

    def _infer_ratio_interval(self) -> Tuple[float, float]:
        target_ratio = self._infer_target_ratio()
        # if target_ratio >= 11 / 4.:
        #     delta_l = target_ratio - 11 / 4.
        # else:
        #     delta_l = 0.
        #
        # r_lower = 1.5 + delta_l
        # r_upper = 2.5 + r_lower

        if target_ratio >= 2.2:
            r_lower = target_ratio / 2.
        else:
            r_lower = 1.1

        r_upper = 3. * r_lower

        return r_lower, r_upper
