import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
import math


@dataclass
class EstimatingScheme:
    est_shift: np.ndarray
    est_coeff: np.ndarray
    est_h_power: int
    est_order: int
    r_lower: float
    r_upper: float
    r_shift: np.ndarray
    r_coeff: np.ndarray
    h_init: Optional[Callable[[float], float]] = field(default=None)
    name: Optional[str] = field(default=None)

    def __repr__(self):
        if self.name is not None:
            return f"EstimatingScheme({self.name})"
        else:
            return "EstimatingScheme()"


FD = EstimatingScheme(
    est_shift=np.array([0., 1.]),
    est_coeff=np.array([-1., 1.]),
    est_h_power=1,
    est_order=1,
    r_lower=1.5,
    r_upper=4.,
    r_shift=np.array([0., 1., 2.]),
    r_coeff=np.array([0.25, -0.5, 0.25]),
    h_init=math.sqrt,
    name="forward_differencing"
)

CD = EstimatingScheme(
    est_shift=np.array([-1., 1.]),
    est_coeff=np.array([-.5, .5]),
    est_h_power=1,
    est_order=1,
    r_lower=1.5,
    r_upper=4.,
    r_shift=np.array([-2., -1., 1., 2.]),
    r_coeff=np.array([-1/6., 1/3., -1/3., 1/6.]),
    h_init=(lambda x: x ** (1/3.)),
    name="central_differencing"
)

L2C = EstimatingScheme(
    est_shift=np.array([-1., 0., 1.]),
    est_coeff=np.array([1., -2., 1.]),
    est_h_power=2,
    est_order=2,
    r_lower=1.5,
    r_upper=4.,
    r_shift=np.array([-2., -1., 0., 1., 2.]),
    r_coeff=np.array([1/16., -4/16., 6/16., -4/16., 1/16.]),
    h_init=(lambda x: x ** (1/4.)),
    name="L2_central"
)