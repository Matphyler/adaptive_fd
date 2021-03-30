import numpy as np
from estimating_scheme import EstimatingScheme
import pandas as pd

from testing_functions import *

CD_family = [
    EstimatingScheme(
    est_shift=np.array([-1., 1.]),
    est_coeff=np.array([-.5, .5]),
    est_h_power=1,
    est_order=1,
    scaling_factor=i,
    name=f"CD_{i}") for i in [1.01, 1.05, 1.1, 1.2, 1.5, 2., 2.5, 3., np.pi, 4., 5., 6., 10., 20., 100., 1E3]]

# [s._infer_ratio_interval() for s in CD_family]

func1.deterministic = True
func2.deterministic = True
print(pd.DataFrame([func2.ada_est(x=0.01, scheme=s) for s in CD_family]).to_string())
