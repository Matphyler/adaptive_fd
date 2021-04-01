import pandas as pd

from common_schemes import *
from testing_functions import *

scaling_factors = [1.01, 1.05, 1.1, 1.2, 1.5, 2., 2.5, 3., 4., 5., 6., 10., 20., 100., 1E3]

test_set = [
    [func1, -8.],
    [func2, 0.01],
    [func3, 0.9999],
    [func5, 1E-9],
    [func6, 1]
]

schemes = [FD, FD3P, FD4P, CD, CD4P]
large_sep = "=" * 80

for scheme in schemes:
    scheme_list = [EstimatingScheme.generate_scheme(scheme, scaling_factor=sf, name=f"{scheme.name}_{sf}") for sf in
                   scaling_factors]
    for func, xp in test_set:
        print(f"\n{large_sep}")
        print(f"f(x) = {func.f_expr}, x = {xp}, eps_f = {func.eps_f}\n")
        print(pd.DataFrame(
            [func.ada_est(x=xp, scheme=sc).to_dict() for sc in
             scheme_list]).to_string())
