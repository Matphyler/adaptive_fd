from obj_func import *
from common_schemes import FD, CD, FD3P, L2C, FD4P, CD4P
from testing_functions import *
import pandas as pd

schemes = [FD, FD3P, FD4P, CD, CD4P, L2C]

sep = "-" * 40
large_sep = "=" * 80

test_set = [
    [func1, -8.],
    [func2, 0.01],
    [func3, 0.9999],
    [func5, 1E-9],
    [func6, 1]
]

i = 0
h_lower, h_upper = 1E-3, 1E0
for scheme in [FD, FD3P, FD4P, CD, CD4P]:
    func1.ada_est_gen_plot(x=-8., scheme=scheme, h_lower=h_lower, h_upper=h_upper)
plt.tight_layout()
plt.savefig(f"figs/func_{i}.png", dpi=400, transparent=True)
plt.show()

i += 1
h_lower, h_upper = 1E-5, 1E3
for scheme in [FD, FD3P, FD4P, CD, CD4P]:
    func3.ada_est_gen_plot(x=0.99999, scheme=scheme,
                           h_lower=h_lower, h_upper=h_upper
    )
plt.tight_layout()
plt.savefig(f"figs/func_{i}.png", dpi=400, transparent=True)
plt.show()


i += 1
h_lower, h_upper = 1E-5, 1E3
for scheme in [FD, FD3P, FD4P, CD, CD4P]:
    func5.ada_est_gen_plot(x=1E-9, scheme=scheme,
                           h_lower=h_lower, h_upper=h_upper
    )
plt.tight_layout()
plt.savefig(f"figs/func_{i}.png", dpi=400, transparent=True)
plt.show()

i += 1

func6 = ObjFunc(expr= sp.cos(1E2*x), eps_f=1E-5)
for scheme in [FD, FD3P, FD4P, CD, CD4P]:
    func6.ada_est_gen_plot(x=1, scheme=scheme
    )

func6 = ObjFunc(expr= sp.cos(1E-1*x), eps_f=1E-5)
for scheme in [FD, FD3P, FD4P, CD, CD4P]:
    func6.ada_est_gen_plot(x=1, scheme=scheme
    )
plt.show()


plt.tight_layout()
plt.savefig(f"figs/func_{i}.png", dpi=400, transparent=True)
plt.show()

for func, xp in test_set:
    print(f"\n{large_sep}\n")
    func.enable_cache = True
    print(f"f(x) = {func.f_expr}, x = {xp}, eps_f = {func.eps_f}")
    print(f"{sep} with cache {sep}\n", pd.DataFrame([func.ada_est(x=xp, scheme=s) for s in schemes]).to_string())
    func.enable_cache = False
    print(f"{sep} without cache {sep}\n", pd.DataFrame([func.ada_est(x=xp, scheme=s) for s in schemes]).to_string())
