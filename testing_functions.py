from obj_func import *

x = sp.Symbol('x', real=True)

func1 = ObjFunc(expr=(sp.exp(x) - 1.) ** 2, eps_f=1E-6)
func2 = ObjFunc(expr=sp.exp(100 * x), eps_f=4E-6)
func3 = ObjFunc(expr=(x**4 + 3 * x**2 - 10 * x), eps_f=7E-6)
func5 = ObjFunc(expr=(10000 * x**3 + 0.01 * x**2 + 5 * x), eps_f=1E-6)
func6 = ObjFunc(expr= 1E3 * sp.cos(x), eps_f=1E-5)

