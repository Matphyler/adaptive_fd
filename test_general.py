from obj_func import *
from estimating_scheme import FD, CD, L2C
import pandas as pd

x = sp.Symbol('x', real=True)

func1 = ObjFunc(expr=(sp.exp(x) - 1.) ** 2, eps_f=1E-10)

func1.ada_general(x=-8.,scheme=FD)
func1.ada_general(x=-8., scheme=CD)
func1.ada_general(x=-8., scheme=L2C)