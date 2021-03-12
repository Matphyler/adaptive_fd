from obj_func import *

x = sp.Symbol('x', real=True)

func1 = ObjFunc(expr=(sp.exp(x) - 1.) ** 2, eps_f=1E-1)
func1.gen_plot(x=-8., h_lower=1E-7, h_upper=10)
plt.show()

func2 = ObjFunc(expr=sp.exp(100 * x), eps_f=4E-6)
func2.gen_plot(x=0.01, h_lower=1E-7, h_upper=1E-2)
plt.show()

func3 = ObjFunc(expr=(x**4 + 3 * x**2 - 10 * x), eps_f=7E-6)
func3.gen_plot(x=0.99999, h_lower=1E-4, h_upper=1E-2, base=1.01)
plt.show()

func4 = ObjFunc(expr=sp.log(sp.functions.Abs(x)), eps_f=2E-5)
func4.gen_plot(x=1E-8, h_lower=1E-16, h_upper=1E-7, base=1.01)
plt.show()

func5 = ObjFunc(expr=(10000 * x**3 + 0.01 * x**2 + 5 * x), eps_f=1E-6)
func5.gen_plot(x=1E-9, base=1.01)
plt.show()


func6 = ObjFunc(expr=1E4 * sp.cos(x), eps_f=1E-5)
func6.gen_plot(x=1., base=1.01)
plt.show()

# func6 = ObjFunc(expr=sp.cos(x), eps_f=.5)
# func6.gen_plot(x=1., base=1.1)
# plt.show()
