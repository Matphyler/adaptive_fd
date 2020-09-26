from AdaptiveFD import *
from Polynomial import *
from InjectNoise import *
import pandas as pd
from matplotlib import rc_context
from matplotlib import pyplot as plt
import datetime
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging
import json
import click
import os
import sys


def run_test_single_polynomial(poly_coeff_bd: Tuple[float, float] = (-10., 10.),
                               poly_order: int = 4,
                               x_range: Tuple[float, float] = (-2., 2.),
                               l_lower_bd: float = 1E-10,
                               err_range: Tuple[int, int] = (-13, 0),
                               logger: Optional[logging.Logger] = None,
                               **kwargs
                               ):
    if logger is None:
        logger = logging.getLogger("adaptive_fd_experiment")

    logger.info(
        """========== Test information ==========
polynomial coefficient bound: [{poly_coeff_l:.3e}, {poly_coeff_u:.3e}]
polynomial order: {poly_order}
x_range: [{x_range_l:.3E}, {x_range_u:.3E}]
error range: 1E{err_l} to 1E{err_u}
        """.format(
            poly_coeff_l=poly_coeff_bd[0],
            poly_coeff_u=poly_coeff_bd[1],
            poly_order=poly_order,
            x_range_l=x_range[0],
            x_range_u=x_range[1],
            err_l=err_range[0],
            err_u=err_range[1]
        )
    )

    coeff = np.random.uniform(
        low=poly_coeff_bd[0],
        high=poly_coeff_bd[1],
        size=poly_order + 1
    )
    obj = Polynomial(coeff=coeff)
    logger.info(f"f(x) = {obj.__repr__()}")

    x = np.random.uniform(low=x_range[0], high=x_range[1])
    logger.info("x = {:.3E}".format(x))

    D0 = obj(x)
    D1 = obj.D(1)(x)
    D2 = np.abs(obj.D(2)(x))
    D3 = np.abs(obj.D(3)(x))
    logger.info(
        """Derivatives at x:
0-th order (f(x)):     {:.3E} 
1-st order (grad(x)):  {:.3E}
2-nd order (abs val):  {:.3E} 
3-rd order (abs val):  {:.3E}
        """.format(D0, D1, D2, D3)
    )

    if D2 < l_lower_bd:
        raise ValueError(
            "second order derivative {:.3E} "
            "is smaller than {:.3E}".format(D2, l_lower_bd)
        )

    logger.info(
        """========== Start running =========="""
    )

    df = []

    for eps_f in 10. ** (np.arange(err_range[0], err_range[1] + 1)):
        logger.info(
            "running eps_f = {:.2e}".format(eps_f)
        )
        obj = add_noise(eps_f=eps_f, obj=obj)

        assert hasattr(obj, "eps_f")
        assert hasattr(obj, "n_eval")

        res = {'eps_f': obj.eps_f}
        res.update(adaptive_fd(obj=obj, x=x, **kwargs))

        if res['LS_flag'] != 0:
            logger.warning("Failed LS for eps_f = {:.2e}".format(eps_f))
            continue

        res['n_eval'] = obj.n_eval

        res['fwd_error'] = res['fwd_grad'] - D1
        res['bwd_error'] = res['bwd_grad'] - D1

        h_opt = 2.0 * np.sqrt(eps_f / D2)  # theoretical

        res["h_opt"] = h_opt

        res['fwd_grad_opt'] = (obj.call(x + h_opt) - obj.call(x)) / h_opt
        res['bwd_grad_opt'] = (obj.call(x) - obj.call(x - h_opt)) / h_opt
        res['fwd_error_opt'] = res['fwd_grad_opt'] - D1
        res['bwd_error_opt'] = res['bwd_grad_opt'] - D1
        res["h/h_opt"] = res["h"] / res["h_opt"]
        res['error_th'] = 2 * np.sqrt(D2 * eps_f)
        df.append(res)

    experiment_results = dict()

    experiment_results['config'] = {
        "poly_coeff_bd": poly_coeff_bd,
        "poly_order": poly_order,
        "x_range": x_range,
        "l_lower_bd": l_lower_bd,
        "err_range": err_range,
        "extra_parameters": kwargs
    }

    experiment_results['test_info'] = {
        "f_str": obj.__repr__(),
        "f_latex": obj.to_latex(),
        "f_coeff": obj.coeff.tolist(),
        "x": x,
        "derivatives": {
            "0": D0,
            "1": D1,
            "2": D2,
            "3": D3
        }
    }

    experiment_results['test_data'] = df

    """========== Experiments done =========="""

    return experiment_results


def generate_figures(results, dpi=400, transparent=True, logger=None, path=""):

    if logger is None:
        logger = logging.getLogger("adaptive_fd_experiment")

    df = pd.DataFrame(results['test_data'])
    obj = Polynomial(results['test_info']['f_coeff'])
    x = results['test_info']['x']

    plt.plot(df['eps_f'], np.abs(df['fwd_error_opt']), '--o', label='optimal')
    plt.plot(df['eps_f'], np.abs(df['fwd_error']), '--o', label='LS')
    plt.plot(df['eps_f'], np.abs(df['error_th']), '-', label='theoretical')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('$\\epsilon_f$')
    plt.ylabel('$\\epsilon_g$')
    plt.title(f"$f(x) = {obj.to_latex()}$, $x = {'{:.2f}'.format(x)}$", fontdict = {'fontsize' : 5})
    # plt.title("$\\epsilon_g$ vs $\\epsilon_f$")
    file_path = os.path.join(path, "epsilon_g.png")
    plt.savefig(file_path, dpi=dpi, transparent=transparent)
    logger.info(f"saved '{file_path}'")
    plt.clf()

    plt.plot(df['eps_f'], df['h_opt'], '--o', label='optimal')
    plt.plot(df['eps_f'], df['h'], '--o', label='LS')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('$\\epsilon_f$')
    plt.ylabel('$h$')
    plt.title(f"$f(x) = {obj.to_latex()}$, $x = {'{:.2f}'.format(x)}$", fontdict = {'fontsize' : 5})
    # plt.title("$h$ vs $\\epsilon_f$")
    file_path = os.path.join(path, "interval.png")
    plt.savefig(file_path, dpi=dpi, transparent=transparent)
    logger.info(f"saved '{file_path}'")
    plt.clf()

    plt.plot(df['eps_f'], df['n_eval'], '--o')
    plt.xscale('log')
    plt.xlabel('$\\epsilon_f$')
    plt.ylabel('n_eval')
    plt.title(f"$f(x) = {obj.to_latex()}$, $x = {'{:.2f}'.format(x)}$", fontdict = {'fontsize' : 5})
    # plt.title("# func evaluations")
    file_path = os.path.join(path, "n_func_eval.png")
    plt.savefig(file_path, dpi=dpi, transparent=transparent)
    logger.info(f"saved '{file_path}'")
    plt.clf()


@click.command()
@click.option('-o', '--order',
              required=False,
              default=4,
              type=int,
              help="order of the polynomial"
              )
@click.option('--cl',
              required=False,
              default = -10.,
              type=float,
              help="lower bound of polynomial coefficient"
              )
@click.option('--cu',
              required=False,
              default=10.,
              type=float,
              help="upper bound of polynomial coefficient"
              )
@click.option('--xl',
              required=False,
              default=-2.,
              type=float,
              help="lower bound of x"
              )
@click.option('--xu',
              required=False,
              default=2.,
              type=float,
              help="upper bound of x"
              )
@click.option('--el',
              required=False,
              default=-13,
              type=int,
              help="lower bound of error (in terms of 1E<lower_bound>)"
              )
@click.option('--eu',
              required=False,
              default=0,
              type=int,
              help="upper bound of error (in terms of 1E<upper_bound>)"
              )
@click.option('--out',
              required=False,
              default="",
              type=str,
              help="path to output"
              )
def main(order, cl, cu, xl, xu, el, eu, out):

    logger = logging.getLogger("adaptive_fd_experiment")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)

    results = run_test_single_polynomial(
        poly_coeff_bd=(cl, cu),
        poly_order=order,
        x_range=(xl, xu),
        err_range=(el, eu),
        logger=logger
    )

    if not Path(out).is_dir():
        Path(out).mkdir()
        logger.info(f"output directory '{out}' did not exist and is created.")

    json_path = os.path.join(out, "experiments.json")
    with open(json_path, 'w+') as file:
        json.dump(results, file, indent=2)
    logger.info(f"saved experiment results to '{json_path}'")

    generate_figures(results, logger=logger, path=out)


if __name__ == "__main__":
    main()
