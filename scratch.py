if __name__ == "__main__":

    np.random.seed(int(datetime.datetime.utcnow().timestamp()))

    POLY_COEFF_BD = 1E1
    POLY_ORDER = 4
    X_BD = 2.
    L_UB = 1E-10
    ERR_LB = -1
    ERR_UB = 13

    # random point
    x = np.random.uniform(low=-X_BD, high=X_BD)
    # x = 1.

    f = random_polynomial(order=POLY_ORDER, c_lb=-POLY_COEFF_BD, c_ub=POLY_COEFF_BD)

    g = f.D()
    true_grad = g(x)

    L = np.abs(f.D().D()(x))

    print(f"f.order = {f.order}")
    print("f.coeff.bound = {:.2e}".format(POLY_COEFF_BD))
    print(f"f = {f}")
    print("x = {:.3e}".format(x))

    print("|D2f(x)| = {:.2e}".format(L))
    print("|D3f(x)| = {:.2e}".format(np.abs(f.D().D().D()(x))))

    assert L > L_UB, "L is too small, bad luck : ) Try again!"

    print(f"test from eps_f = 1E{-ERR_UB + 1} to 1E{-ERR_LB}")

    df = []
    for eps in 10. ** (-np.arange(ERR_LB, ERR_UB)):
        add_noise(f, eps)
        res = {'eps': f.eps_f}
        res.update(adaptive_fd(f, x))
        if res['LS_flag'] != 0:
            print("--- failed eps_f = {:.2e}".format(eps))
            continue
        res['n_eval'] = f.n_eval
        res['fwd_delta'] = res['fwd_grad'] - true_grad
        res['bwd_delta'] = res['bwd_grad'] - true_grad
        h_t = 2.0 * np.sqrt(eps / L)
        res["h_t"] = h_t
        res['fwd_grad_t'] = (f.eval(x + h_t) - f.eval(x)) / h_t
        res['bwd_grad_t'] = (f.eval(x) - f.eval(x - h_t)) / h_t
        res['fwd_delta_t'] = res['fwd_grad_t'] - true_grad
        res['bwd_delta_t'] = res['bwd_grad_t'] - true_grad
        res["h / h_t"] = res["h"] / res["h_t"]
        res['delta_th'] = 2 * np.sqrt(L * eps)
        df.append(res)

    print(f"succeeded {len(df)} of {ERR_UB - ERR_LB}")
    assert df, "Failed!"

    df = pd.DataFrame(df)

