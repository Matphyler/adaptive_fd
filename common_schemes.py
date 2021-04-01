from estimating_scheme import *

FD = EstimatingScheme(
    est_shift=np.array([0., 1.]),
    est_coeff=np.array([-1., 1.]),
    name="FD"
)

CD = EstimatingScheme(
    est_shift=np.array([-1., 1.]),
    est_coeff=np.array([-.5, .5]),
    name="CD"
)

L2C = EstimatingScheme(
    est_shift=np.array([-1., 0., 1.]),
    est_coeff=np.array([1., -2., 1.]),
    name="L2_C"
)

FD3P = EstimatingScheme(
    est_shift=np.array([0., 1., 2.]),
    est_coeff=np.array([-1.5, 2., -.5]),
    name="FD_3P"
)

FD4P = EstimatingScheme(
    est_shift=np.array([0., 1., 2., 3.]),
    est_coeff=np.array([-11 / 6., 18 / 6., -9 / 6., 2 / 6.]),
    name="FD_4P"
)

CD4P = EstimatingScheme(
    est_shift=np.array([-2., -1., 1., 2.]),
    est_coeff=np.array([1 / 12., -8 / 12., 8 / 12., -1 / 12.]),
    name="CD_4P"
)
