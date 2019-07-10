import numpy as np
import dlsfunctions as dls

m = 20
c = 1
eta = 1e-3
angle = np.pi/2
n = 1.33
k_b = 1.38e-23
t = 298.15
lambda_0 = 638e-9
beta = 1
start = 5e-10
stop = 1e-8
r = 2.2439608235145855e-09
sigma = 2e-10


def get_lin_time():
    return np.linspace(1e-4, 1e-1, 200)*0.001


def get_log_time():
    return np.logspace(-4, -1, num=200, base=10)*0.001


def get_distance():
    return np.linspace(start, stop, m)


def get_gamma():
    return dls.calc_gamma(eta, n, angle, k_b, t, lambda_0)


def get_central_mean():
    return r*2


def get_left_mean():
    return r*2 - 3e-9


def get_right_mean():
    return r*2 + 3e-9


def get_narrow_sigma():
    return sigma/3


def get_normal_sigma():
    return sigma


def get_wide_sigma():
    return sigma*3


def get_real_wide_sigma():
    return sigma*6


