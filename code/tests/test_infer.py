from infer.dls_infer import g2, numerical_deriv, log_prior_beta, log_likelihood, log_posterior, create_start_pos, \
    create_sampler, infer
import emcee
import numpy as np
import matplotlib.pyplot as plt
import pandas
from infer.pre_infer import calc_gamma, generate_distribution

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

d = np.linspace(start, stop, m)
time = np.logspace(-4, -1, num=200, base=10)* 0.001
mean = (stop - start) / 2
gamma = calc_gamma(eta, n, angle, k_b, t, lambda_0)

test_distribution = generate_distribution(d, mean, sigma, 1)

theta_input = np.append(test_distribution, beta)
test_data = g2(theta_input, d, gamma, time)


def test_g2():
    assert len(test_data) == len(time)


def test_log_likelihood():
    likelihood = log_likelihood(theta_input, d, test_data, m, gamma, time)
    assert likelihood == 0


def test_numerical_deriv():
    deriv = numerical_deriv(test_distribution, 2)
    assert len(deriv) == len(test_distribution)



