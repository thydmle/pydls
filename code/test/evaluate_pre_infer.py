import numpy as np
import scipy
import scipy.integrate
from infer import pre_infer


d = []
for i in range(21):
    d[i] = np.linspace(i * 10**-10, i * 10**-8)

mean = []
for i in range(21):
    mean[i] = np.average(d[i])


sigma = 2e-10


def generate_nonnormal_distr(d_in, mean_in, sigma_in):
    test_f = (1/(sigma_in * np.sqrt(2 * np.pi))) * np.exp(-0.5*((d_in - mean_in) / sigma_in**2))
    return test_f


def test_normalize():
    test_f = []
    test_constants = []
    norm_constants = []
    for i in range(len(d)):
        test_f[i] = generate_nonnormal_distr(d[i], mean[i], sigma)
    for i in range(len(test_f)):
        test_constants[i] = 1 / (scipy.integrate.trapz(test_f[i], d[i][1] - d[i][0]))
        norm_constants[i] = pre_infer.normalize(test_f[i], 1, d[i][1] - d[i][0])
        assert test_constants[i] == norm_constants[i]


def test_generate_distr():
    test_f = []
    control_f = []
    for i in range(len(d)):
        test_f[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((d[i] - mean[i]) / sigma)**2)
        norm = 1 / (scipy.integrate.trapz(test_f[i], d[i][1] - d[i][0]))
        test_f[i] = test_f[i] * norm

        control_f[i] = pre_infer.generate_distribution(d[i], mean[i], sigma, 1)

        assert test_f[i] == control_f[i]

