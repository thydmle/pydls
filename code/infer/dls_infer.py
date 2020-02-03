import numpy as np
import emcee
from infer.pre_infer import normalize
import infer.post_infer
from infer.rayleigh_gans import Rayleigh_Gans as rg


def g2(theta, d, gamma, time):
    m = len(d)
    beta = 1. # hard code for now, don't keep this
    f = theta
    size = len(time)
    y = np.zeros(size)
    delta_d = d[1] - d[0]
    f = f * normalize(f, 1, delta_d)
    for i in range(size):
        expo = np.exp(-(gamma * time[i]) / d)
        sum_squared = (np.sum(f * expo * delta_d))**2
        y[i] = beta * sum_squared
    return y


def g2_multiangle(theta, d, gamma, n_p, n_s, angle, wavelength, time):
    """

    :param theta: tuple of starting particle size distribution, plus beta appended in the end of the tuple
    :param d: array of particle diameters
    :param gamma: gamma factor previously calculatd
    :param n_p: index of refraction of the particle (scatterer)
    :param n_s: index of refraction of surrounding
    :param angle: angle at which this specific g2 measurement was made. IS IMPORTANT for multiangle. EACH g2 HAS ITS OWN
                    angle and thus altering the rayleigh coefficient and thus the likelihood distribution
    :param wavelength: laser's wavelength
    :param time: time scale in ms
    :return: array of scattered intensity for each diameter member within d
    """
    m = len(d)
    beta = theta[m]
    f = theta[0:m]
    size = len(time)
    y = np.zeros(size)
    delta_d = d[1] - d[0]
    # have to calculate Rayleigh-Gans coefficient prior to normalizing
    # because normalization step uses rayleigh-gans coefficient
    coefficients = calc_rayleigh_gans(d, n_p, n_s, angle, wavelength)
    f = f * normalize(f, coefficients, delta_d)
    for i in range(size):
        expo = np.exp(-(gamma * time[i]) / d)
        sum_squared = (np.sum(f * coefficients * expo * delta_d))**2
        y[i] = beta * sum_squared
    return y

def calc_rayleigh_gans(d, n_p, n_s, angle, wavelength):
    coefficients = []

    for i in d:
        current_rg = rg(n_p, n_s, i, angle, wavelength)
        coefficients.append(current_rg.s1)
    return coefficients

def numerical_deriv(f, degree):
    result = np.zeros(len(f))
    for i in range(degree):
        result = np.gradient(f)
        f = result
    return result


def log_prior(theta, m, guess_pos, d):
    #beta = theta[m]
    f = theta * guess_pos
    # distribution needs to be normalized at all steps 
    # therefore also needs to be normalized here
    # 2/3/20
    f = f * normalize(f, 1, d[1] - d[0])
    f_2nd_deriv = numerical_deriv(f, 2)
    a = np.dot(f_2nd_deriv, f_2nd_deriv.transpose())
    not_ok = False
    if (f < 0).any(): # check if any element of size dist is negative
        return -np.inf
    else:
        return -a


#def log_prior_beta(theta, m):
#    beta = theta[m]
#    not_ok = False
#    sigma = 1e-1
#    beta_0 = 1
#    if beta <= 0 or beta > 2:
#        not_ok = True
#    if not_ok:
#        return -np.inf
#    else:
#        return -((beta-beta_0)**2)/(2*sigma**2)


def log_likelihood(theta, d, y, m, gamma, time, guess_pos):
    f = theta * guess_pos
    g2_result = g2(f, d, gamma, time)
    # keep in mind that g2 will require beta factor in the future
    residuals = (y - g2_result)**2
    chi_square = np.sum(residuals)
    return -(m/2)* np.log(chi_square)


def log_likelihood_multiangle(theta, d, y, m, gamma, n_p, n_s, angle, wavelength, time):
    g2_result = g2_multiangle(theta, d, gamma, n_p, n_s, angle, wavelength, time)
    residuals = (y - g2_result)**2
    chi_square = np.sum(residuals)
    return -(m/2) * chi_square


def log_posterior(theta, d, y, m, gamma, time, guess_pos):
    # theta will be an array of size (m+1, )
    # log_prior and log_likelihood will need to slice theta correctly

    return log_prior(theta, m, guess_pos, d) + log_likelihood(theta, d, y, m, gamma, time, guess_pos)


#def log_posterior_multiangle(theta, d, y, m, gamma, n_p, n_s, angle, wavelength, time):
#    return log_prior(theta, m) + log_prior_beta(theta, m) + log_likelihood_multiangle(theta, d, y, m,  gamma, n_p, n_s,
#                                                                                      angle, wavelength, time)


def create_start_pos(theta, ndim, nwalkers):
    print(theta)
    start_pos = np.array([theta + 1e-4*theta*np.absolute(np.random.randn(ndim))
                          for i in range(nwalkers)])
    print(start_pos)
    # start_pos / theta = relative ratio distribution that is input into inference
    # theta = actual (generated/true) distribution
    # start_pos = gaussan-balled distribution
    return start_pos / theta


def create_sampler(nwalkers, ndim, d, y, m, gamma, time, guess_pos):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(d, y, m, gamma, time, guess_pos))
    return sampler

def create_sampler_multiangle(nwalkers, ndim, d, y, m, gamma, n_p, n_s, angle, wavelength, time):
    pass

def infer(sampler, start_pos, nsteps):
    result = sampler.run_mcmc(start_pos, nsteps)
    return sampler


