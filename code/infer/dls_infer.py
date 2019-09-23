import numpy as np
import emcee
import pre_infer


def g2(theta, d, gamma, time):
    m = len(d)
    beta = theta[m]
    f = theta[0:m]
    size = len(time)
    y = np.zeros(size)
    delta_d = d[1] - d[0]
    f = f * pre_infer.normalize(f, 1, delta_d)
    for i in range(size):
        expo = np.exp(-(gamma * time[i]) / d)
        sum_squared = (np.sum(f * expo * delta_d))**2
        y[i] = beta * sum_squared
    return y


def numerical_deriv(f, degree):
    result = np.zeros(len(f))
    for i in range(degree):
        result = np.gradient(f)
        f = result
    return result


def log_prior(theta, m):
    beta = theta[m]
    f = theta[0:m]
    f_2nd_deriv = numerical_deriv(f, 2)
    a = np.dot(f_2nd_deriv, f_2nd_deriv.transpose())
    not_ok = False
    for i in range(m):
        if f[i] < 0:
            not_ok = True
    if not_ok:
        return -np.inf
    else:
        return -a


def log_prior_beta(theta, m):
    beta = theta[m]
    not_ok = False
    sigma = 1e-1
    beta_0 = 1
    if beta <= 0 or beta > 2:
        not_ok = True
    if not_ok:
        return -np.inf
    else:
        return -((beta-beta_0)**2)/(2*sigma**2)


def log_likelihood(theta, d, y, m, gamma, time):
    g2_result = g2(theta, d, m, gamma, time)
    # keep in mind that g2 will require beta factor in the future
    residuals = (y - g2_result)**2
    chi_square = np.sum(residuals)
    return -(m/2)*chi_square


def log_posterior(theta, d, y, m, gamma, time):
    # theta will be an array of size (m+1, )
    # log_prior and log_likelihood will need to slice theta correctly

    return log_prior(theta, m) + log_prior_beta(theta, m) + log_likelihood(theta, d, y, gamma, time)


def create_start_pos(theta, ndim, nwalkers):
    start_pos = [theta + 1e-4*np.absolute(np.random.randn(ndim)) for i in range(nwalkers)]
    return start_pos


def create_sampler(nwalkers, ndim, d, y, m, gamma, time):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(d, y, m, gamma, time))
    return sampler


def infer(sampler, start_pos, nsteps):
    result = sampler.run_mcmc(start_pos, nsteps)
    return sampler


