import numpy as np
import matplotlib.pyplot as plt
import emcee
import seaborn as sns
import pandas as pd
import scipy
import scipy.optimize
import scipy.integrate


# function returns the normalization constant
# for a distribution
def normalize(f, mie_fraction, delta_d):
    g1_integrad = f*mie_fraction
    integral = scipy.integrate.trapz(g1_integrad, dx=delta_d)
    normconst = 1/integral
    return normconst


# function checks if a distribution is normalized
# RETURNS: 1 if normalized; something else otherwise
def check_distribution_norm(f, delta_d):
    return scipy.integrate.trapz(f, dx=delta_d)


# function generates a normalized distribution
def generate_distribution(d, mean, sigma, mie_fract):
    f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((d-mean)/sigma)**2)
    normconst = normalize(f, mie_fract, d[1]-d[0])
    f = normconst * f
    return f


# function calculates the gamma factor of a dls experiment
def calc_gamma(eta, n, angle, k_b, t, lambda_0):
    # return (16*(n**2)*(np.pi**2)*(np.sin(theta/2))**2*k_b*t)/(2*lambda_0**2*eta)
    return (16*np.pi*(n**2)*((np.sin(angle/2))**2)*k_b*t) / (3*(lambda_0**2)*eta)


# function for single exponential fit
# useful for when you have a dls data set and you're trying to
# find a decent starting position for the walkers in Bayesian analysis
# use this fit to find the most probable radius size
def single_exponential_fit(t, C, const, B):
    return const*np.exp(-C*t) + B


# TODO: this function also depends on the Mie fraction in the exponential term
# that will need to be included once the Mie fraction's calculations are computed
#def g2(f, d, beta, gamma, time):

#    size = len(time)
#    g2 = np.zeros(size)
#    delta_d = d[1] - d[0]
#
#    for i in range(size):
#        expo = np.exp(-(gamma*time[i])/d)
#        sum_squared = (np.sum(f*expo*delta_d))**2
#        g2[i] = beta*sum_squared
#    return g2

def g2(theta, d, m, gamma, time):
    beta = theta[m]
    f = theta[0:m]
    size = len(time)
    y = np.zeros(size)
    delta_d = d[1] - d[0]
    f = f*normalize(f, 1, delta_d)
    for i in range(size):
        expo = np.exp(-(gamma*time[i])/d)
        sum_squared = (np.sum(f*expo*delta_d))**2
        y[i] = beta*sum_squared
    return y


def determine_radius(C, n, lambda_0, theta, eta, k_b, t):
    q = ((4*np.pi*n)/lambda_0)*np.sin(theta/2)
    D = C/(q**2) / 0.001
    return (k_b*t)/(6*np.pi*eta*D)


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
 #   if beta <= 0 or beta > 2:
 #      not_ok = True
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


# TODO: don't need m if you have d
#DONE
def log_likelihood(theta, d, y, m, gamma, time):
    g2_result = g2(theta, d, m, gamma, time)
    # keep in mind that g2 will require beta factor in the future
    residuals = (y - g2_result)**2
    chi_square = np.sum(residuals)
    return -(m/2)*chi_square


def log_posterior(theta, d, y, m, gamma, time):
    # theta will be an array of size (m+1, )
    # log_prior and log_likelihood will need to slice theta correctly

    return log_prior(theta, m) + log_prior_beta(theta, m) + log_likelihood(theta, d, y, m, gamma, time)


def create_start_pos(theta, ndim, nwalkers):
    start_pos = [theta + 1e-4*np.absolute(np.random.randn(ndim)) for i in range(nwalkers)]
    return start_pos


def create_sampler(nwalkers, ndim, d, y, m, gamma, time):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(d, y, m, gamma, time))
    return sampler


def infer(sampler, start_pos, nsteps):
    result = sampler.run_mcmc(start_pos, nsteps)
    return sampler


def view_burnin_plot(sampler, first_param, second_param):
    plt.style.use('seaborn-deep')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}

    fig, (ax0, ax1) = plt.subplots(2)
    ax0.set(ylabel='f(d' + str(first_param) + ')')
    ax1.set(ylabel='f(d' + str(second_param) + ')')

    for j in range(10):
        sns.tsplot(sampler.chain[j, :, first_param], ax=ax0)
        sns.tsplot(sampler.chain[j, :, second_param], ax=ax1)
    # plots 2 inferred parameters at a time


def chain(sampler, step_to_chain_at, ndim):
    return sampler.chain[:, step_to_chain_at:,:]


def create_dataframe(chained_sampler, param_num):
    traces = chained_sampler.reshape(-1, param_num).T
    samples_dictionary = {}
    for i in range(param_num-1):
        samples_dictionary["f"+str(i)] = traces[i]
    samples_df = pd.DataFrame(samples_dictionary)
    return samples_df


def get_infer_f(quantiled_samples, m):
    from_df = quantiled_samples.quantile([0.5], axis=0)
    result = from_df.values
    return result[0]


def get_beta(chained_sampler, ndim):
    traces = chained_sampler.reshape(-1, ndim).T
    beta = {'beta': traces[ndim-1]}
    beta_df = pd.DataFrame(beta)
    return beta_df
