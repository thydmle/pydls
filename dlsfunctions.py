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
    f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp*(-0.5*((d-mean)/sigma)**2)
    normconst = normalize(f, mie_fract, d[1]-d[0])
    f = normconst * f
    return f


# function calculates the gamma factor of a dls experiment
def calc_gamma(m, c, eta, n, theta, k_b, t, lambda_0, beta):
    return (16*(np.pi**2)*(np.sin(theta/2))**2*k_b*t)/(2*lambda_0**2*eta)


# function for single exponential fit
# useful for when you have a dls data set and you're trying to
# find a decent starting position for the walkers in Bayesian analysis
# use this fit to find the most probable radius size
# then generate a diameter size distribution with the mean at the most probable diameter size
def single_exponential_fit(t, C, const, B):
    return const*np.exp(-C*t) + B


# TODO: this function also depends on the Mie fraction in the exponential term
# that will need to be included once the Mie fraction's calculations are computed
def g2(f, d, beta, gamma, time):

    g2 = np.zeros(len(time))
    delta_d = d[1] - d[0]

    for i in len(time):
        expo = np.exp(-(gamma*time[i])/d)
        sum_squared = (np.sum(f*expo*delta_d))**2
        g2[i] = beta*sum_squared
    return g2

