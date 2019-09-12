import numpy as np
import scipy



# function returns the normalization constant
# for a distribution
# VARIABLES:
#       f : distribution that needs normalization
#       mie_fraction : Mie scattering fraction associated with the dls conditions for a specific trial
#       delta_d : the step size of the particle-size bins
#       g1_integrand : the product between the distribution f and the Mie fraction
#       intergal : the result of the integration using scipy.integrate.trapz
#       normconst : the normalization constant that is eventually returned
# RETURNS:
#       normconst : normalization constant to multiply to the input f distribution to normalize the distribution
def normalize(f, mie_fraction, delta_d):
    g1_integrad = f*mie_fraction
    integral = scipy.integrate.trapz(g1_integrad, dx=delta_d)
    normconst = 1/integral
    return normconst


# function checks if a distribution is normalized
# VARIABLES:
#       none
# RETURNS:
#       1 if normalized; something else otherwise
def check_distribution_norm(f, delta_d):
    return scipy.integrate.trapz(f, dx=delta_d)


# function generates a normalized distribution around a given range of particle size bins
# INPUT PARAMETERS:
#       d : range of particle sizes
#       mean :
# VARIABLES:
#       f : variable to hold the distribution generated. Unimodal distribution only
#       normconst : normalization constant for the distribution
# RETURNS:
#       f : normalized, unimodal distribution of particle sizes in range of d
def generate_distribution(d, mean, sigma, mie_fract):
    f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((d-mean)/sigma)**2)
    normconst = normalize(f, mie_fract, d[1]-d[0])
    f = normconst * f
    return f


# function generates normalized bimodal distribution with a given range of particle size bins
# VARIABLES:
#       f1
def generate_bimodal_distribution(d, mean1, sigma1, mean2, sigma2, mie_fract):
    f1 = 1 / (sigma1 * np.sqrt(2 * np.pi ** 2)) * np.exp(-(d - mean1) ** 2 / (2 * sigma1 ** 2))
    f1 = f1 * normalize(f1, 1, d[1] - d[0])

    f2 = 1 / (sigma2 * np.sqrt(2 * np.pi ** 2)) * np.exp(-(d - mean2) ** 2 / (2 * sigma2 ** 2))
    f2 = f2 * normalize(f2, 1, d[1] - d[0])

    f = f1 + f2
    f = f * normalize(f, 1, d[1] - d[0])
    return f


def generate_unequal_bimodal_distribution(d, mean1, sigma1, mean2, sigma2, mie_fract):
    f1 = 1 / (sigma1 * np.sqrt(2 * np.pi ** 2)) * np.exp(-(d - mean1) ** 2 / (2 * sigma1 ** 2))
    f1 = f1 * normalize(f1, 1, d[1] - d[0])

    f2 = 1 / (sigma2 * np.sqrt(2 * np.pi ** 2)) * np.exp(-(d - mean2) ** 2 / (2 * sigma2 ** 2))
    f2 = f2 * normalize(f2, 1, d[1] - d[0])

    f = f1 + 2 * f2
    f = f * normalize(f, 1, d[1] - d[0])
    return f


def calc_gamma(eta, n, angle, k_b, t, lambda_0):
    # return (16*(n**2)*(np.pi**2)*(np.sin(theta/2))**2*k_b*t)/(2*lambda_0**2*eta)
    return (16*np.pi*(n**2)*((np.sin(angle/2))**2)*k_b*t) / (3*(lambda_0**2)*eta)


# function for single exponential fit
# useful for when you have a dls data set and you're trying to
# find a decent starting position for the walkers in Bayesian analysis
# use this fit to find the most probable radius size
def single_exponential_fit(t, C, const, B):
    return const*np.exp(-C*t) + B