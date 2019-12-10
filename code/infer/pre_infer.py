import numpy as np
import scipy
import scipy.integrate


# tested
def normalize(f, rayleigh_gans_fract, delta_d):
    g1_integrand = f * rayleigh_gans_fract
    integral = scipy.integrate.trapz(g1_integrand, dx=delta_d)
    normconst = 1/integral
    return normconst


# tested
def determine_radius(C, n, lambda_0, theta, eta, k_b, t):
    q = ((4*np.pi*n)/lambda_0)*np.sin(theta/2)
    D = C/(q**2) / 0.001
    return (k_b*t)/(6*np.pi*eta*D)


# tested
def check_distribution_norm(f, delta_d):
    return scipy.integrate.trapz(f, dx=delta_d)


# tested
def generate_distribution(d, mean, sigma, rayleigh_gans_fract):
    f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((d-mean)/sigma)**2)
    normconst = normalize(f, rayleigh_gans_fract, d[1]-d[0])
    f = normconst * f
    return f


# tested
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


#tested
def calc_gamma(eta, n, angle, k_b, t, lambda_0):
    return (16*np.pi*(n**2)*((np.sin(angle/2))**2)*k_b*t) / (3*(lambda_0**2)*eta)


# tested
def single_exponential_fit(time, C, const, B):
    return const*np.exp(-C*time) + B

