import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy
import scipy.integrate


class NonParametrized(object):

    # TODO: will do later once I have a better idea of what properties an object of this class needs
    # def __init__(self):

    @staticmethod
    def g2(f, d, y, stuff, time):
        m, c, delta_d, eta, n, theta, k_b, temp, lambda_0, beta = stuff
        g2 = np.ones(len(d))
        currentsum = 0

        # assuming fixed radius uncertainty
        for j in range(len(time)):

            for i in range(len(d)):
                currentradius = d[i]
                currentweight = f[i]
                gamma = (16 * np.pi * (n ** 2) * np.sin(theta / 2) * k_b * temp) / (3 * eta * lambda_0 ** 2)
                exponential = np.exp(-(gamma * time[j]) / currentradius)

                sumready = currentweight * c * exponential * delta_d
                currentsum = currentsum + sumready

            g2[j] = currentsum
            currentsum = 0

        return g2

#######################################################################################
    @staticmethod
    def numerical_deriv(degree, f):
        # calculates the nth numerical derivative of a given array of data points, using finite difference
        # degree : the degree of your desired numerical derivative calculation
        # f : array of data points

        # the derivative is taken to be f"(x) = { f(x+h) - f(x-h) } / 4h^2
        # from Numerical Recipes by William H. Press, Teukolsky, Vetterling, et al

        result = np.zeros(len(f))
        for i in range(degree):
            result = np.gradient(f)
            f = result

        return result

#######################################################################################

    def log_prior(self, f):
        # f : the particle size distribution that we are trying to model
        # f will be passed in as an array
        #    f_transpose = f.transpose()
        #    f_firstDeriv = np.gradient(f)              # is this legal?
        #    f_2ndDeriv = np.gradient(f_firstDeriv)
        #    a = np.dot(-f_tranpose, f_2ndDeriv)        # look up Numerical Recipes for second derivative matrix

        f_2ndDeriv = self.numerical_deriv(2, f)
        a = np.dot(f_2ndDeriv, f_2ndDeriv.transpose())

        foundZero = False

        for i in range(len(f)):
            if f[i] < 0:
                foundZero = True

        # the prior was described to be:
        # L = p(f) = exp(-f_transpose*f_2ndDeriv) for f >= 0
        # ln(L) = -f_transpose*f_2ndDeriv for f >= 0
        # from Boualem, Kabloun, Ravier, Naiim and Jalocha

        if foundZero is False:
            return -a  # if all values of f were non negative, the log of the prior is the exponential part
        else:
            return -np.inf
        # if a value of f was negative, then the prior is zero, and ln(0) is infinity
#######################################################################################

    def log_likelihood(self, f, d, y, stuff, time):
        m, c, delta_d, eta, n, theta, k_b, temp, lambda_0, beta = stuff

        g2_result = self.g2(f, d, stuff, time)

        # usually, this parameter is given as part of the instrumentation
        sig_y = 1e-2  # infer noise variance

        function = g2_result  # some instruments spit out 1 + g2
        residuals = (y - function) ** 2
        chi_square = np.sum((residuals) / sig_y ** 2)

        # K constant before chi square is defined as
        # K = ln(1/(2*pi)^(m/2)*sig_y^(m))
        # k = np.log(1/((2*np.pi)**(m/2)*sig_y**m))  Gets into errors of zero division so if we simplify the log
        # even further, we get a much nicer expression

        k = -m * (0.5 * np.log(2 * np.pi) + np.log(sig_y))

        return k - 0.5 * chi_square
    ###################################################################################

    @staticmethod
    def normalize(f, d, stuff, time):
        # this function normalizes the integral of g(1) before the inference stage
        # returns the normalization constant that sticks to the front of the g(1) integral
        m, c, delta_d, eta, n, theta, k_b, temp, lambda_0, beta = stuff

        gamma = (16 * np.pi * (n ** 2) * ((np.sin(theta / 2)) ** 2) * k_b * temp) / (3 * eta * lambda_0 ** 2)
        g1_integrand = f * np.e ** (-(gamma / d) * time)
        integral = scipy.integrate.trapz(g1_integrand)

        normalizationconstant = 1 / integral

        return normalizationconstant

    ####################################################################################

