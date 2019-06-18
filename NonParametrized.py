import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy
import scipy.integrate
import seaborn as sns
import pandas as pn


class NonParametrized(object):

    # TODO: will do later once I have a better idea of what properties an object of this class needs
    def __init__(self, ndim, d_bins):
        self.sampler = 0
        # the user who initialized the NonParametrized object should have an idea
        # of the dimension of the parameter space
        self.ndim = ndim
        # how many bins of particle diameter sizes a user would like to infer
        self.d = np.zeros(d_bins)

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
    # TODO: Needs to rewrite likelihood to give the correct model
    # needs to redefine likelihood function
    # before form? after integration form?

    def log_likelihood(self, f, y, stuff, time):
        m, c, delta_d, eta, n, theta, k_b, temp, lambda_0, beta = stuff

        g2_result = self.g2(f, self.d, stuff, time)

        # usually, this parameter is given as part of the instrumentation
        sig_y = 1e-2  # infer noise variance

        # sig_y = the noise variance
        function = g2_result  # some instruments spit out 1 + g2
        residuals = (y - function)**2

        # K constant before chi square is defined as
        # K = ln(1/(2*pi)^(m/2)*sig_y^(m))
        # k = np.log(1/((2*np.pi)**(m/2)*sig_y**m))  Gets into errors of zero division so if we simplify the log
        # even further, we get a much nicer expression

        # k = -m * (0.5 * np.log(2 * np.pi) + np.log(sig_y))

        # alternative model for the likelihood based on integration
        # over noise variance
        return -(m/2)*np.log(residuals)

    ###################################################################################
    # dont normalize g1, normalize the denominator thing! no tau dependence
    # TODO: fix
    # DONE - Thy 6/17/19 3:49pm
    @staticmethod
    def normalize(f, mie_fraction):
        # this function normalizes the integral of g(1) before the inference stage
        # returns the normalization constant that sticks to the front of the g(1) integral

        g1_integrand = f*mie_fraction
        integral = scipy.integrate.trapz(g1_integrand)

        normalizationconstant = 1 / integral

        return normalizationconstant

    ####################################################################################

    def log_posterior(self, f, y, stuff, time):
        return self.log_prior(f) + self.log_likelihood(f, self.d, y, stuff, time)

    ################################################################################

    def infer(self, nsteps, nwalkers, g2_data, d, stuff, time):
        prelim_pos = np.zeros(self.ndim)
        start_pos = [prelim_pos + 1e-4*np.random.randn(self.ndim)]

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior, args=(g2_data, self.d, stuff, time))
        self.sampler.run_mcmc(start_pos, nsteps)

    ####################################

    # def plot_sampler(self):
    # still not sure how to automate this plotting process
    # TODO: Implement stub function correctly? Figure out how to automate the plotting process of the walkers
    # stub function?
    # find the burn-in average
    # OK SO we can write a stub function that plots a thing that is called everytime something needs to be plotted
    # can be computationally expensive

    ## Stub function for plotting
    def walker_plot(self):
        plt.style.use('seaborn-deep')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 5),
                  'axes.labelsize': 'x-large',
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 'x-large',
                  'ytick.labelsize': 'x-large'}
        axes = np.array([])
        fig, axes = plt.subplots(self.ndim)
        for i in range(self.ndim):
            axes[i].set(ylabel='f of ' + self.d[i])

        for i in range(self.ndim):  # type: int
            for j in range(10):
                sns.tsplot(self.sampler.chain[j, :, i], ax=axes[i])

    #########################################################

    # a function to find the burn in average?