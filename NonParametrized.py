import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy
import scipy.integrate
import seaborn as sns
import pandas as pn
import dlsfunctions as dls

# 06/21/2019: I'm not sure if I want to pursue an object oriented approach to this problem anymore because
# it seems that packing all necessary function calls into a package that can be imported is all that is
# required.

class NonParametrized(object):

    # TODO: will do later once I have a better idea of what properties an object of this class needs
    def __init__(self, ndim, nwalkers, nsteps, d_bins):
        self.sampler = 0
        # the user who initialized the NonParametrized object should have an idea
        # of the dimension of the parameter space
        self.ndim = ndim
        # how many bins of particle diameter sizes a user would like to infer
        self.d = np.zeros(d_bins)
        self.nwalkers = nwalkers
        self.nsteps = nsteps

    @staticmethod
    def g2(f, d, y, beta, gamma, time):

        delta_d = d[1] - d[0]
        g2 = np.zeros(len(time))

        for i in range(len(time)):
            exponential = np.exp(-(gamma*time[i])/d)
            # keep in mind that there is a Mie fraction C in sumready
            # assumed to be 1 during development
            sumready = f*exponential*delta_d
            currentsum = np.sum(sumready)**2
            g2[i] = beta*currentsum

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

        function = g2_result  # some instruments spit out 1 + g2
        residuals = (y - function)**2
        chi_square = np.sum(residuals)

        # alternative model for the likelihood based on integration
        # over noise variance
        return -(m/2)*np.log(chi_square)

    ###################################################################################
    # dont normalize g1, normalize the denominator thing! no tau dependence
    # TODO: fix
    # DONE - Thy 6/17/19 3:49pm
    @staticmethod
    def normalize(f, mie_fraction, delta_d):
        # this function normalizes the integral of g(1) before the inference stage
        # returns the normalization constant that sticks to the front of the g(1) integral

        g1_integrand = f*mie_fraction
        integral = scipy.integrate.trapz(g1_integrand, dx=delta_d)

        normalizationconstant = 1 / integral

        return normalizationconstant

    ####################################################################################

    def log_posterior(self, f, y, gamma, time):
        return self.log_prior(f) + self.log_likelihood(f, self.d, y, gamma, time)

    ################################################################################

    def infer(self, g2_data, beta, gamma, time):

        # prelim_pos has to be a particle size distribution that is somewhat
        # relevant to the size distribution being inferred
        prelim_pos = np.zeros(self.ndim)
        start_pos = [prelim_pos + 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, args=(self.d, g2_data, beta, gamma, time))
        self.sampler.run_mcmc(start_pos, self.nsteps)

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
        # axes = np.array([])
        # fig, axes = plt.subplots(self.ndim)
        # for i in range(self.ndim):
        #     axes[i].set(ylabel='f of ' + self.d[i])
        #
        # for i in range(self.ndim):  # type: int
        #     for j in range(10):
        #         sns.tsplot(self.sampler.chain[j, :, i], ax=axes[i])
        # REASON for commenting out code: np.array wouldn't work with a list of Axes objects. Will need to come up with
        # another method of grouping the figures together

    #########################################################

    # a function to find the burn in average?