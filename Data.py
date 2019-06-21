import numpy as np
import Metadata
import matplotlib.pyplot as plt


class Data(object):
    def __init__(self, filename, rowstoskip, rowstoread, metadata):
        self.dls_data = np.loadtxt(filename, encoding='latin1', skiprows=rowstoskip, max_rows=rowstoread)
        temp = metadata.temp
        boltzmann_const = metadata.boltzmann_const
        viscosity = metadata.viscosity
        refractive_index = metadata.refractive_index
        theta = metadata.theta
        lambda_0 = metadata.lambda_0
        self.gamma = (16*np.pi*refractive_index**2*np.sin(theta/2)**2*boltzmann_const*temp) / (2*lambda_0**2*viscosity)
        self.intensity_acf = self.dls_data[:, 1]
        self.tau = self.dls_data[:, 0]

    def plot_raw_data(self):
        plt.semilogx(self.tau, self.intesity_acf, 'bo')
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel('g(2)')