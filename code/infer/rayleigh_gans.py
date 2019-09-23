import numpy as np


# File to hold code that would generate the angular dependent scatttering


class Rayleigh_Gans:
    def __init__(self, n_p, n_s, d, angle, wavelength):
        self.n_p = n_p
        self.n_s = n_s
        self.d = d
        self.angle = angle
        self.wavelength = wavelength
        self.s1 = 0
        self.k = (2 * np.pi) / self.wavelength
        self.u = (2 * np.pi * self.n_s * self.d / 2) / self.wavelength

    def calc_scattering_matrix(self):
        s1 = (-1j * self.k**3) / (4 * np.pi) * ((self.n_p - self.n_s) / self.n_s) * self.d**3 / self.u**3 * \
             (np.sin(self.u) - self.u * np.cos(self.u))
        self.s1 = s1


# limiting cases:
    # very small particle -> rayleigh
    # very large
    # mie solution - MIE calculators