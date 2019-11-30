import numpy as np


# File to hold code that would generate the angular dependent scatttering
# limiting cases:
    # very small particle -> rayleigh
    # very large -> mie solution - MIE calculators

class Rayleigh_Gans:
    def __init__(self, n_p, n_s, d, angle, wavelength):
        self.n_p = n_p
        self.n_s = n_s
        self.d = d
        self.angle = angle
        self.wavelength = wavelength
        self.s1 = 0
        self.k = (2 * np.pi) / self.wavelength
        self.u = 2*np.pi * self.n_s * self.d / self.wavelength * np.sin(angle/2)
        self.volume = 4 * np.pi / 3 * (d / 2)**3
        self.m = n_p / n_s
        self.f = 3 / self.u**3 * (np.sin(self.u) - self.u * np.cos(self.u))

    def calc_scattering_matrix(self):
        self.s1 = ((self.k**3) / (2 * np.pi)) * (self.m - 1) * self.volume * self.f


