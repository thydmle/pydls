from infer.rayleigh_gans import Rayleigh_Gans
import numpy as np

n_s = 1
n_p = 1.33
d = 4.48e-9
angle = np.pi / 2
wavelength = 638e-9
very_small_diameter = d * 10 ** -7
very_large_diameter = d * 10 ** 7


def rayleigh_coefficient(angle_in, radius_in, wavelength_in, n_s_in, n_p_in):
    return (8 * np.pi ** 4 * n_s_in * radius_in ** 4) / (wavelength_in ** 4) * (
                abs(((n_p_in / n_s_in) ** 2 - 1) / ((n_p_in / n_s_in) ** 2 + 1)) ** 2) * \
           (1 + np.cos(angle_in) ** 2)


def small_particle_test():
    rg_coefficient = Rayleigh_Gans(n_p, n_s, very_small_diameter, angle, wavelength)
    incident_intensity = 1
    assert incident_intensity * rg_coefficient.calc_scattering_matrix() == incident_intensity * rayleigh_coefficient(
        angle, very_small_diameter / 2, wavelength,n_s, n_p)
