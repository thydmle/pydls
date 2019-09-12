import numpy as np


# File to hold code that would generate the angular dependent scatttering


# The Rayleigh-Gans scattering coefficient (instead of the Mie fraction) is supposed to
# fit right into the discrete sum of all g2 values as a single coefficient for
# each angle for each g2

# how does the form factor in the scattering matrix elements affect the intensity?


# Input: theta - angle of detection
#        x - the distance | Rcos(R, e_z - e_r) | from origin to a plane of constant phase
#               for plane waves this would mean a cross section area inside the volume of the sphere
# u = 2*x*np.sin(theta/2)
# f = 3/(u**3)*(np.sin(u) - u*np.cos(u))

# necessary parameters

def find_scattering_matrix_element(wavelength, angle, diameter, n_particle, n_medium):
    u = 2 * diameter * np.sin(angle/2)
    form_factor = (3 / u**3) * (np.sin(u) - u * np.cos(u))
    wave_number = (2 * np.pi) / wavelength
    s1 = - ((1j * wave_number**3) / 4) * diameter**3 * ((n_particle / n_medium) - 1) * form_factor
    return s1



