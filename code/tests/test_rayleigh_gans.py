from infer.rayleigh_gans import Rayleigh_Gans as rg
import numpy as np

n_p = 1.33
n_s = 1
angle = np.pi/2
wavelength = 639e-9

particle_1 = wavelength / 2

rg_1 = rg(n_p, n_s, particle_1, angle, wavelength)
rg_1.calc_scattering_matrix()

print("Particle diameter: " + str(particle_1))
print("Scattering coefficient: " + str(rg_1.s1))

particle_2 = wavelength / 3

rg_2 = rg(n_p, n_s, particle_2, angle, wavelength)
rg_2.calc_scattering_matrix()

print("Particle diameter: " + str(particle_2))
print("Scattering coefficient: " + str(rg_2.s1))
