from infer.rayleigh_gans import Rayleigh_Gans as rg
import numpy as np

wavelength = 639e-9

very_large_particle = wavelength * 1e1
very_small_particle = wavelength * 1e-3

n_p = 1.33
n_s = 1
angle = np.pi/2

very_small_rg = rg(n_p, n_s, very_small_particle, angle, wavelength)
very_large_rg = rg(n_p, n_s, very_large_particle, angle, wavelength)

very_small_rg.calc_scattering_matrix()
very_large_rg.calc_scattering_matrix()

intensity_small_rg = very_small_rg.s1 * 1
intensity_large_rg = very_large_rg.s1 * 1

print("Small particle")
print(intensity_small_rg)
print("Rayleigh regime: " + str(1 / wavelength**4))
print(intensity_large_rg)
# odd results, come back later
print(str(very_large_particle))
print(str(very_small_particle))