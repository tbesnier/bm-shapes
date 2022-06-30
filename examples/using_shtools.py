import pyshtools as pysh
import numpy as np

lmax = 10
a = 20  # scale length
degrees = np.arange(lmax+1, dtype=float)
power = 1. / (1. + (degrees / a) ** 2) ** 0.5

coeffs_global = pysh.SHCoeffs.from_random(power)
grid = coeffs_global.expand()

a = grid.to_array()
print(a.shape())