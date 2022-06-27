import pyshtools as pysh
import matplotlib.pyplot as plt
import numpy as np

basis_fns = lambda theta, phi: pysh.expand.spharm(2, theta, phi)

cilm = np.array([[[1]], [[0]]])
silm = np.array([[[0]], [[1]]])