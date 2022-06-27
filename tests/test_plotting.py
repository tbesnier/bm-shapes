import numpy as np

from utils import plotting

def test_plot_coords():
    x = [1]
    y = [2]
    z = [3]
    plotting.plot_coords(x, y, z)

def test_plot_polar():
    # Should give x=y=0, z=1
    lat = np.asarray([0])
    lon = np.asarray([0])
    r = np.asarray([1])
    plotting.plot_polar(r=r, phi=lat, theta=lon)
