import trimesh as tri
import numpy as np
import pyshtools as pysh

from examples import construct_sphere as c


class Sphere:
    def __init__(self, radius=1.0, count=[32, 32]) -> None:
        self.radius = radius
        self.mesh = tri.creation.uv_sphere(radius=radius, count=count)
        self.points_cartesian = np.asarray(self.mesh.vertices)

        #vector_to_spherical returns points of form (phi, theta) = (azimuth, inclination) = (latitude, longitude) <- wrong
        self.points_polar = tri.util.vector_to_spherical(cartesian=self.points_cartesian)

        _, self.latitudes, self.longitudes = c.construct_rad_from_linspace()

    def spherical_harmonics_coeffs_cartesian(self, lmax=10):
        pass

    def sh_coeffs_lats(self):
        coeffs = pysh.expand.SHExpandLSQ(d=self.latitudes, lat=self.latitudes, lon=self.longitudes, lmax=self.latitudes.shape[0]/2-1)[0]
        return coeffs

    def sh_coeffs_lons(self):
        coeffs = pysh.expand.SHExpandLSQ(d=self.longitudes, lat=self.latitudes, lon=self.longitudes, lmax=self.latitudes.shape[0]/2-1)[0]
        return coeffs

    def sh_coeffs_radius(self):
        r, lat, lon = c.construct_degs_from_linspace()
        coeffs = pysh.expand.SHExpandLSQ(d=r, lat=lat, lon=lon, lmax=5)[0]
        return coeffs


