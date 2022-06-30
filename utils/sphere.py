import trimesh as tri
import numpy as np
import pyshtools as pysh

def radians_to_degrees(radians: np.ndarray):
    return radians*180/np.pi

def degrees_to_radians(degrees: np.ndarray):
    return degrees*np.pi/180

def compute_coefficients(f_1, f_2, f_3, lat, lon, lmax=10):
    """Given a function f: S^2 -> R^3, takes a list of samples of the function f(lat, lon) = (f_1, f_2, f_3) 
    and returns the coefficients for the spherical harmonic basis."""
    coeffs_1 = pysh.expand.SHExpandLSQ(d=f_1, lat=lat, lon=lon, lmax=lmax)
    coeffs_2 = pysh.expand.SHExpandLSQ(d=f_2, lat=lat, lon=lon, lmax=lmax)
    coeffs_3 = pysh.expand.SHExpandLSQ(d=f_3, lat=lat, lon=lon, lmax=lmax)
    return coeffs_1[0], coeffs_2[0], coeffs_3[0]


class Sphere:
    def __init__(self, radius=1.0, count=[32, 32]) -> None:
        self.radius = radius
        self.mesh = tri.creation.uv_sphere(radius=radius, count=count)
        self.points_cartesian = np.asarray(self.mesh.vertices)

        #vector_to_spherical returns points of form (phi, theta) = (azimuth, inclination) = (latitude, longitude)
        self.points_polar = tri.util.vector_to_spherical(cartesian=self.points_cartesian)

        # latitudes have values in [-pi, pi]
        self.latitudes = self.points_polar[:, 0]

        # longitudes have values in [0, pi]
        self.longitudes = self.points_polar[:, 1]

    def spherical_harmonics_coeffs_cartesian(self, lmax=10):
        pass

    def spherical_harmonics_coeffs_polar(self, lmax=10):
        pass

    

    def _convert_latitudes_and_longitudes(self):
        """Latitudes need to be converted from radians with values in [-pi, pi] to degrees with values in [-90, 90].
        Longitudes need to be converted from radians with values in [0, pi] to degrees with values in [0, 360]"""        
        lat_degs = radians_to_degrees(self.latitudes)
        lon_degs = radians_to_degrees(self.longitudes)
        lat, lon = self._latitudes_less_than_minus_90(lat_degs, lon_degs)
        lat, lon = self._latitudes_greater_than_90(lat, lon)
        lon = self._longitudes_less_than_0(lon)
        return lat, lon

    def _latitudes_less_than_minus_90(self, latitudes, longitudes):
        new_latitudes = np.where(latitudes<-90, latitudes+180, latitudes)
        new_longitudes = np.where(latitudes<-90, -longitudes, longitudes)
        return new_latitudes, new_longitudes

    def _latitudes_greater_than_90(self, latitudes, longitudes):
        new_latitudes = np.where(latitudes>90, latitudes-180, latitudes)
        new_longitudes = np.where(latitudes>90, -longitudes, longitudes)
        return new_latitudes, new_longitudes

    def _longitudes_less_than_0(self, longitudes):
        new_longitudes = np.where(longitudes<0, longitudes+360, longitudes)
        return new_longitudes
        

