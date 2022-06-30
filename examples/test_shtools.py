import pyshtools as pysh
import numpy as np
import trimesh as tri
import pyssht

from utils import plotting
from utils.sphere import Sphere, radians_to_degrees, degrees_to_radians


def compute_coefficients(f_1, f_2, f_3, lat, lon, lmax=10):
    """Given a function f: S^2 -> R^3, takes a list of samples of the function f(lat, lon) = (f_1, f_2, f_3) 
    and returns the coefficients for the spherical harmonic basis."""
    coeffs_1 = pysh.expand.SHExpandLSQ(d=f_1, lat=lat, lon=lon, lmax=lmax)
    coeffs_2 = pysh.expand.SHExpandLSQ(d=f_2, lat=lat, lon=lon, lmax=lmax)
    coeffs_3 = pysh.expand.SHExpandLSQ(d=f_3, lat=lat, lon=lon, lmax=lmax)
    return coeffs_1[0], coeffs_2[0], coeffs_3[0]

def add_noise_to_coefficients(coefficients, standard_deviation=0.1, mean= 0):
    random_vars = np.random.normal(loc=mean, scale=standard_deviation, size=coefficients.shape)
    zeroed_vars = np.where(coefficients==0, 0, random_vars)
    return coefficients + zeroed_vars
        

if __name__ == "__main__":
    sphere = Sphere()

    # lat_deg, lon_deg = sphere._convert_latitudes_and_longitudes()
    lat_deg, lon_deg = sphere.latitudes, sphere.longitudes

    x_coeffs = pysh.expand.SHExpandLSQ(d=sphere.points_cartesian[:, 0], lat=lat_deg, lon=lon_deg, lmax=1)
    y_coeffs = pysh.expand.SHExpandLSQ(d=sphere.points_cartesian[:, 1], lat=lat_deg, lon=lon_deg, lmax=1)
    z_coeffs = pysh.expand.SHExpandLSQ(d=sphere.points_cartesian[:, 2], lat=lat_deg, lon=lon_deg, lmax=1)

    x_recon = pysh.expand.MakeGridPoint(x_coeffs[0], lat=lat_deg, lon=lon_deg)
    y_recon = pysh.expand.MakeGridPoint(y_coeffs[0], lat=lat_deg, lon=lon_deg)
    z_recon = pysh.expand.MakeGridPoint(z_coeffs[0], lat=lat_deg, lon=lon_deg)

    lat_coeffs = pysh.expand.SHExpandLSQ(lat_deg, lat=lat_deg, lon=lon_deg, lmax=1)
    lon_coeffs = pysh.expand.SHExpandLSQ(lon_deg, lat=lat_deg, lon=lon_deg, lmax=1)
    r_coeffs = pysh.expand.SHExpandLSQ(np.ones(shape=lat_deg.shape), lat=lat_deg, lon=lon_deg, lmax=1)

    lat_recon = pysh.expand.MakeGridPoint(lat_coeffs[0], lat=lat_deg, lon=lon_deg)
    lon_recon = pysh.expand.MakeGridPoint(lon_coeffs[0], lat=lat_deg, lon=lon_deg)
    r_recon = pysh.expand.MakeGridPoint(r_coeffs[0], lat=lat_deg, lon=lon_deg)

    # lat_noisy_coeff = add_noise_to_coefficients(lat_coeffs[0])
    # lon_noisy_coeff = add_noise_to_coefficients(lon_coeffs[0])
    # r_noisy_coeff = add_noise_to_coefficients(r_coeffs[0])

    # lat_noisy = pysh.expand.MakeGridPoint(lat_noisy_coeff, lat=lat, lon=lon)
    # lon_noisy = pysh.expand.MakeGridPoint(lon_noisy_coeff, lat=lat, lon=lon)
    # r_noisy = pysh.expand.MakeGridPoint(r_noisy_coeff, lat=lat, lon=lon)

    # (x, y, z) = pyssht.spherical_to_cart(r_noisy, lat_noisy, lon_noisy)


    plotting.plot_coords(sphere.points_cartesian[:, 0], sphere.points_cartesian[:, 1], sphere.points_cartesian[:, 2], title="Original")
    plotting.plot_coords(x_recon, y_recon, z_recon, title="Reconstructed cart")
    plotting.plot_polar(r= r_recon, phi=lat_recon, theta=lon_recon, title="Reconstructed polar")
    # plotting.plot_coords(x, y, z, "With Noise")

    # recon_mesh = tri.Trimesh(vertices=np.stack((x, y, z), -1), faces=sphere.mesh.faces)

 #   mesh_recons = tri.convex.convex_hull(np.array([x_coord_recons, y_coord_recons, z_coord_recons]).T)


    # plotting.plot_mesh(recon_mesh)