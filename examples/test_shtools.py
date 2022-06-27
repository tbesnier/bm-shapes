import pyshtools as pysh
import numpy as np
import trimesh as tri
import matplotlib.pyplot as plt
import pyssht


def plot_mesh(mesh: tri.Trimesh, title: str = "Mesh"):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2])
    ax.set_title(title)
    plt.show()

def plot_coords(x_coords: np.array, y_coords: np.array, z_coords: np.array, title: str = "Coords"):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords)
    ax.set_title(title)
    plt.show()

def plot_polar(r: np.ndarray, theta: np.array, phi: np.array, title= "Polar"):
    xyz = pyssht.spherical_to_cart(r, theta, phi)
    plot_coords(xyz[0], xyz[1], xyz[2], title)

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

class Sphere:
    def __init__(self, radius=1.0, count=[32, 32]) -> None:
        self.radius = radius
        self.mesh = tri.creation.uv_sphere(radius=radius, count=count)
        self.cartesian_coords = self.mesh.vertices
        self.polar_coords = tri.util.vector_to_spherical(cartesian=self.cartesian_coords)
        

if __name__ == "__main__":
    sphere = Sphere()
    phi = sphere.polar_coords[:, 0]
    theta = sphere.polar_coords[:, 1]

    lat=theta*180/np.pi
    lon=phi*180/np.pi

    sample_lat = np.linspace(0, 180, 100)
    sample_lon = np.linspace(0, 360, 100)

    x_coeffs = pysh.expand.SHExpandLSQ(d=sphere.cartesian_coords[:, 0], lat=lat, lon=lon, lmax=6)
    y_coeffs = pysh.expand.SHExpandLSQ(d=sphere.cartesian_coords[:, 1], lat=lat, lon=lon, lmax=6)
    z_coeffs = pysh.expand.SHExpandLSQ(d=sphere.cartesian_coords[:, 2], lat=lat, lon=lon, lmax=6)

    x_recon = pysh.expand.MakeGridPoint(x_coeffs[0], lat=lat, lon=lon)
    y_recon = pysh.expand.MakeGridPoint(y_coeffs[0], lat=lat, lon=lon)
    z_recon = pysh.expand.MakeGridPoint(z_coeffs[0], lat=lat, lon=lon)

    lat_coeffs = pysh.expand.SHExpandLSQ(lat, lat=lat, lon=lon, lmax=6)
    lon_coeffs = pysh.expand.SHExpandLSQ(lon, lat=lat, lon=lon, lmax=6)
    r_coeffs = pysh.expand.SHExpandLSQ(np.ones(shape=lat.shape), lat=lat, lon=lon, lmax=6)

    lat_recon = pysh.expand.MakeGridPoint(lat_coeffs[0], lat=lat, lon=lon)
    lon_recon = pysh.expand.MakeGridPoint(lon_coeffs[0], lat=lat, lon=lon)
    r_recon = pysh.expand.MakeGridPoint(r_coeffs[0], lat=lat, lon=lon)

    lat_noisy_coeff = add_noise_to_coefficients(lat_coeffs[0])
    lon_noisy_coeff = add_noise_to_coefficients(lon_coeffs[0])
    r_noisy_coeff = add_noise_to_coefficients(r_coeffs[0])

    lat_noisy = pysh.expand.MakeGridPoint(lat_noisy_coeff, lat=lat, lon=lon)
    lon_noisy = pysh.expand.MakeGridPoint(lon_noisy_coeff, lat=lat, lon=lon)
    r_noisy = pysh.expand.MakeGridPoint(r_noisy_coeff, lat=lat, lon=lon)

    (x, y, z) = pyssht.spherical_to_cart(r_noisy, lat_noisy, lon_noisy)


    plot_coords(sphere.cartesian_coords[:, 0], sphere.cartesian_coords[:, 1], sphere.cartesian_coords[:, 2], title="Original")
    plot_coords(x_recon, y_recon, z_recon, title="Reconstructed cart")
    plot_polar(r= r_recon, phi=lat_recon, theta=lon_recon, title="Reconstructed polar")
    plot_coords(x, y, z, "With Noise")

    recon_mesh = tri.Trimesh(vertices=np.stack((x, y, z), -1), faces=sphere.mesh.faces)

 #   mesh_recons = tri.convex.convex_hull(np.array([x_coord_recons, y_coord_recons, z_coord_recons]).T)


    plot_mesh(recon_mesh)