import numpy as np
import pyshtools as pysh


# Functions for constructing coordinates for the sphere in different ways

def construct_degs_from_sh_grid(grid='DH', extend=False):
    zero_grid = pysh.SHGrid.from_zeros(20, grid=grid, extend=extend)
    lats, lons = zero_grid.lats(), zero_grid.lons()
    return _make_grid(lats, lons)

def construct_degs_from_linspace(endpoint=False):
    lat_deg, lon_deg = np.linspace(90, -90, endpoint=endpoint), np.linspace(0, 360, endpoint=endpoint)
    return _make_grid(lat_deg, lon_deg)

def construct_degs_transposed(endpoint=False):
    r, lat_deg, lon_deg = construct_degs_from_linspace()
    return r, lat_deg, lon_deg.T

def construct_degs_scaled(endpoint=False):
    r, lat_deg, lon_deg = construct_degs_from_linspace()
    lat_scaled, lon_scaled = lat_deg/180, lon_deg/180
    return r, lat_scaled, lon_scaled

def construct_rad_from_linspace(grid_size=50, endpoint=False):
    lat_rad, lon_rad = np.linspace(np.pi, -np.pi, endpoint=endpoint, num=grid_size), np.linspace(0, 2*np.pi, endpoint=endpoint, num=grid_size)
    return _make_grid(lat_rad, lon_rad)

def _make_grid(lats, lons):
    grid = np.meshgrid(lats, lons, indexing='ij')
    lat_grid, lon_grid = grid[0], grid[1]
    r_grid = np.ones_like(lat_grid)
    return r_grid, lat_grid, lon_grid

# Functions for reconstructing functions on sphere using spherical harmonics basis

def reconstruct_function_dh(f, lats, lons):
    coeffs = pysh.expand.SHExpandDH(f)
    grid = pysh.expand.MakeGridDH(coeffs)
    return grid

def reconstruct_function_lsq(f, lats, lons):
    coeffs = pysh.expand.SHExpandLSQ(d=f, lat=lats, lon=lons, lmax=lons.shape[0]/2-1)[0]
    f_recon = pysh.expand.MakeGridPoint(coeffs, lat=lats, lon=lons)
    return f_recon

def reconstruct_function_shgrid(f, lats, lons):
    grid = pysh.SHGrid.from_array(f)
    coeffs = grid.expand()
    f_recon = coeffs.expand(lat=lats, lon=lons)
    return f_recon


def error(true, predicted):
    error = np.abs(true-predicted)
    print(np.max(error))
    return error


def add_noise_to_coefficients(coefficients, standard_deviation=0.1, mean= 0):
    random_vars = np.random.normal(loc=mean, scale=standard_deviation, size=coefficients.shape)
    zeroed_vars = np.where(coefficients==0, 0, random_vars)
    return coefficients + zeroed_vars



