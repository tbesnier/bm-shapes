import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from scipy.integrate import quad, dblquad

import numpy as np
import trimesh as tri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
import six

sys.path.insert(0,"../")
import pyssht

from utils import brownian_motion, sph_utils, mesh_processing

def make_grid(N):
    sphere = tri.creation.uv_sphere(count=[N, N])
    x, y ,z = np.array(sphere.vertices[:,0]), np.array(sphere.vertices[:,1]), np.array(sphere.vertices[:,2])
    (r, theta, phi) = pyssht.cart_to_spherical(x, y, z)
    return(theta, phi)

def sph_reconstr(x, y, z, n = 25):
    
    basis = sph_utils.SphHarmBasis(n_coeffs=n)

    coeffs_x = basis.sph_harm_transform(x)
    coeffs_y = basis.sph_harm_transform(y)
    coeffs_z = basis.sph_harm_transform(z)

    reconstr_function_x = my_basis.sph_harm_reconstruct(coeffs_x)
    reconstr_function_y = my_basis.sph_harm_reconstruct(coeffs_y)
    reconstr_function_z = my_basis.sph_harm_reconstruct(coeffs_z)
    
    return(reconstr_function_x, reconstr_function_y, reconstr_function_z)

def compute_wiener_process(x, y, z, theta, phi, Q, n = 25, t=1, n_step = 10, make_gif = True, file_dir = "tests", file_name = "test.gif"):
    
    """Compute Q-Wiener process on the spherical harmonic basis of a shape defined by (x,y,z) triplet of functions from the sphere

    Input: 
        - x,y,z: functions from the sphere (theta, phi)
        - n: resolution of the spherical harmonic decomposition (has to be squared number) [int]
        - t: end time of the process [float]
        - Q: Covariance matrix [n x n array]
        - n_step: number of steps of the process [int]
        - make_gif: tell if we want to output a gif file showing the process [Boolean]
    Output:
        - listpq: list of shape coordinates during the process [n_step x 3 x n array]
    """
    
    basis = sph_utils.SphHarmBasis(n_coeffs=n)

    coeffs_x = basis.sph_harm_transform(x)
    coeffs_y = basis.sph_harm_transform(y)
    coeffs_z = basis.sph_harm_transform(z)
    
    reconstr_stoch_x = basis.sph_harm_reconstruct_random(coeffs_x, Q, t, theta = theta, phi = phi, n_step = n_step)
    reconstr_stoch_y = basis.sph_harm_reconstruct_random(coeffs_y, Q, t, theta = theta, phi = phi, n_step = n_step)
    reconstr_stoch_z = basis.sph_harm_reconstruct_random(coeffs_z, Q, t, theta = theta, phi = phi, n_step = n_step)
    
    listpq = []
    
    for t in range(reconstr_stoch_x.shape[0]):
        (x_coord_stoch, y_coord_stoch, z_coord_stoch) = reconstr_stoch_x[t], reconstr_stoch_y[t], reconstr_stoch_z[t]
        listpq.append([x_coord_stoch, y_coord_stoch, z_coord_stoch])
        
    listpq = np.array(listpq)
    
    if make_gif==True:
        mesh_processing.create_gif(listpq, file_dir, file_name, auto_scale = True)
    
    return(listpq)

def compute_wiener_process_shape(x_mesh, y_mesh, z_mesh, faces, Q, n = 25, t=1, n_step=10, make_gif=True,                                  file_dir = "tests/wiener_process_shape", file_name = "test.gif", x_view=20, y_view=20, z_view=-10):
    
    (r_shape, theta_shape, phi_shape) = pyssht.cart_to_spherical(x_mesh, y_mesh, z_mesh)
    r_shape_proj = np.divide(r_shape, r_shape)
    #x_mesh_proj, y_mesh_proj, z_mesh_proj = pyssht.spherical_to_cart(r_shape_proj, theta_shape, phi_shape)
    
    def sphere_function_x(theta, phi):
        return np.sin(theta)*np.cos(phi)
    def sphere_function_y(theta, phi):
        return np.sin(theta)*np.sin(phi)
    def sphere_function_z(theta, phi):
        return np.cos(theta)
    
    sphere_process = compute_wiener_process(sphere_function_x, sphere_function_y, sphere_function_z,
                                         theta = theta_shape, phi = phi_shape, Q = Q, n = n, t=t, n_step=n_step, make_gif = False)
    
    shape_process = []
    for step in sphere_process:
        (r, theta_shape, phi_shape) = pyssht.cart_to_spherical(step[0], step[1], step[2])
        r_new = r - np.ones(r.shape) + r_shape
        (x_shape, y_shape, z_shape) = pyssht.spherical_to_cart(r_new, theta_shape, phi_shape)
        shape_process.append([x_shape, y_shape, z_shape])
    shape_process = np.array(shape_process)
    
    if make_gif==True:
        mesh_processing.create_gif_shape(list_data = shape_process, faces=faces, file_dir=file_dir, file_name=file_name, x_view=x_view,                                                  y_view=y_view, z_view=z_view, auto_scale = True)
    
    return(shape_process)


def compute_simplebridge(x0, y0, z0, x1, y1, z1, theta, phi, Q, n = 25, t=1, n_step = 10, make_gif = True,
                         file_dir = "tests", file_name = "test_bridge.gif"):
    """Compute Q-wiener bridge process on the spherical harmonic basis from a source shape defined by (x0,y0,z0) 
    triplet of functions from the sphere to a target shape defined by (x1,y1,z1) 

    Input: 
        - x0,y0,z0,x1,y1,z1: functions from the sphere (theta, phi)
        - n: resolution of the spherical harmonic decomposition (has to be squared number) [int]
        - t: end time of the process [float]
        - Q: Covariance matrix [n x n array]
        - n_step: number of steps of the process [int]
        - make_gif: tell if we want to output a gif file showing the process [Boolean]
    Output:
        - listpq: list of shape coordinates during the process [n_step x 3 x n array]
    """
    basis = sph_utils.SphHarmBasis(n_coeffs=n)
    
    coeffs_x_source = basis.sph_harm_transform(x0)
    coeffs_y_source = basis.sph_harm_transform(y0)
    coeffs_z_source = basis.sph_harm_transform(z0)

    coeffs_x_target = basis.sph_harm_transform(x1)
    coeffs_y_target = basis.sph_harm_transform(y1)
    coeffs_z_target = basis.sph_harm_transform(z1)
    
    reconstr_function_x = basis.sph_harm_reconstruct_bridge(coeffs_x_source, coeffs_x_target,
                                                           t = t, theta = theta, phi = phi, Q = Q, n_step = 50)[0]
    reconstr_function_y = basis.sph_harm_reconstruct_bridge(coeffs_y_source, coeffs_y_target,
                                                           t = t, theta = theta, phi = phi, Q = Q, n_step = 50)[0]
    reconstr_function_z = basis.sph_harm_reconstruct_bridge(coeffs_z_source, coeffs_z_target,
                                                           t = t, theta = theta, phi = phi, Q = Q, n_step = 50)[0]
   
    listpq = []
    for t in range(reconstr_function_x.shape[0]):
        (x_coord_stoch, y_coord_stoch, z_coord_stoch) = reconstr_function_x[t], reconstr_function_y[t], reconstr_function_z[t]
        listpq.append([x_coord_stoch, y_coord_stoch, z_coord_stoch])
        
    listpq = np.array(listpq)
    
    mesh_processing.create_gif(listpq, file_dir, file_name, auto_scale = False)
    
    return(listpq)
    