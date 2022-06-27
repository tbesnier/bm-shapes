import matplotlib.pyplot as plt
import trimesh as tri
import numpy as np
import pyssht

# theta = longitude = left to right: [0, pi]
# phi = latitude = up/down: [0, 2*pi]

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
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_polar(r: np.ndarray, theta: np.array, phi: np.array, title= "Polar"):
    xyz = pyssht.spherical_to_cart(r, theta, phi)
    plot_coords(xyz[0], xyz[1], xyz[2], title)