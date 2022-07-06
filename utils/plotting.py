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
    return fig

def plot_polar_rad(r: np.ndarray, lats: np.array, lons: np.array, title= "Polar"):
    x, y, z = polar_to_cartesian(r, lats, lons)
    plot_coords(x, y, z, title)
    return x, y, z

def plot_polar_deg(r: np.ndarray, lats: np.array, lons: np.array, title= "Polar"):
    lats_rad, lons_rad = np.deg2rad(lats), np.deg2rad(lons)
    return plot_polar_rad(r, lats_rad, lons_rad)

def polar_to_cartesian(radius, latitudes, longitudes):
    x = radius*np.cos(latitudes)*np.sin(longitudes)
    y = radius*np.sin(latitudes)*np.sin(longitudes)
    z = radius*np.cos(longitudes)
    return x, y, z