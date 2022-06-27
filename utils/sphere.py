import trimesh as tri

class Sphere:
    def __init__(self, radius=1.0, count=[32, 32]) -> None:
        self.radius = radius
        self.mesh = tri.creation.uv_sphere(radius=radius, count=count)
        self.points_cartesian = self.mesh.vertices
        self.points_polar = tri.util.vector_to_spherical(cartesian=self.points_cartesian)