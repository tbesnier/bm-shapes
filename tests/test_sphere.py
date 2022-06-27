from utils.sphere import Sphere

def test_points_polar_ordered_same_as_points_cartesian():
    sphere = Sphere()
    polar = sphere.points_polar
    cart = sphere.points_cartesian

    