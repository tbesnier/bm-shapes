from math import cos, sin
import numpy as np
from utils.sphere import Sphere, radians_to_degrees, degrees_to_radians

def test_points_polar_ordered_same_as_points_cartesian():
    sphere = Sphere()
    polar = sphere.points_polar
    cart = sphere.points_cartesian

    latitude = polar[800][0]
    longitude = polar[800][1]
    x = sphere.radius*cos(latitude)*sin(longitude)
    y = sphere.radius*sin(latitude)*sin(longitude)
    z = sphere.radius*cos(longitude)

    assert x == cart[800][0]
    assert y == cart[800][1]
    assert z == cart[800][2]

def test_latitudes():
    sphere = Sphere()
    latitudes = sphere.points_polar[:, 0]
    assert np.all(-np.pi <= latitudes)
    assert np.all(latitudes <= np.pi)
    assert np.any(latitudes <= 0)
    assert np.any(latitudes>0)

def test_longitude():
    sphere = Sphere()
    longitudes = sphere.points_polar[:, 1]
    assert np.all(0 <= longitudes)
    assert np.all(longitudes <= np.pi)

def test_radians_to_degrees():
    radians = np.array([np.pi/2])
    degrees = np.array([90])
    assert radians_to_degrees(radians) == degrees

def test_degrees_to_radians():
    radians = np.array([np.pi/2])
    degrees = np.array([90])
    assert degrees_to_radians(degrees) == radians

def test_convert_latitudes_and_longitudes_equivalent():
    sphere = Sphere()
    converted_latitudes, converted_longitudes = sphere._convert_latitudes_and_longitudes()
    x_out, y_out, z_out = polar_to_cartesian(1, np.deg2rad(converted_latitudes),
                                                 np.deg2rad(converted_longitudes))
    assert np.allclose(x_out, sphere.points_cartesian[:, 0])
    assert np.allclose(y_out, sphere.points_cartesian[:, 1])
    assert np.allclose(z_out, sphere.points_cartesian[:, 2])

def test_convert_latitudes_and_longitudes_range():
    sphere = Sphere()
    converted_latitudes, converted_longitudes = sphere._convert_latitudes_and_longitudes()
    assert np.all(converted_latitudes <= 90)
    assert np.all(converted_latitudes >= -90)
    assert np.any(converted_latitudes > 2*np.pi)

    assert np.all(converted_longitudes <= 360)
    assert np.all(converted_longitudes >= 0)
    assert np.any(converted_longitudes > 2*np.pi)
    

def polar_to_cartesian(radius, latitudes, longitudes):
        x = radius*np.cos(latitudes)*np.sin(longitudes)
        y = radius*np.sin(latitudes)*np.sin(longitudes)
        z = radius*np.cos(longitudes)
        return x, y, z
