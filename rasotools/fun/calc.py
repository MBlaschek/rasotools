# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['nanrange', 'nancount', 'fuzzy_equal', 'fuzzy_all', 'distance']


def nanrange(x, axis=0):
    """ Calculate min and max removing NAN

    Args:
        x (ndarray): input data
        axis (int): axis

    Returns:
        tuple : min, max
    """
    return np.nanmin(x, axis=axis), np.nanmax(x, axis=axis)


def nancount(x, axis=0, keepdims=False):
    """ Count values finite

    Args:
        x (array):
        axis (int):
        keepdims (bool):

    Returns:

    """
    return np.sum(np.isfinite(x), axis=axis, keepdims=keepdims)


def fuzzy_all(x, axis=0, thres=2):
    if np.sum(x, axis=axis) > (np.shape(x)[axis] / np.float(thres)):
        return True
    else:
        return False


def fuzzy_equal(x, y, z):
    return (y < (x + z)) & (y > (x - z))


def distance(lon, lat, ilon, ilat):
    """
    Calculates the distance between a point and an array of points
    Distance in kilometers

    Parameters
    ----------
    lon     Longitudes of points
    lat     Latitudes of points
    ilon    Longitude of Position
    ilat    Latitude of Position

    Returns
    -------
    numpy.array / same as input

    Notes
    -----
    Haversine Formula
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    rad_factor = np.pi / 180.0  # for trignometry, need angles in radians
    # implicit copyies !
    lat *= rad_factor
    lon *= rad_factor
    ilat *= rad_factor
    ilon *= rad_factor
    #
    dlon = lon - ilon
    dlat = lat - ilat
    # vector + vector * value * vector
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(ilat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r
