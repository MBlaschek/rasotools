# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from . import grib
from . import helpers

__all__ = ['extract_locations']

"""
get xarrays and use them with these functions here

"""


def extract_locations(data, lon, lat, method='bilinear', raw=False, concat=None, debug=False, verbose=0):
    """ Extract location(s) from DataArray

    Args:
        data (DataArray) :
        lon (float, int, list):
        lat (float, int, list):
        method (str):
        raw (bool):
        debug (bool):
        verbose (int):

    Returns:
        list or DataArray
    """
    from .helpers import _bilinear_weights, _get_lonlat, _get_pos, _rectangle, _distance_weights

    if not isinstance(data, xr.DataArray):
        raise ValueError()

    if isinstance(lon, (int, float)):
        lon = [float(lon)]

    if isinstance(lat, (int, float)):
        lat = [float(lat)]

    if method not in ['point', 'bilinear', 'distance']:
        raise ValueError("Method unknown: point, bilinear, distance")

    locations = []
    lon = np.array(lon)
    lat = np.array(lat)
    name_lon, name_lat = _get_lonlat(list(data.dims))  # list
    if name_lat is None or name_lon is None:
        print(data.dims)
        raise ValueError("DataArray does not have named lon, lat coordinates")

    # need 0 to 360 deg
    xlons = data[name_lon].values
    if (xlons < 0).any():
        data[name_lon].values = np.where(xlons < 0, xlons + 360., xlons)
        print("Adjusting GRID Longitudes ...")

    # Input needs to be on the sam lon
    if (lon < 0).any():
        lon = np.where(lon < 0, lon + 360., lon)
        print("Adjusting INPUT Longitudes ...")

    order = data.dims   # tuple
    ilon = order.index(name_lon)
    ilat = order.index(name_lat)
    lons = data[name_lon].values[:]   # copy
    lats = data[name_lat].values[:]   # copy
    # get both axis
    iaxis = sorted([ilon, ilat])

    iorder = [i for i in order if i not in [name_lat, name_lon]]
    dattrs = dict(data.attrs)
    newcoords = {i: data[i].copy() for i in iorder}

    if method == 'points':
        npoints = 1
    else:
        npoints = 4

    for jlon, jlat in zip(lon, lat):
        try:
            # indices and weights for location
            indices = [slice(None)] * len(order)
            #
            #  Point or Bilinear interpolation from rectangle
            #
            if method == 'point':
                ix, iy = _get_pos(jlon, jlat, lons, lats)
                weights = np.array([1.])

            elif method == 'distance':
                ix, iy, weigths = _distance_weights(lons, lats, jlon, jlat)

            else:
                idx, points = _rectangle(jlon, jlat, lons, lats)
                ix, iy, weights = _bilinear_weights(jlon, jlat, idx, lons, lats)
                # Example for ERA-Interim GRID
                # _plot_weights(ix, iy, weights, data.values[0, 0, :, :], lons, lats, jlon, jlat,
                #               lonlat=ilon < ilat, dx=-np.diff(lons).mean() / 2., dy=np.diff(lats).mean() / 2.)
            indices[ilon] = ix
            indices[ilat] = iy
            #
            # get values
            #
            tmp = data.values[indices]
            if len(order) > 2:
                w = np.empty_like(tmp)
                w[::] = weights  # fill in
                weights = w

            if method != 'point':
                # Adjust Weights to shape and missing values
                weights = np.where(np.isnan(tmp), np.nan, weights)
                ws = np.nansum(weights, axis=iaxis[0], keepdims=True)
                w = np.where(ws == 0, np.nan, weights / ws)  # nansum changed !!
                weights = w

                tmp = np.nansum(tmp * weights, axis=iaxis[0])  # only the lower axis
            else:
                tmp = tmp * weights

            # results
            if not raw:
                dattrs['extract'] = method
                newcoords[name_lon] = jlon - 360. if jlon > 180 else jlon
                newcoords[name_lat] = jlat
                tmp = xr.DataArray(tmp, coords=newcoords, dims=iorder, name=data.name, attrs=dattrs)
            locations.append(tmp.copy())

        except Exception as e:
            if debug:
                raise e

            print(e)
            locations.append([])

    if len(locations) == 1:
        return locations[0]

    if concat is not None:
        if isinstance(concat, str):
            locations = xr.concat(locations, concat)
        else:
            locations = xr.concat(locations, concat)

    return locations
