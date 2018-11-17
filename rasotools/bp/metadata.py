# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr


def process_information(data, recognized_events, severity, as_break=1):
    # data ist eine tabelle ?
    # entweder es gibt eine ID spalte
    # pro ID wird hier eine Zeitreihe gerechnet
    # keep original stuff inside (message)
    #
    pass


def calculate_probability():
    # calc. timeseries and probability of a breakpoint from metadata changes
    # this can be combined with SNHT breakpoints
    pass


def location_change(lon, lat, ilon=None, ilat=None):
    from ..fun import distance

    if not isinstance(lon, xr.DataArray) and not isinstance(lat, xr.DataArray):
        raise ValueError("requires xarray DataArray")

    out = lon.copy()
    if ilon is None and ilat is None:
        distance = np.vectorize(distance)
        tmp = distance(lon[1:], lat[1:], lon[:-1], lat[:-1])  # distance between more recent and less recent
        tmp = tmp.append(tmp, tmp[-1])
        out[:] = tmp
        out.attrs['method'] = 'Backwards'
    else:
        out[:] = distance(lon, lat, ilon, ilat)
        out.attrs['method'] = 'Point(%f E, %f N)' % (ilon, ilat)
    out.attrs['units'] = 'km'

    # count occurence of each coordinate pair
    # use the most common (also the most recent?)
    # to estimate distance from,
    # era-interim has a distance of about 80km so only if larger it would make sense to split?

    return out
