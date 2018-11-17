# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
from .fun import message

__all__ = ['switch_dim']


def switch_dim(data, neworder):
    """ Function to rollaxis (numpy)

    Args:
        data (DataArray) : data
        neworder (list, tuple): new dimension order

    Returns:
        DataArray :
    """
    if not isinstance(data, xr.DataArray):
        raise ValueError("Requires a DataArray")
    old = list(data.dims)
    neworder = list(neworder)
    for idim in neworder:
        old_pos = old.index(idim)
        new_pos = neworder.index(idim)
        other = old[new_pos]
        data = np.swapaxes(data, old_pos, new_pos)
        old[new_pos] = idim
        old[old_pos] = other
    return data


def standard_sounding_times(data, times=[0, 12], span=12, freq='12h', datedim='date', **kwargs):
    if not isinstance(data, xr.DataArray):
        raise ValueError("Requires a DataArray")
    if datedim not in data.dims:
        raise ValueError('Datetime dimension called %s?' % datedim)
    # add hour as dimension, and add hour as dimension with real times

    coords = dict(data.coords)
    dates = data[datedim].values.copy()

    #  all possible dates
    alldates = pd.date_range(pd.Timestamp(dates.min()).replace(hour=np.min(times)),
                             pd.Timestamp(dates.max()).replace(hour=np.max(times)), freq=freq)

    shape = list(data.values.shape)
    shape[data.dims.index('date')] = alldates.size

    # original time information
    timeinfo = xr.DataArray(alldates.hour.values, coords=[alldates], dims=[datedim], name='sounding_times',
                            attrs=coords[datedim].attrs)
    coords[datedim] = (datedim, alldates, coords[datedim].attrs)

    new = xr.DataArray(np.full(tuple(shape), np.nan), coords=coords, dims=data.dims, attrs=data.attrs)

    # find common
    itx = np.where(np.isin(dates, new[datedim].values))[0]  # dates fitting newdates (indices)
    jtx = np.where(np.isin(new[datedim].values, dates))[0]  # newdates fitting dates (indices)
    newindex = [slice(None)] * data.ndim
    oldindex = [slice(None)] * data.ndim
    idate = data.dims.index(datedim)
    newindex[idate] = jtx
    oldindex[idate] = itx
    new.values[newindex] = data.values[oldindex]  # transfer data of fitting dates
    timeinfo.values[~jtx] = np.nan  # not fitting dates

    # All times not yet filled
    # Is there some data that fits within the given time window
    for itime in new[datedim].values[~jtx]:
        diff = (itime - dates) / np.timedelta64(1, 'h')  # closest sounding
        n = np.sum(np.abs(diff) < span)  # number of soundings within time window
        if n > 0:
            i = np.where(alldates == itime)[0]  # index for new array
            if n > 1:
                # many choices, count data
                k = np.where(np.abs(diff) < span)[0]
                oldindex[idate] = k
                # count data of candidates
                # weight by time difference (assuming, that a sounding at the edge of the window is less accurate)
                counts = np.sum(np.isfinite(data.values[oldindex]), axis=idate) / np.abs(diff[k])
                j = k[np.argmax(counts)]  # use the one with max data / min time diff
            else:
                j = np.argmin(np.abs(diff))  # only one choice
            newindex[idate] = i
            oldindex[idate] = j
            new.values[newindex] = data.values[oldindex]  # update data array
            timeinfo.values[i] = diff[j]  # datetime of minimum

    return new, timeinfo


def split_by_location(data, longitudes, latitudes, maxdist=20, ilon=None, ilat=None, **kwargs):
    from .fun import distance
    if not isinstance(data, xr.Dataset):
        raise ValueError("requires a dataset")

    if longitudes not in data.data_vars:
        raise ValueError('longitudes not found')
    if latitudes not in data.data_vars:
        raise ValueError('latitudes not found')

    locs = data[[longitudes, latitudes]].to_dataframe()  # has time
    nfo = locs.groupby(by=['lon', 'lat']).size().reset_index().rename(columns={0: 'counts'})
    if ilon is None or ilat is None:
        imax = nfo.idxmax()['counts']
        ilon = nfo.lon[imax]
        ilat = nfo.lat[imax]
    message("using: ", ilon, ilat, mname='SPLIT', **kwargs)
    nfo['distance'] = distance(nfo.lon, nfo.lat, ilon, ilat)
    if (nfo['distance'] > maxdist).any():
        message("Found significant changes in location of radiosonde", mname='SPLIT', **kwargs)
        message(nfo, **kwargs)

    pass
