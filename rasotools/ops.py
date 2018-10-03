# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr

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

    alldates = pd.date_range(pd.Timestamp(dates.min()).replace(hour=np.min(times)),
                             pd.Timestamp(dates.max()).replace(hour=np.max(times)), freq=freq)

    shape = list(data.values.shape)
    shape[data.dims.index('date')] = alldates.size

    timeinfo = xr.DataArray(alldates.hour.values, coords=[alldates], dims=[datedim], name='sounding_times', attrs=coords[datedim].attrs)
    coords[datedim] = (datedim, alldates, coords[datedim].attrs)

    new = xr.DataArray(np.full(tuple(shape), np.nan), coords=coords, dims=data.dims, attrs=data.attrs)

    # find common
    itx = np.where(np.isin(dates, new[datedim].values))[0]
    jtx = np.where(np.isin(new[datedim].values, dates))[0]
    newindex = [slice(None)] * data.ndim
    oldindex = [slice(None)] * data.ndim
    newindex[data.dims.index(datedim)] = jtx
    oldindex[data.dims.index(datedim)] = itx
    new.values[newindex] = data.values[oldindex]
    timeinfo.values[~jtx] = np.nan   # no data yet

    for itime in new[datedim].values[~jtx]:
        diff = (itime - dates)/np.timedelta64(1, 'h')  # closest
        n = np.sum(np.abs(diff) < span)
        if n > 0:
            i = np.where(alldates == itime)[0]
            if n > 1:
                # count data?
                k = np.where(np.abs(diff) < span)[0]
                oldindex[data.dims.index(datedim)] = k
                counts = np.sum(np.isfinite(data.values[oldindex]), axis=data.dims.index(datedim)) / np.abs(diff[k])
                j = k[np.argmax(counts)]  # use the one with max data
            else:
                j = np.argmin(np.abs(diff))  # only one choice
            newindex[data.dims.index(datedim)] = i
            oldindex[data.dims.index(datedim)] = j
            new.values[newindex] = data.values[oldindex]
            timeinfo.values[i] = diff[j]   # value of minimum

    return new, timeinfo

