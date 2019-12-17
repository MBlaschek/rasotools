# -*- coding: utf-8 -*-

__all__ = ['standard_datetime_hours']

"""
Wrap some functions from the modules to be available right here
"""


def standard_datetime_hours(data, dim='time', times=(0, 6, 12, 18), span=3, freq='6h', levels=None, **kwargs):
    from .std import align_datetime, to_hours
    from .. import config

    if levels is None:
        levels = config.std_plevels

    data = align_datetime(data.sel(plev=levels), dim=dim, times=times, span=span, freq=freq, **kwargs)
    return to_hours(data, dim=dim, times=times, **kwargs)


def access_odb_table(data, dim='time', pvar=None, sel=None):
    from numpy import unique
    from xarray import concat

    data = data.sel(**{dim: sel}).set_coords(pvar)
    print(unique(data[dim].values))
    tmp = dict(data.groupby(dim))
    for ikey in tmp.keys():
        tmp[ikey] = tmp[ikey].swap_dims({dim: pvar}).assign_coords({dim: ikey})
    data = concat(tmp.values(), dim=dim)
    return data
