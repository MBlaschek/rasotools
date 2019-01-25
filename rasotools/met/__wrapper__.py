# -*- coding: utf-8 -*-

__all__ = ['dataset_to_hours']


def dataset_to_hours(data, std=None, variables=None, dim='date', lev='pres', suffix='', interpolate=False, levels=None, verbose=0):
    from .time import standard_sounding_times, sel_hours
    from .convert import vertical_interpolation
    from ..fun import message
    from xarray import Dataset, concat
    from pandas import Index
    data = data.copy()
    new = Dataset()
    if std is not None:
        new[std + suffix], idx = standard_sounding_times(data[std], return_indices=True, verbose=verbose, level=1)
        message("Standard: ", std, new[std+suffix].shape, verbose=verbose)

    for i in list(data.data_vars):
        if i == std:
            continue
        if variables is not None and i not in variables:
            continue

        if interpolate and 'pres' in data[i].dims:
            tmp = vertical_interpolation(data[i], lev, levels=levels, verbose=verbose, level=1)
            new[i + suffix] = sel_hours(tmp)
        else:
            new[i + suffix] = sel_hours(data[i])

        if std is not None:
            # if dim in dims
            if dim in data[i].dims:
                inew = [slice(None)] * new[i + suffix].ndim
                iold = [slice(None)] * data[i].ndim
                inew[new[i + suffix].dims.index(dim)] = idx[:, 0]
                iold[data[i].dims.index(dim)] = idx[:, 1]
                new[i + suffix].values[tuple(inew)] = data[i].values[tuple(iold)]  # copy same

        message("Converted:", i + suffix, verbose=verbose)

    data = dict(new.groupby(dim + '.hour'))
    for ikey in data.keys():
        idata = data.pop(ikey)
        idata[dim].values = idata[dim].to_index().to_period('D').to_timestamp().values
        data[ikey] = idata

    data = concat(data.values(), dim=Index(data.keys(), name='hours'))
    if 'delay' in data.coords:
        data = data.reset_coords('delay').rename({'delay': 'delay' + suffix})

    return data
