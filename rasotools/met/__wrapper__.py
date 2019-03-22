# -*- coding: utf-8 -*-

__all__ = ['dataset_to_hours']


def dataset_to_hours(data, std=None, variables=None, dim='date', lev='pres', hour='hour', suffix='', interpolate=False, levels=None,
                     **kwargs):
    """ convert all DataArrays in the Dataset to hours

    Args:
        data (Dataset): xarray Dataset
        std (str): name of DataArray to act as target for the rest
        variables (list): variables to consider
        dim (str): datetime dimension
        lev (str): pressure level dimension
        suffix (str): suffix for new Dataset
        interpolate (bool): interpolate along pressure levels
        levels (list): interpolation pressure levels
        **kwargs
    Returns:
        Dataset : rearanged and standardized Dataset
    """
    from .time import standard_sounding_times, sel_hours
    from .convert import vertical_interpolation
    from ..fun import message, update_kw
    from xarray import Dataset, concat
    from pandas import Index
    data = data.copy()
    new = Dataset()
    if std is not None:
        new[std + suffix], idx = standard_sounding_times(data[std], return_indices=True, dim=dim,
                                                         **update_kw('level', 1, **kwargs))
        message("Standard: ", std, new[std + suffix].shape, **kwargs)

    for i in list(data.data_vars):
        if i == std:
            continue

        if variables is not None and i not in variables:
            continue

        if interpolate and lev in data[i].dims:
            tmp = vertical_interpolation(data[i], lev, levels=levels, **update_kw('level', 1, **kwargs))
            new[i + suffix] = sel_hours(tmp, dim=dim, **update_kw('level', 1, **kwargs))
        else:
            new[i + suffix] = sel_hours(data[i], dim=dim, **update_kw('level', 1, **kwargs))

        if std is not None and idx.size > 0:
            # if dim in dims
            if dim in data[i].dims:
                inew = [slice(None)] * new[i + suffix].ndim
                iold = [slice(None)] * data[i].ndim
                inew[new[i + suffix].dims.index(dim)] = idx[:, 1]
                iold[data[i].dims.index(dim)] = idx[:, 0]
                new[i + suffix].values[tuple(inew)] = data[i].values[tuple(iold)]  # copy same

        message("Converted:", i + suffix, **kwargs)

    data = dict(new.groupby(dim + '.hour'))
    for ikey in data.keys():
        idata = data.pop(ikey)
        idata[dim].values = idata[dim].to_index().to_period('D').to_timestamp().values
        data[ikey] = idata

    data = concat(data.values(), dim=Index(data.keys(), name=hour))
    if 'delay' in data.coords:
        data = data.reset_coords('delay').rename({'delay': 'delay' + suffix})

    return data
