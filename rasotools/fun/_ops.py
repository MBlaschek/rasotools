# -*- coding: utf-8 -*-

__all__ = ['array2dataset']


def array2dataset(data, dim, rename=None):
    """ Convert a DataArray to Dataset and consider dependent coordinates

    Args:
        data (DataArray): Input DataArray
        dim (str): Coordinate to use as variables
        rename (dict): renaming policy

    Returns:
        Dataset :
    """
    from xarray import DataArray

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    data = data.copy()
    tmp = {}

    for i, j in data.coords.items():
        if dim in j.dims and i != dim:
            tmp[i] = data[i]
            data = data.drop(i)

    data = data.to_dataset(dim=dim)
    for i, j in tmp.items():
        data[i] = j  # add as Coords / Variables

    if rename is not None:
        return data.rename(rename)

    return data

