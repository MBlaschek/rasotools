# -*- coding: utf-8 -*-


def xarray_function_wrapper(x, wfunc=None, **kwargs):
    """ Map a numpy function that is not currently in xarray to xarray with apply_ufunc

    Args:
        x (DataArray): Input data
        wfunc (callable): function to call, e.g.: np.nanpercentile
        **kwargs: all arguments to function

    Keyword Args:
        dim (str): Dimension to remove
        axis (int): axis for numpy functions
        debug (bool): show debug information of call

    Returns:
        DataArray : result of function call retains attrs

    Examples:
        >>> def myfunc(x, **kwargs):
        >>>     return np.isfinite(x).sum(**kwargs)
        >>> data = xr.DataArray(np.random.randn(1000,2), dims=('time','lev'), coords=[pd.date_range('1-1-2019', periods=1000), [10, 12]])
        >>> xarray_function_wrapper(data, wfunc=myfunc, dim='time', axis=0)
    """
    import xarray as xr
    if not isinstance(x, xr.DataArray):
        raise ValueError('requires a DataArray')

    jdims = list(x.dims)
    if 'dim' in kwargs.keys():
        jdims.remove(kwargs.pop('dim'))
        if 'axis' not in kwargs.keys():
            raise RuntimeWarning('axis keyword not present')
            # kwargs['axis'] = x.dims.index(kwargs['dim'])   # add axis keyword

    if kwargs.pop('debug', False):
        print(x.dims, x.shape, wfunc)

    return xr.apply_ufunc(wfunc, x, kwargs=kwargs,
                          input_core_dims=[x.dims], output_core_dims=[jdims],
                          keep_attrs=True)


def set_attrs(data, name, set='', add='', default=''):
    from xarray import Dataset, DataArray
    if not isinstance(data, (Dataset, DataArray)):
        raise ValueError("Requires an XArray Dataset or Array")

    if name not in data.attrs:
        data.attrs[name] = default
    elif set != '':
        data.attrs[name] = set
    else:
        data.attrs[name] += add


def idx2shp(idx, axis, shape):
    index = [slice(None)] * len(shape)
    index[axis] = idx
    return tuple(index)
