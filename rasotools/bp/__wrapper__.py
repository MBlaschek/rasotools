# -*- coding: utf-8 -*-
import numpy as np
from xarray import Dataset, DataArray
from ..fun import message

__all__ = ['detect_breakpoints', 'snht', 'adjustments', 'any_breakpoints', 'breakpoint_statistics', 'get_breakpoints']


def snht(data, dim='date', var=None, dep=None, window=1460, missing=600, **kwargs):
    """ Calculate a Standard Normal Homogeinity Test

    Args:
        data (DataArray, Dataset):
        dim (str): datetime dimension
        var (str): variable if Dataset
        dep (str, DataArray): departure variable
        window (int): running window (timesteps)
        missing (int): allowed missing values in window
        **kwargs:

    Returns:
        Dataset : test statistics
    """
    from .det import test

    if not isinstance(data, (DataArray, Dataset)):
        raise ValueError("Require a DataArray / Dataset object")

    if isinstance(data, DataArray):
        if data.name is None:
            raise ValueError("DataArray needs a var")
        idata = data.copy()
    else:
        ivars = list(data.data_vars)
        if len(ivars) == 1:
            var = ivars[0]
        elif var is None:
            raise ValueError("Dataset requires a var")
        else:
            pass
        idata = data[var].copy()

    if dim not in idata.dims:
        raise ValueError('requires a datetime dimension', dim)

    axis = idata.dims.index(dim)
    attrs = idata.attrs.copy()

    if dep is not None:
        if isinstance(dep, str) and isinstance(data, Dataset):
            dep = data[dep]
        elif isinstance(dep, DataArray):
            dep = dep
        else:
            raise ValueError("dep var not present")

        idata = (idata - dep)
        attrs['cell_method'] = 'departure ' + dep.name + attrs.get('cell_method', '')
        idata.attrs.update(attrs)

    stest = np.apply_along_axis(test, axis, idata.values, window, missing)
    attrs.update({'units': '1', 'window': window, 'missing': missing})

    if isinstance(data, DataArray):
        iname = data.name
        data = data.to_dataset()
    else:
        iname = var

    if dep is not None:
        data[iname + '_dep'] = idata

    data[iname + '_snht'] = (list(data.dims), stest)
    data[iname + '_snht'].attrs.update(attrs)
    return data



def combine_breakpoints():
    # use snht break points
    # use sonde type changes
    # use documentation changes
    # use radiosonde intercomparison data to adjust radiosonde types?
    # how to weight these changes
    # probability of a breakpoint ?
    pass


def detect_breakpoints(data, name=None, axis=0, thres=50, dist=730, min_levels=3, ensemble=False, **kwargs):
    """ Break Detection in timeseries using a Standard Normal Homogeneity Test (SNHT)

    Parameters
    ----------
    data : xr.DataArray / xr.Dataset
        Input Radiosonde Data (standard)
    name : str
        name of variable in Dataset
    axis : int
        datetime axis
    thres : int
        threshold for SNHT
    dist : int
        distance between breakpoints
    min_levels : int
        minimum required number of significant levels
    ensemble : bool
        use ensemble of 50 thresholds
    kwargs : dict

    Returns
    -------
    xr.Dataset
        xarray Dataset with variable name_breaks
    """
    from .det import detector, detector_ensemble
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise ValueError("Require a DataArray / Dataset object")

    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError("DataArray needs a name")
        values = data.values
    else:
        ivars = list(data.data_vars)
        if len(ivars) == 1:
            name = ivars[0]
        elif name is None:
            raise ValueError("Dataset requires a name")
        else:
            pass
        values = data[name].values.copy()

    if values.ndim > 2:
        raise ValueError("Too many dimensions, please iterate")

    params = {'units': '1', 'thres': thres, 'dist': dist, 'min_levels': min_levels}

    if ensemble:
        breaks = detector_ensemble(values, axis=axis, nthres=50, **kwargs)
        params['thres'] = 'ens50'
    else:
        breaks = detector(values, axis=axis, dist=dist, thres=thres, min_levels=min_levels, **kwargs)

    if isinstance(data, xr.DataArray):
        iname = data.name
        data = data.to_dataset()
    else:
        iname = name

    data[iname + '_breaks'] = (list(data.dims), breaks)
    data[iname + '_breaks'].attrs.update(params)
    return data


def any_breakpoints(data, name=None, axis=0, **kwargs):
    """ Check for breakpoints

    Parameters
    ----------
    data : xr.DataArray / xr.Dataset
    name : str
    axis : int
    kwargs : dict

    Returns
    -------
    bool
        Any breakpoints?
    """

    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise ValueError("Require a DataArray / Dataset object")

    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError("DataArray needs a name")
        i, d, n = get_breakpoints(data, axis=axis, dates=True, nlevs=True)
    else:
        ivars = list(data.data_vars)
        if len(ivars) == 1:
            name = ivars[0]
        elif name is None:
            raise ValueError("Dataset requires a name")
        else:
            pass
        i, d, n = get_breakpoints(data[name], axis=axis, dates=True, nlevs=True)

    if len(i) > 0:
        message("[%8s] [%27s]    [#]" %('idx','date'), **kwargs)
        message("\n".join(["[%8s] %s L: %3d" % (j, k, l) for j, k, l in zip(i, d, n)]), **kwargs)
        return True

    message("No breakpoints", **kwargs)
    return False


def adjustments(data, name, breakname, axis=0, dep_var=None, mean_cor=True, quantile_cor=True, quantile_adj=None, quantilen=None,
                **kwargs):
    """Detect and Correct Radiosonde biases from Departure Statistics
    Use ERA-Interim departures to detect breakpoints and
    adjustments these with a mean and a quantile adjustment going back in time.

    uses raso.timeseries.bp.analyse / correction

    Args:
        data         (DataArray) :  input data
        bdata        (DataArray) :  input data breakpoints
        dep_var      (str)       :  calculate departure from that variable
        mean_cor     (bool)      :  calc. mean adjustments
        quantile_cor (bool)      :  cacl. quantile adjustments
        quantile_adj (bool)      :  cacl. quantile adjustments from that variable
        quantilen    (list)      :  percentiles

    Keyword Args:
        sample_size (int)   :  minimum Sample size [130]
        borders     (int)   :  biased sample before and after a break [None]
        bounded     (tuple) :  limit correction to bounds
        recent      (bool)  :  Use all recent Data to adjustments
        ratio       (bool)  :  Use ratio instead of differences
        ref_period  (slice) :  Reference period for quantile_adj
        adjust_reference (bool) : adjust reference for quantile_adj

    Returns:
        bool, DataArray, ... :
    """
    # todo post adjustment departures and similarity of past and new breakpoints
    # todo success report
    from . import adj
    if not isinstance(data, xr.Dataset):
        raise ValueError("Requires a Dataset object")
    if not isinstance(name, str):
        raise ValueError("Requires a string name")
    if name not in data.data_vars:
        raise ValueError("data var not present")
    if breakname not in data.data_vars:
        raise ValueError("requires a breaks data var")

    idata = data[name].copy()
    if dep_var is not None:
        if dep_var not in data.data_vars:
            raise ValueError("dep var not present")

        idata = (idata - dep_var)

    if quantilen is None:
        quantilen = np.arange(0, 101, 10)

    if quantile_adj is not None:
        idata, quantile_adj = xr.align(idata, quantile_adj, join='left')  # Make sure it is the same shape
        if dep_var is not None:
            quantile_adj = quantile_adj - dep_var  # Make sure it's the same space

    values = idata.values
    ibreaks = get_breakpoints(data[breakname], axis=axis)  # just indices

    params = {'sample_size': kwargs.get('sample_size', 730),
              'borders': kwargs.get('borders', 180),
              'bounded': str(kwargs.get('bounded', '')),
              'recent': kwargs.get('recent', False),
              'ratio': kwargs.get('ratio', True)}

    if mean_cor:
        data[idata.name + '_m'] = (idata.dims, adj.mean(values, ibreaks, **kwargs))
        data[idata.name + '_m'].attrs.update(params)
        data[idata.name + '_m'].attrs['biascor'] = 'mean'
        data[idata.name + '_m'].attrs['standard_name'] += '_mean_adj'

    if quantile_cor:
        data[idata.name + '_q'] = (idata.dims, adj.quantile(values, ibreaks, axis=axis, quantilen=quantilen, **kwargs))
        data[idata.name + '_q'].attrs.update(params)
        data[idata.name + '_q'].attrs['biascor'] = 'quantil'
        data[idata.name + '_q'].attrs['standard_name'] += '_quantil_adj'

    if quantile_adj:
        qe_adj, qa_adj = adj.quantile_reference(values, quantile_adj.values, ibreaks, axis=axis, quantilen=quantilen,
                                                **kwargs)
        data[idata.name + '_qe'] = (idata.dims, qe_adj)
        data[idata.name + '_qe'].attrs.update(params)
        data[idata.name + '_qe'].attrs['biascor'] = 'quantil_era'
        data[idata.name + '_qe'].attrs['standard_name'] += '_quantil_era_adj'

        data[idata.name + '_adj'] = (idata.dims, qe_adj)
        data[idata.name + '_adj'].attrs.update(params)
        data[idata.name + '_adj'].attrs['biascor'] = 'quantil_era'
        data[idata.name + '_adj'].attrs['standard_name'] += '_quantil_era_adj'
        data[idata.name + '_adj'].attrs['standard_name'] += '_quantil_era_adj'

    return data


#
# def correct_loop(data, dep_var=None, use_dep=False, mean_cor=False, quantile_cor=False, quantile_adj=None,
#                  quantilen=None, clim_ano=True, **kwargs):
#     funcid = "[DC] Loop "
#     if not isinstance(data, DataArray):
#         raise ValueError(funcid + "Requires a DataArray class object")
#
#     if not mean_cor and not quantile_cor and quantile_adj is None:
#         raise RuntimeError(funcid + "Requires a correction: mean_cor, quantile_cor or quantile_adj")
#
#     if np.array([mean_cor, quantile_cor, quantile_adj is not None]).sum() > 1:
#         raise RuntimeError(funcid + "Only one Method at a time is allowed!")
#
#     xdata = data.copy()
#
#     # Make Large Arrays with all iterations ?
#     data = data.copy()
#     dims = data.get_dimension_values()
#     dims['iter'] = [0]
#     order = data.dims.list[:] + ['iter']
#     data.update_values_dims_remove(np.expand_dims(data.values, axis=-1), order, dims)
#     # data.dims['iter'].set_attrs({''})  # ?
#     sdata = data.copy()
#     sdata.values[:] = 0.
#     sdata.name += '_snht'
#     bdata = data.copy()
#     bdata.values[:] = 0.
#     bdata.name += '_breaks'
#     status = True
#     i = 1
#     while status:
#         status, stest, breaks, xdata = adjustments(xdata, dep_var=dep_var, use_dep=use_dep, mean_cor=mean_cor,
#                                                    quantile_cor=quantile_cor, quantile_adj=quantile_adj,
#                                                    quantilen=quantilen, clim_ano=clim_ano,
#                                                    **kwargs)
#         # combine
#         data.values = np.concatenate((data.values, np.expand_dims(xdata.values, axis=-1)), axis=-1)
#         # data.update_values_dims()
#         bdata.values = np.concatenate((bdata.values, np.expand_dims(breaks.values, axis=-1)), axis=-1)
#         sdata.values = np.concatenate((sdata.values, np.expand_dims(stest.values, axis=-1)), axis=-1)
#
#         # Does the adjustments still change anything ?
#         test = np.abs(np.nansum(data.values[:, :, i - 1] - xdata.values))  # sum of differences
#         if test < 0.1:
#             break
#         message(funcid + "%02d Breaks: \n" % i, **kwargs)
#         i += 1
#     # SAVE
#     data.update_values_dims(data.values, {'iter': range(i + 1)})
#     sdata.update_values_dims(sdata.values, {'iter': range(i + 1)})
#     bdata.update_values_dims(bdata.values, {'iter': range(i + 1)})
#     sdata.attrs['iterations'] = i
#
#     params = {'sample_size': kwargs.get('sample_size', 730),
#               'borders': kwargs.get('borders', 180),
#               'bounded': str(kwargs.get('bounded', '')),
#               'recent': kwargs.get('recent', False),
#               'ratio': kwargs.get('ratio', True)}
#
#     message(funcid + "Breaks: \n", **kwargs)
#     # print_breaks(bdata.subset(dims={'iter': i - 1}), verbose)
#
#     if mean_cor:
#         data.name += '_m_iter'
#         data.attrs['biascor'] = 'mean'
#         data.attrs['standard_name'] += '_mean_adj'
#         data.attrs.set_items(params)
#
#     elif quantile_cor:
#         data.name += '_q_iter'
#         data.attrs['biascor'] = 'quantile'
#         data.attrs['standard_name'] += '_quantile_adj'
#         data.attrs.set_items(params)
#
#     elif quantile_adj is not None:
#         data.name += '_qe_iter'
#         data.attrs['biascor'] = 'quantile_era_adjusted'
#         data.attrs['standard_name'] += '_quantile_era_adj'
#         data.attrs.set_items(params)
#     else:
#         pass
#
#     return status, sdata, bdata, data


def correct_2var(xdata, ydata):
    # Make a 3D (time, var1, var2) per level Test Statistics
    # Use that to adjustments both variables at the same time
    # ? water vapor transform -> how to unsplit vp to t,rh ?
    # t, rh -> td (esatfunc) -> vp
    # large errors -> temperature problem ?
    # smaller errors -> humidity problem ?
    # t, rh percentage of contribution to vp
    # vp (esat_inv) -> td
    pass


def breakpoint_statistics(data, name, breakname, axis=0, functions=None, borders=30, max_sample=1460, func_kwargs={}, **kwargs):
    from ..fun import vrange
    from .adj import break_iterator

    if not isinstance(data, xr.Dataset):
        raise ValueError("Requires a Dataset class object")

    if not isinstance(name, str):
        raise ValueError("Requires a str")

    if name not in data.data_vars:
        raise ValueError("var name not present")

    if breakname not in data.data_vars:
        raise ValueError("var name breaks not present")

    if max_sample is None or max_sample < 0:
        max_sample = 1e9  # too large, will never be used #todo max_sample = None

    if functions is None:
        functions = [np.nanmean, np.nanstd]

    ibreaks = get_breakpoints(data[breakname], axis=axis)

    nb = len(ibreaks)
    if nb == 0:
        message("Warning no Breakpoints found", level=0, **kwargs)
        return

    date_dim = data[name].dims[axis]
    shape = list(data[name].values.shape)

    dep = {getattr(ifunc, '__name__'): [] for ifunc in functions}

    dep['counts'] = []
    dates = data.coords[date_dim].values
    jbreaks = sorted(ibreaks, reverse=True)
    jbreaks.append(0)
    idims = list(data[name].dims)
    jdims = idims.copy()
    jdims.pop(axis)
    func_kwargs.update({'axis': axis})
    # iterate from now to past breakpoints
    for i, ib in enumerate(break_iterator(ibreaks, axis, shape, borders=borders, max_sample=max_sample)):
        period = vrange(dates[ib[axis]])
        idate = dates[jbreaks[i]]
        tmp = np.sum(np.isfinite(data[name][ib]), axis=axis)  # is an DataArray
        tmp.coords[date_dim] = idate
        tmp.coords['start'] = period[0]
        tmp.coords['stop'] = period[1]
        dep['counts'].append(tmp.copy())  # counts

        for j, ifunc in enumerate(functions):
            iname = getattr(ifunc, '__name__')
            # Requires clear mapping of input and output dimensions
            tmp = xr.apply_ufunc(ifunc, data[name][ib],
                                 input_core_dims=[idims],
                                 output_core_dims=[jdims],
                                 kwargs=func_kwargs)
            # tmp = ifunc(data[name][ib], axis=axis, **func_kwargs)  # only for functions with ufunc capability
            tmp.coords[date_dim] = idate
            tmp.coords['start'] = period[0]
            tmp.coords['stop'] = period[1]
            dep[iname].append(tmp.copy())

    for ifunc, ilist in dep.items():
        dep[ifunc] = xr.concat(ilist, dim=date_dim)

    dep = xr.Dataset(dep)
    return dep


#
# def reference_period(data, dep_var=None, quantilen=None, clim_subset=None, **kwargs):
#     funcid = "[DC] Ref.P "
#     if not isinstance(data, DataArray):
#         raise ValueError(funcid + "Requires a DataArray class object")
#
#     data = data.copy()
#     date = data.get_date_dimension()
#     dates = data.dims[date].values
#     if dep_var is not None:
#         if not isinstance(dep_var, DataArray):
#             raise ValueError(funcid + "Requires a DataArray class object")
#
#         dep_var = data.align(dep_var)  # implicit copy
#         dep = data - dep_var  # Departures (do the units match?)
#     else:
#         dep_var, _ = anomaly(data, period=clim_subset)
#         dep = dep_var
#
#     if quantilen is None:
#         quantilen = np.arange(0, 101, 10)
#
#     status, sdata, bdata = detect(dep, **kwargs)
#     period = slice(None)
#
#     # if status:
#     #     # find best matching period
#     #     if quantilen is None:
#     #         np.nanmedian( sample1, sample2 )
#     #     else:
#     #         np.nanpercentile()
#     #         difference
#
#     return period


def get_breakpoints(data, sign=1, axis=0, dates=False, nlevs=False):
    """ Get breakpoint datetime indices from Breakpoint DataArray

    Parameters
    ----------
    data : xr.DataArray
    sign : int
    axis : int
    dates : bool
    nlevs : bool

    Returns
    -------

    """
    if not isinstance(data, xr.DataArray):
        raise ValueError()

    t = np.where(data.values >= sign)
    datedim = data.dims[axis]
    indices = list(map(int, np.unique(t[axis])))
    if dates and nlevs:
        return indices, data[datedim].values[indices], [(data.values[i] > 1).sum(axis=axis) for i in indices]
    if dates:
        return indices, data[datedim].values[indices]
    return indices
