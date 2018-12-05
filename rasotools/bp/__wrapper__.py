# -*- coding: utf-8 -*-
import numpy as np
from xarray import Dataset, DataArray, set_options
from ..fun import message, dict2str

__all__ = ['apply_threshold', 'snht', 'adjustments', 'any_breakpoints', 'breakpoint_statistics', 'get_breakpoints']


def snht(data, dim='date', var=None, dep=None, suffix=None, window=1460, missing=600, **kwargs):
    """ Calculate a Standard Normal Homogeinity Test

    Args:
        data (DataArray, Dataset):
        dim (str): datetime dimension
        var (str): variable if Dataset
        dep (str, DataArray): departure variable
        suffix (str): add to name of new variables
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
        idata = data.copy()
        var = idata.name if idata.name is not None else 'var'
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

    if suffix is not None:
        if suffix[0] != '_':
            suffix = '_' + suffix
            raise Warning('suffix needs an _. Added:', suffix)
    else:
        suffix = ''

    axis = idata.dims.index(dim)
    attrs = idata.attrs.copy()
    dep_add = False

    if dep is not None:
        if isinstance(dep, str) and isinstance(data, Dataset):
            dep = data[dep]

        elif isinstance(dep, DataArray):
            dep = dep
            dep_add = True

        else:
            raise ValueError("dep var not present")

        #
        with set_options(keep_attrs=True):
            idata = (idata - dep.reindex_like(idata))

        attrs['cell_method'] = 'departure ' + dep.name + attrs.get('cell_method', '')
        idata.attrs.update(attrs)  # deprecated (xr-patch)

    stest = np.apply_along_axis(test, axis, idata.values, window, missing)
    attrs.update({'units': '1', 'window': window, 'missing': missing})

    if isinstance(data, DataArray):
        data = data.to_dataset(name=var)

    if dep is not None:
        data[var + '_dep' + suffix] = idata
        if dep_add:
            data[dep.name if dep.name is not None else 'dep'] = dep

    data[var + '_snht' + suffix] = (list(idata.dims), stest)
    data[var + '_snht' + suffix].attrs.update(attrs)
    return data


def combine_breakpoints(data, dim='date', var=None, **kwargs):
    #
    # use snht break points
    # use sonde type changes
    # use documentation changes
    # use radiosonde intercomparison data to adjust radiosonde types?
    # how to weight these changes
    # probability of a breakpoint ?
    pass


def apply_threshold(data, dim='date', var=None, name='breaks', suffix=None, thres=50, dist=730, min_levels=3,
                    ensemble=False, **kwargs):
    """ Apply threshold on SNHT to detect breakpoints

    Args:
        data (DataArray, Dataset):
        dim (str): datetime dimension
        var (str): variable if Dataset
        name (str): name of new variable with above threshold (breaks)
        suffix (str): add to name of new variables
        thres (int, float): threshold value
        dist (int): distance between breaks
        min_levels (int): minimum significant levels for breaks
        ensemble (bool): run ensemble on thresholds, nthres=50,
    Returns:
        Dataset
    """
    from xarray import DataArray, Dataset
    from .det import detector, detector_ensemble

    if not isinstance(data, (DataArray, Dataset)):
        raise ValueError("Require a DataArray / Dataset object")

    if not isinstance(data, (DataArray, Dataset)):
        raise ValueError('Requires an xarray DataArray or Dataset', type(data))

    if suffix is not None:
        if suffix[0] != '_':
            suffix = '_' + suffix
            raise Warning('suffix needs an _. Added:', suffix)
    else:
        suffix = ''

    if isinstance(data, DataArray):
        idata = data.copy()  # link
        var = idata.name if idata.name is not None else 'var'
    else:
        if var is None or var not in list(data.data_vars):
            raise ValueError('Requires a variable name: var=', list(data.data_vars))

        idata = data[var]  # link

    if idata.ndim > 2:
        raise ValueError("Maximum of 2 dimensions: ", idata.shape)

    if dim not in idata.dims:
        raise ValueError('requires a datetime dimension', dim)

    axis = idata.dims.index(dim)
    params = {'units': '1', 'thres': thres, 'dist': dist, 'min_levels': min_levels}

    if ensemble:
        kwargs['nthres'] = kwargs.get('nthres', 50)
        breaks = detector_ensemble(idata.values, axis=axis, **kwargs)
        params['thres'] = 'ens50'
    else:
        breaks = detector(idata.values, axis=axis, dist=dist, thres=thres, min_levels=min_levels, **kwargs)

    name = var + '_' + name + suffix
    if isinstance(data, DataArray):
        data = idata.to_dataset(name=var)

    data[name] = (list(idata.dims), breaks)
    data[name].attrs.update(params)
    return data


def any_breakpoints(data, dim='date', var=None, **kwargs):
    """ Check if there are any breakpoints

    Args:
        data (DataArray): input data
        dim (str): datetime dim
        var (str): variable
        **kwargs:

    Returns:
        bool : breakpoints yes/no
    """
    if not isinstance(data, (DataArray, Dataset)):
        raise ValueError("Require a DataArray / Dataset object")

    if isinstance(data, Dataset):
        if var not in data.data_vars:
            raise ValueError("Dataset requires a var")
        idata = data[var]
    else:
        idata = data

    if dim not in idata.dims:
        raise ValueError("Requires a datetime dimension", idata.dims)

    i, d, n = get_breakpoints(idata, dim=dim, dates=True, nlevs=True)

    if len(i) > 0:
        message("[%8s] [%27s]    [#]" % ('idx', 'date'), **kwargs)
        message("\n".join(["[%8s] %s L: %3d" % (j, k, l) for j, k, l in zip(i, d, n)]), **kwargs)
        return True

    message("No breakpoints", **kwargs)
    return False


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


def adjustments(data, name, breakname, dim='date', dep_var=None, suffix=None, mean_cor=True, quantile_cor=True,
                quantile_adj=None, quantilen=None, **kwargs):
    """Detect and Correct Radiosonde biases from Departure Statistics
Use ERA-Interim departures to detect breakpoints and
adjustments these with a mean and a quantile adjustment going back in time.


    Args:
        data (Dataset): Input Dataset with different variables
        name (str): Name of variable to adjust
        breakname (str): Name of variable with breakpoint information
        dim (str): datetime dimension
        dep_var (str): Name of variable to use as a departure
        suffix (str): add to name of new variables
        mean_cor (bool): apply mean adjustments
        quantile_cor (bool): apply quantil adjustments
        quantile_adj (bool): apply quantil-ERA adjustments
        quantilen (list): quantiles for quantile_cor

    Optional Args:
        sample_size (int):  minimum Sample size [130]
        borders (int):  biased sample before and after a break [None]
        bounded (tuple):  limit correction to bounds
        recent (bool):  Use all recent Data to adjustments
        ratio (bool):  Use ratio instead of differences

    Returns:
        Dataset
    """
    from . import adj
    if not isinstance(data, Dataset):
        raise ValueError("Requires a Dataset object", type(data))

    if not isinstance(name, str):
        raise ValueError("Requires a string name", type(name))

    if name not in data.data_vars:
        raise ValueError("data var not present")

    if breakname not in data.data_vars:
        raise ValueError("requires a breaks data var")

    idata = data[name].copy()

    if dep_var is not None:
        if dep_var not in data.data_vars:
            raise ValueError("dep var not present", data.data_vars)

        with set_options(keep_attrs=True):
            idata = (idata - data[dep_var].reindex_like(idata))

    if suffix is not None:
        if suffix[0] != '_':
            suffix = '_' + suffix
            Warning('suffix needs an _. Added:', suffix)
    else:
        suffix = ''

    if quantilen is None:
        quantilen = np.arange(0, 101, 10)

    if quantile_adj is not None:
        if quantile_adj not in data.data_vars:
            raise ValueError("quantile_adj var not present", data.data_vars)

        adj = data[quantile_adj].reindex_like(idata)
        # if dep_var is not None:
        #     with set_options(keep_attrs=True):
        #         adj = adj - data[dep_var].reindex_like(idata)  # Make sure it's the same space

    values = idata.values
    axis = idata.dims.index(dim)
    params = idata.attrs.copy()  # deprecated (xr-patch)
    ibreaks = get_breakpoints(data[breakname], dim=dim)  # just indices
    message(name, str(values.shape), 'A:', axis, 'Q:', np.size(quantilen), "Dep:", str(dep_var), "Adj:", str(quantile_adj),
            '#B:', len(ibreaks), **kwargs)

    params.update({'sample_size': kwargs.get('sample_size', 730),
                   'borders': kwargs.get('borders', 180),
                   'bounded': str(kwargs.get('bounded', '')),
                   'recent': kwargs.get('recent', False),
                   'ratio': kwargs.get('ratio', True)})

    message(dict2str(params), level=1, **kwargs)
    stdn = data[name].attrs.get('standard_name', name)

    if mean_cor:
        data[name + '_m' + suffix] = (idata.dims, adj.mean(values, ibreaks, axis=axis, **kwargs))
        data[name + '_m' + suffix].attrs.update(params)
        data[name + '_m' + suffix].attrs['biascor'] = 'mean'
        data[name + '_m' + suffix].attrs['standard_name'] = stdn + '_mean_adj'

    if quantile_cor:
        data[name + '_q' + suffix] = (
        idata.dims, adj.quantile(values, ibreaks, axis=axis, quantilen=quantilen, **kwargs))
        data[name + '_q' + suffix].attrs.update(params)
        data[name + '_q' + suffix].attrs['biascor'] = 'quantil'
        data[name + '_q' + suffix].attrs['standard_name'] = stdn + '_quantil_adj'

    if quantile_adj is not None:
        qe_adj, qa_adj = adj.quantile_reference(values, adj.values, ibreaks, axis=axis, quantilen=quantilen,
                                                **kwargs)
        data[name + '_qe' + suffix] = (idata.dims, qe_adj)
        data[name + '_qe' + suffix].attrs.update(params)
        data[name + '_qe' + suffix].attrs['biascor'] = 'quantil_era'
        data[name + '_qe' + suffix].attrs['standard_name'] = stdn + '_quantil_era_adj'

        data[name + '_adj' + suffix] = (idata.dims, qe_adj)
        data[name + '_adj' + suffix].attrs.update(params)
        data[name + '_adj' + suffix].attrs['biascor'] = 'quantil_era'
        data[name + '_adj' + suffix].attrs['standard_name'] = stdn + '_quantil_era_adj'
        data[name + '_adj' + suffix].attrs['standard_name'] = stdn + '_quantil_era_adj'

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


def breakpoint_statistics(data, breakname, dim='date', agg='mean', borders=None, inbetween=True, max_sample=None,
                          **kwargs):
    if not isinstance(data, Dataset):
        raise ValueError("Requires a Dataset class object")

    if dim not in data.coords:
        raise ValueError("Requires a datetime dimension", data.coords)

    if breakname not in data.data_vars:
        raise ValueError("var name breaks not present")

    if agg not in dir(Dataset):
        raise ValueError("agg not found", agg)

    data = data.copy()
    ibreaks = get_breakpoints(data[breakname], dim=dim)

    nb = len(ibreaks)
    if nb == 0:
        message("Warning no Breakpoints found", level=0, **kwargs)
        return

    if borders is None:
        borders = 0

    # calculate regions
    region = np.zeros(data[dim].size)
    j = 0
    k = len(ibreaks) + 1
    i = 0
    for i in ibreaks:
        if max_sample is not None:
            region[slice(i - borders - max_sample, i - borders)] = k
        else:
            region[slice(j, i - borders)] = k

        if j > 0 and borders > 0 and inbetween:
            region[slice(j - 2 * borders, j)] = k + 0.5  # in between

        j = i + borders
        k -= 1

    if borders > 0 and inbetween:
        region[slice(i - borders, i + borders)] = k + 0.5

    region[slice(i + borders, None)] = k
    data['region'] = ('date', region)
    region = data['region'].copy()
    # Use regions to groupby and apply functions
    data = eval("data.groupby('region').%s('%s')" % (agg, dim))
    data = data.isel(region=data.region > 0)  # remove 0 region (to be excluded)
    return data, region.where(region > 0)

    # shape = list(data[name].values.shape)
    #
    # dep = {getattr(ifunc, '__name__'): [] for ifunc in functions}
    #
    # dep['counts'] = []
    # dates = data.coords[dim].values
    # jbreaks = sorted(ibreaks, reverse=True)
    # jbreaks.append(0)
    # idims = list(data[name].dims)
    # jdims = idims.copy()
    # jdims.pop(axis)
    # func_kwargs.update({'axis': axis})
    # #
    # # iterate from now to past breakpoints
    # #
    # for i, ib in enumerate(break_iterator(ibreaks, axis, shape, borders=borders, max_sample=max_sample)):
    #     period = vrange(dates[ib[axis]])
    #     idate = dates[jbreaks[i]]
    #     tmp = np.sum(np.isfinite(data[name][ib]), axis=axis)  # is an DataArray
    #     tmp.coords[dim] = idate
    #     tmp.coords['start'] = period[0]
    #     tmp.coords['stop'] = period[1]
    #     dep['counts'].append(tmp.copy())  # counts
    #
    #     for j, ifunc in enumerate(functions):
    #         iname = getattr(ifunc, '__name__')
    #         # Requires clear mapping of input and output dimensions
    #         tmp = apply_ufunc(ifunc, data[name][ib],
    #                           input_core_dims=[idims],
    #                           output_core_dims=[jdims],
    #                           kwargs=func_kwargs)
    #         # tmp = ifunc(data[name][ib], axis=axis, **func_kwargs)
    #         # only for functions with ufunc capability
    #         tmp.coords[dim] = idate
    #         tmp.coords['start'] = period[0]
    #         tmp.coords['stop'] = period[1]
    #         dep[iname].append(tmp.copy())
    #
    # for ifunc, ilist in dep.items():
    #     dep[ifunc] = concat(ilist, dim=dim)
    #
    # dep = Dataset(dep)
    # return dep


def reference_period(data, dim='date', dep_var=None, period=None, **kwargs):
    from ..met.time import anomaly

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object", type(data))

    if dim not in data.dims:
        raise ValueError("Requires a datetime dimension", data.dims)

    data = data.copy()
    # attrs ?

    if dep_var is not None:
        if not isinstance(dep_var, DataArray):
            raise ValueError("Requires a DataArray class object", type(dep_var))

        dep = data - dep_var  # Departures (do the units match?)
    else:
        dep, _ = anomaly(data, dim=dim, period=period)

    # find best matching period (lowest differences)
    #
    return None


def get_breakpoints(data, dim='date', sign=1, dates=False, nlevs=False):
    """ Get breakpoint datetime indices from Breakpoint DataArray

    Args:
        data (DataArray): input data
        dim (str): datetime dimension
        sign (int): threshold value
        dates (bool): return dates
        nlevs (bool): return levels

    Returns:
        list

    """
    from xarray import DataArray
    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    t = np.where(data.values >= sign)
    axis = data.dims.index(dim)
    indices = list(map(int, np.unique(t[axis])))

    if dates and nlevs:
        return indices, data[dim].values[indices], [(data.values[i] > 1).sum(axis=axis) for i in indices]

    if dates:
        return indices, data[dim].values[indices]

    return indices
