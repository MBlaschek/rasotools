# -*- coding: utf-8 -*-
import numpy as np
from xarray import Dataset, DataArray, set_options

from ..fun import message, dict2str, kwu

__all__ = ['apply_threshold', 'snht', 'adjust_mean', 'adjust_percentiles', 'adjust_percentiles_ref',
           'breakpoint_statistics', 'get_breakpoints', 'breakpoint_data']


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


def get_breakpoints(data, value=2, dim='date', var=None, return_startstop=False, startstop_min=0, **kwargs):
    """ Check if there are any breakpoints

    Args:
        data (DataArray): input data
        value (int): breakpoint indicator value
        dim (str): datetime dim
        var (str): variable
        **kwargs:

    Returns:
        list : breakpoints
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

    axis = idata.dims.index(dim)
    tmp = np.where(idata.values == value)
    i = list(map(int, np.unique(tmp[axis])))
    dates = np.datetime_as_string(idata[dim].values, unit='D')
    e = []
    s = []
    if idata.ndim > 1:
        summe = idata.values.sum(axis=1 if axis == 0 else 0)
    else:
        summe = idata.values
    for k in i:
        l = np.where(summe[:k][::-1] <= startstop_min)[0][0]
        m = np.where(summe[k:] <= startstop_min)[0][0]
        e += [k - l]
        s += [k + m]

    if kwargs.get('verbose', 0) > 0:
        if len(i) > 0:
            message("Breakpoints for ", idata.name, **kwargs)
            message("[%8s] [%8s] [%8s] [%8s] [ #]" % ('idx', 'end', 'peak', 'start'), **kwargs)
            message("\n".join(
                ["[%8s] %s %s %s %4d" % (j, dates[l], dates[j], dates[k], k - l) for j, k, l in zip(i, s, e)]),
                **kwargs)
    if return_startstop:
        return i, e, s
    return i


def adjust_mean(data, name, breakname, dim='date', suffix='_m', **kwargs):
    """Detect and Correct Radiosonde biases from Departure Statistics
Use ERA-Interim departures to detect breakpoints and
adjustments these with a mean  adjustment going back in time.


    Args:
        data (Dataset): Input Dataset with different variables
        name (str): Name of variable to adjust
        breakname (str): Name of variable with breakpoint information
        dim (str): datetime dimension
        suffix (str): add to name of new variables

    Optional Args:
        sample_size (int):  minimum Sample size [130]
        borders (int):  biased sample before and after a break [None]
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
    values = idata.values
    breaks = get_breakpoints(data[breakname], dim=dim, **kwu('level', 1, **kwargs))
    axis = idata.dims.index(dim)
    params = idata.attrs.copy()  # deprecated (xr-patch)

    message(name, str(values.shape), 'A:', axis, **kwargs)

    params.update({'sample_size': kwargs.get('sample_size', 730),
                   'borders': kwargs.get('borders', 180),
                   'recent': int(kwargs.get('recent', False)),
                   'ratio': int(kwargs.get('ratio', True))})

    message(dict2str(params), **kwu('level', 1, **kwargs))
    stdn = data[name].attrs.get('standard_name', name)

    data[name + suffix] = (idata.dims, adj.mean(values, breaks, axis=axis, **kwargs))
    data[name + suffix].attrs.update(params)
    data[name + suffix].attrs['biascor'] = 'mean'
    if 'niter' in data[name + suffix].attrs:
        data[name + suffix].attrs['niter'] += 1
    else:
        data[name + suffix].attrs['niter'] = 1
        data[name + suffix].attrs['standard_name'] = stdn + '_mean_adj'

    return data


def adjust_percentiles(data, name, breakname, dim='date', dep_var=None, suffix='_q', percentilen=None, **kwargs):
    """Detect and Correct Radiosonde biases from Departure Statistics
Use ERA-Interim departures to detect breakpoints and
adjustments these with a mean and a percentile adjustment going back in time.


    Args:
        data (Dataset): Input Dataset with different variables
        name (str): Name of variable to adjust
        breakname (str): Name of variable with breakpoint information
        dim (str): datetime dimension
        dep_var (str): Name of variable to use as a departure
        suffix (str): add to name of new variables
        percentilen (list): percentiles for percentile_cor

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

    if percentilen is None:
        percentilen = np.arange(0, 101, 10)

    values = idata.values
    breaks = get_breakpoints(data[breakname], dim=dim, **kwu('level', 1, **kwargs))
    axis = idata.dims.index(dim)
    params = idata.attrs.copy()  # deprecated (xr-patch)

    message(name, str(values.shape), 'A:', axis, 'Q:', np.size(percentilen), "Dep:", str(dep_var), **kwargs)

    params.update({'sample_size': kwargs.get('sample_size', 730),
                   'borders': kwargs.get('borders', 180),
                   'recent': int(kwargs.get('recent', False)),
                   'ratio': int(kwargs.get('ratio', True))})

    message(dict2str(params), **kwu('level', 1, **kwargs))
    stdn = data[name].attrs.get('standard_name', name)

    data[name + suffix] = (
        idata.dims, adj.percentile(values, breaks, axis=axis, percentilen=percentilen, **kwargs))
    data[name + suffix].attrs.update(params)
    data[name + suffix].attrs['biascor'] = 'percentil'
    data[name + suffix].attrs['standard_name'] = stdn + '_percentil_adj'

    return data


def adjust_percentiles_ref(data, name, adjname, breakname, dim='date', suffix='_qa', percentilen=None,
                           adjust_reference=True, **kwargs):
    """Detect and Correct Radiosonde biases from Departure Statistics
Use ERA-Interim departures to detect breakpoints and
adjustments these with a mean and a percentile adjustment going back in time.


    Args:
        data (Dataset): Input Dataset with different variables
        name (str): Name of variable to adjust
        adjname (str): Name of adjust variable
        breakname (str): Name of variable with breakpoint information
        dim (str): datetime dimension
        suffix (str): add to name of new variables
        percentilen (list): percentilen
        

    Optional Args:
        sample_size (int):  minimum Sample size [130]
        borders (int):  biased sample before and after a break [None]
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

    if adjname not in data.data_vars:
        raise ValueError("data var not present")

    if breakname not in data.data_vars:
        raise ValueError("requires a breaks data var")

    if suffix is not None:
        if suffix[0] != '_':
            suffix = '_' + suffix
            Warning('suffix needs an _. Added:', suffix)
    else:
        suffix = ''

    if percentilen is None:
        percentilen = np.arange(0, 101, 10)

    values = data[name].values.copy()
    avalues = data[adjname].values.copy()
    breaks = get_breakpoints(data[breakname], dim=dim, **kwu('level', 1, **kwargs))
    axis = data[name].dims.index(dim)
    params = data[name].attrs.copy()  # deprecated (xr-patch)

    message(name, str(values.shape), 'A:', axis, 'Q:', np.size(percentilen), "Adj:", adjname, **kwargs)

    params.update({'sample_size': kwargs.get('sample_size', 730),
                   'borders': kwargs.get('borders', 180),
                   'recent': int(kwargs.get('recent', False)),
                   'ratio': int(kwargs.get('ratio', True))})

    message(dict2str(params), **kwu('level', 1, **kwargs))
    stdn = data[name].attrs.get('standard_name', name)

    values, adjusted = adj.percentile_reference(values, avalues, breaks, axis=axis, percentilen=percentilen,
                                                **kwargs)
    data[name + suffix] = (data[name].dims, values)
    data[name + suffix].attrs.update(params)
    data[name + suffix].attrs['biascor'] = 'percentil_ref'
    data[name + suffix].attrs['standard_name'] = stdn + '_percentil_ref_adj'
    data[name + suffix].attrs['reference'] = adjname

    data[name + suffix + '_ref'] = (data[name].dims, adjusted)
    data[name + suffix + '_ref'].attrs.update(params)
    data[name + suffix + '_ref'].attrs['standard_name'] = stdn + '_percentil_ref'

    return data


def apply_bounds(data, name, other, lower, upper):
    "Apply bounds and replace"
    logic = data[name].values < lower
    n = np.sum(logic)
    data[name].values = np.where(logic, data[other].values, data[name].values)
    logic = data[name].values > upper
    n += np.sum(logic)
    data[name].values = np.where(logic, data[other].values, data[name].values)
    print("Outside bounds [", lower, "|", upper, "] :", n)


#
# def correct_loop(data, dep_var=None, use_dep=False, mean_cor=False, percentile_cor=False, percentile_adj=None,
#                  percentilen=None, clim_ano=True, **kwargs):
#     funcid = "[DC] Loop "
#     if not isinstance(data, DataArray):
#         raise ValueError(funcid + "Requires a DataArray class object")
#
#     if not mean_cor and not percentile_cor and percentile_adj is None:
#         raise RuntimeError(funcid + "Requires a correction: mean_cor, percentile_cor or percentile_adj")
#
#     if np.array([mean_cor, percentile_cor, percentile_adj is not None]).sum() > 1:
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
#                                                    percentile_cor=percentile_cor, percentile_adj=percentile_adj,
#                                                    percentilen=percentilen, clim_ano=clim_ano,
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
#     elif percentile_cor:
#         data.name += '_q_iter'
#         data.attrs['biascor'] = 'percentile'
#         data.attrs['standard_name'] += '_percentile_adj'
#         data.attrs.set_items(params)
#
#     elif percentile_adj is not None:
#         data.name += '_qe_iter'
#         data.attrs['biascor'] = 'percentile_era_adjusted'
#         data.attrs['standard_name'] += '_percentile_era_adj'
#         data.attrs.set_items(params)
#     else:
#         pass
#
#     return status, sdata, bdata, data


def adjust_table(data, name, analysis, dim='date', **kwargs):
    """
    test
Out[23]:
{'dpd':       data
 mean     2
 rmse     3
 var      2, 'era':       M  Q
 mean -3 -3
 rmse  2  2
 var   4  4}

    pd.concat(test, axis=1)
Out[22]:
      dpd era
     data   M  Q
mean    2  -3 -3
rmse    3   2  2
var     2   4  4

    Args:
        data:
        name:
        analysis:
        dim:
        **kwargs:

    Returns:

    """
    import pandas as pd
    # for all reanalysis
    out = {}
    for i, iana in enumerate(analysis):
        tmp = data[[name, iana]].copy()
        # snht
        tmp = snht(tmp, dim=dim, var=name, dep=iana, **kwargs)
        # threshold
        tmp = apply_threshold(tmp, var=name + '_snht', dim=dim)
        out[iana] = {}
        out[iana]['n'] = len(get_breakpoints(tmp, dim=dim, var=name + '_snht_break'))
        out[name] = {'data': {'RMSE': rmse(tmp[name], tmp[iana]),
                              'MEAN': np.nanmean(tmp[name] - tmp[iana]),
                              'VAR': np.nanvar(tmp[name] - tmp[iana])}}
        # adjust
        tmp = adjust_mean(tmp, name, name + '_snht_break', dim=dim, **kwargs)
        out[iana]['mdiff'] = {'RMSE': rmse(tmp[name + '_m'], tmp[iana]),
                              'MEAN': np.nanmean(tmp[name + '_m'] - tmp[iana]),
                              'VAR': np.nanvar(tmp[name + '_m'] - tmp[iana])}

    for ikey, idata in out.items():
        out[ikey] = pd.Dataframe(idata)
    return pd.concat(out, axis=1)


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


def breakpoint_data(data, a, b, c, dim='date', borders=0, sample_size=130, max_sample=1460, recent=False,
                    return_indices=False, **kwargs):
    """ Get Data before and after a breakpoint (index)

    Args:
        data (DataArray): Input data
        a (int): index earlier than breakpoint
        b (int): index of breakpoint
        c (int): index later than breakpoint
        dim (str): datetime dimension
        borders (int): borders around breakpoint to ignore
        sample_size (int): minimum sample size for stats
        max_sample (int): maxmimum sample size for stats
        recent (bool): don't use c
        return_indices (bool): return indices instead of data
        **kwargs:

    Returns:
        DataArray, DataArray DataArray: Before (min, max), Before (all), After Data
        or
        tuple, tuple : Before, Before, After Indices
    """
    from .adj import idx2shp
    if not isinstance(data, DataArray):
        raise ValueError()

    axis = data.dims.index(dim)
    dshape = data.values.shape
    ibiased = slice(a, b)
    isample = slice(b, c)
    if (b - a) - borders > sample_size:
        # [ - ; -borders]
        isample = slice(a, b - borders)
        ibiased = slice(a, b - borders)
        if (b - a) - 2 * borders > sample_size:
            # [ +borders ; -borders ]
            isample = slice(a + borders, b - borders)
            if (b - a) - 2 * borders > max_sample:
                # [ -max_sample ; -borders]
                isample = slice(b - borders - max_sample, b - borders)
                message("A [%d - %d - %d = %d - %d - %d = %d]" % (
                    b, borders, max_sample, isample.start, b, borders, isample.stop), **kwu('level', 1, **kwargs))

    if (c - b) - borders > sample_size:
        iref = slice(b + borders, c)
        if (c - b) - borders > max_sample and not recent:
            iref = slice(b + borders, b + borders + max_sample)
    else:
        iref = slice(b, c)

    ibiased = idx2shp(ibiased, axis, dshape)
    isample = idx2shp(isample, axis, dshape)
    iref = idx2shp(iref, axis, dshape)
    if return_indices:
        return isample, ibiased, iref
    return data[isample], data[ibiased], data[iref]


def breakpoint_statistics(data, breakname, dim='date', agg='mean', borders=None, inbetween=True, max_sample=None,
                          **kwargs):
    """

    Args:
        data:
        breakname:
        dim:
        agg:
        borders:
        inbetween:
        max_sample:
        **kwargs:

    Returns:

    """
    if not isinstance(data, Dataset):
        raise ValueError("Requires a Dataset class object")

    if dim not in data.coords:
        raise ValueError("Requires a datetime dimension", data.coords)

    if breakname not in data.data_vars:
        raise ValueError("var name breaks not present")

    if agg not in dir(Dataset):
        raise ValueError("agg not found", agg)

    data = data.copy()
    ibreaks = get_breakpoints(data[breakname], 2, dim=dim)

    nb = len(ibreaks)
    if nb == 0:
        message("Warning no Breakpoints found", **kwu('level', 0, **kwargs))  # Always print
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
    # data = eval("data.groupby('region').%s('%s')" % (agg, dim))
    data = data.groupby('region').apply(nanfunc, args=(), )  # func, args, kwargs
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

    #
    # find best matching period (lowest differences)
    # run SNHT
    # Split into pieces
    # run stats on each piece (RMSE)
    # choose
    # return piece + index
    return None
