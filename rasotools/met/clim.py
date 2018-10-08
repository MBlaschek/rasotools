# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..fun import nanrange, nancount


__all__ = ['climatology', 'anomaly', 'trend', 'trend_per_month', 'trend_mon_percentile',
           'day_night_departures', 'correlate', 'covariance']

"""
Update : Nur mehr Numpy funktionen
__wrapper__ übernimmt das Xarray dings zeugs

"""

def climatology(data, period=None, min_periods=0, keep_shape=False):
    """

    Args:
        data (DataArray): Input Data
        period (slice, str): Time period to use for calculations
        min_periods:
        keep_shape:

    Returns:
        DataArray : Climate Means
    """
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")

    date_dim = data.get_date_dimension()
    if date_dim is None:
        print("; ".join(["%s: %s" % (i, j) for i, j in zip(data.order, data.axes)]))
        raise RuntimeError("Requires a datetime dimension with axis T")

    data = data.copy()
    dates = data.dims[date_dim].values  # Dates
    iaxis = data.order.index(date_dim)  # Could be used for axis

    if period is not None:
        period = date_selection(dates, period)  # period for climatology applied to whole series
        dates = dates[period]

    dates = pd.DatetimeIndex(dates)
    attrs = {'standard_name': data.attrs['standard_name'] + '_climatology',
             'period': '%d-%d' % nanrange(dates.year.values),
             'cell_method': 'mean of months over years'
             }

    month, grouped = groupby(dates.month.values, iaxis, data.values.ndim)
    data = groupby_apply(grouped, data, np.nanmean, group={date_dim: month}, attrs=attrs,
                         axis=iaxis, min_periods=min_periods, keep_shape=keep_shape)
    if not keep_shape:
        data.dims[date_dim].set_attrs({'units': 'month of year', 'axis': 'T'})

    data.name += '_clim'

    if min_periods > 0:
        data.attrs['min_periods'] = min_periods

    return data


def anomaly(data, period=None, min_periods=0, noattrs=False):
    """ Calculates the anomaly from the climatology per month of a time series

    Args:
        data (DataArray) : Inputdata
        period (slice, str) : Indices of Dates for calculation
        min_periods (int) : minimum required sample size

    Returns:
        DataArray : Anomalies
    """
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")

    data = data.copy()
    date_dim = data.get_date_dimension()
    dates = pd.DatetimeIndex(data.dims[date_dim].values)

    # Calculate Climatology
    clim = climatology(data, period=period, min_periods=min_periods, keep_shape=True)

    # Calculate Anomaly
    data = data - clim

    if period is None:
        period = '%d-%d' % nanrange(dates.year.values)
    else:
        period = '%d-%d' % nanrange(dates.to_series()[period].index.year.values)

    attrs = {'standard_name': data.attrs['standard_name'] + '_anomaly',
             'period': period}

    if not noattrs:
        data.name += '_ano'
        set_attributes(data, attrs)
    return data


def estimate_sample_size(data, ratio=0.6, freq='12h'):
    """ Estimate the sample size from a timeseries, given freq and ratio

    Args:
        data (DataArray): Inputdata
        ratio (float): percentage of data as ratio
        freq (str): Pandas freq str

    Returns:
        int : Sample size according to freq and ratio
    """
    date_dim = data.get_date_dimension()
    dates = pd.DatetimeIndex(data.dims[date_dim])
    axis = data.order.index(date_dim)
    dates = pd.date_range(dates.min().replace(hour=np.min(dates.hour.values)),
                          dates.max().replace(hour=np.max(dates.hour.values)), freq=freq)
    years = nanrange(dates.year)
    print("Estimate Sample size (%d%%, F:%s): %d [%d : %d] %d %d" % (
    int(ratio * 100), freq, int(dates.size * ratio), years[0], years[1], np.diff(years) + 1, dates.size))
    print(100 * data.apply(nancount, axis=axis).to_pandas() / float(dates.size))
    return int(dates.size * ratio)


def counts(data, per=None):
    """ Count the amount of data

    Args:
        data (DataArray): Inputdata
        per (str): Pandas freq str

    Returns:
        DataArray : counts per time unit
    """
    data = data.copy()
    data.values = np.isfinite(data.values)  # 0, 1
    date_dim = data.get_date_dimension()
    axis = data.order.index(date_dim)
    # counts = groupby_apply(grouped, data, nancount, group={date_dim: range(1, 13)}, axis=iaxis, attrs=attrs)
    data.name += '_counts'
    data.attrs['units'] = '1'
    data.attrs['standard_name'] += '_counts'
    if per is not None:
        return data.resample(freq=per, agg=np.sum)
    else:
        return data.apply(np.sum, axis=axis)


def trend(data, use_anomalies=True, min_periods=3, method='polyfit', return_intercept=False, alpha=None):
    """ Calculate Trend estimates

    Args:
        data (DataArray):
        use_anomalies (bool):
        min_periods (int):
        method (str): polyfit or theil_sen
        return_intercept (bool): intercept
        alpha (float): get confidence levels for that p value

    Returns:
        DataArray : trends
    """
    from .stats import trend as xtrend

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

    if method not in ['polyfit', 'theil_sen']:
        raise ValueError("Requires either polyfit or theil_sen as method")

    data = data.copy()
    date_dim = data.get_date_dimension()
    dates = data.dims[date_dim].values  # Dates
    axis = data.order.index(date_dim)

    # can only use it like this
    # infer freq?
    per = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')  # factor for trends

    if use_anomalies:
        data = anomaly(data, noattrs=True)

    # Convert to standard time axis
    idates = dates.astype('long')  # Nano Seconds
    idates -= idates[0]  # relative Times
    # Trends
    # k = [unit]/time
    params = xtrend(data.values, idates, method=method, alpha=alpha, nmin=min_periods, axis=axis)
    params = np.asarray(params)  # might be a list of results
    idx = [slice(None)] * params.ndim
    idx[axis] = 0
    params[idx] = params[idx] * per # trend per day (nano s to day)

    dims = data.get_dimension_values()
    order = list(data.order)
    order.remove(date_dim)
    dims.pop(date_dim)
    data.update_values_dims_remove(params[idx], order, dims)  # update values
    data.attrs['period'] = '%d-%d' % nanrange(pd.DatetimeIndex(dates).year.values)
    data.name += '_trend'
    data.attrs['units'] += '/day'
    data.attrs['standard_name'] += '_trend'
    if min_periods is not None:
        data.attrs['min_periods'] = min_periods

    data.attrs['cell_method'] = 'daily trend of anomalies' if use_anomalies else 'daily trend'

    # Output
    out = (data,)

    if return_intercept:
        interc = data.copy()
        idx[axis] = 1
        interc.values = params[idx]
        interc.name = interc.name.replace('_trend','_intercept')
        interc.attrs['units'] = interc.attrs['units'].replace('/day','')
        interc.attrs['standard_name'] = interc.attrs['standard_name'].replace('_trend','_intercept')
        out += (interc,)

    if alpha is not None:
        medlo = data.copy()
        idx[axis] = 2
        medlo.values = params[idx] * per
        medhi = data.copy()
        idx[axis] = 3
        medhi.values = params[idx] * per
        medlo.name += '_low'
        medhi.name += '_high'
        out += (medlo, medhi,)

    if len(out) == 1:
        return out[0]
    return out


def trend_mon_percentile(data, percentile=None, subset=None, monthly_ratio=0.3, return_counts=False, verbose=0):
    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

    if percentile is None:
        percentile = [25, 50, 75]  # Quartils

    data = data.copy()
    date_dim = data.get_date_dimension()
    dates = data.dims[date_dim].values  # Dates
    axis = data.order.index(date_dim)
    if subset is not None:
        subset = date_selection(dates, subset)
        index = [slice(None)] * len(data.order)
        index[axis] = subset
        data = data.subset(index=index)
        dates = data.dims[date_dim].values  # Dates

    # group = {'month': np.unique(pd.DatetimeIndex(dates).to_period('M').to_timestamp().values)}
    attrs = {date_dim: {'axis': 'T', 'units': 'month of year'},
             'period': '%d-%d' % nanrange(pd.DatetimeIndex(dates).year.values),
             'standard_name': data.attrs['standard_name'] + '_mon_perc'
             }
    # Group by freqency monthly
    # grouped = date_groupby(dates, axis, data.values.ndim)
    trends = []
    counts = []
    for iq in percentile:
        # sample_size ???
        tmp = data.resample(freq='M', agg=np.nanpercentile, q=iq)
        # tmp = groupby_apply(grouped, data, np.nanpercentile, q=iq, axis=axis, group=group, attrs=attrs)
        # Trend pro percentile
        itrend, icounts = trend(tmp, use_anomalies=False, return_counts=True)
        itrend.values = np.where(icounts.values > monthly_ratio, itrend.values, np.nan)
        counts.append(icounts)
        trends.append(itrend)

    trends = vstack(trends, 'percentile', percentile)
    set_attributes(trends, {'percentile': {'units': 'percent'},
                            'cell_method': 'daily trend of monthly quantiles',
                            'limit': monthly_ratio})
    if return_counts:
        counts = vstack(counts, 'percentile', percentile)
        counts.attrs['units'] = 1
        counts.attrs['standard_name'] = data.attrs['standard_name'] + '_mon_perc_count'
        return trends, counts
    return trends


def trend_per_month(data, subset=None, use_anomalies=True):
    """ Calculates the trend per month of a timeseries

    Args:
        data (DataArray) : Input timeseries
        subset (slice, str) : Time period to use
        use_anomalies (bool) : Use anomalies rather than absolute values

    Returns:
        DataArray : Trends
    """
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")

    date_dim = data.get_date_dimension()
    if date_dim is None:
        print("; ".join(["%s: %s" % (i, j) for i, j in zip(data.order, data.axes)]))
        raise RuntimeError("Requires a datetime dimension with axis T")

    data = data.copy()
    dates = data.dims[date_dim].values  # Dates
    iaxis = data.order.index(date_dim)  # Could be used for axis

    if subset is not None:
        subset = date_selection(dates, subset)  # period for climatology applied to whole series
        dates = dates[subset]

    dates = pd.DatetimeIndex(dates)
    attrs = {'standard_name': data.attrs['standard_name'] + '_anomaly',
             'period': '%d-%d' % nanrange(dates.year.values),
             }

    _, grouped = groupby(dates.month.values, iaxis, data.values.ndim)
    out = []
    for i in grouped:
        tmp = data[i]
        itrend = trend(tmp, use_anomalies=use_anomalies, return_counts=False)
        out.append(itrend)

    # ? dimensions mismatch ?
    data.values = np.concatenate(out, axis=iaxis)  # np.vstack(out)  # merge to array
    data.name += '_ano'
    set_attributes(data, attrs)
    return data

    # if not isinstance(data, DataArray):
    #     raise ValueError("Requires a DataArray class (data)")
    #
    # data = data.copy()
    # date_dim = data.get_date_dimension()
    # dates = data.dims[date_dim].values  # Dates
    # if subset is not None:
    #     subset = date_selection(dates, subset)
    #     month = pd.DatetimeIndex(dates).month[subset]
    # else:
    #     month = pd.DatetimeIndex(dates).month
    #
    # per = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')  # factor for trends
    #
    # if use_anomalies:
    #     data, _ = anomaly(data)
    #     data.name = data.name.replace('_ano', '')
    #     data.attrs['standard_name'] = data.attrs['standard_name'].replace('_anomaly', '')  # remove
    #
    # month = pd.DatetimeIndex(dates).month
    # means = np.full((12,) + data.values.shape[1:], np.nan)  # Assumes Axis 0
    # counts = np.zeros((12,) + data.values.shape[1:])
    # if np.issubdtype(data.values.dtype, int):
    #     data.values = data.values.astype(float)
    #
    # index = [slice(None)] * len(data.order)
    # ixd = data.order.index(date_dim)
    # mindex = [slice(None)] * len(means.shape)
    # idates = dates.astype(long)  # Nano Seconds
    # idates -= idates[0]  # relative
    # for imon in range(12):
    #     index[ixd] = np.where((month == (imon + 1)))[0]
    #     mindex[0] = imon
    #     means[mindex] = np.apply_along_axis(_trend_helper, 0, data.values[index],
    #                                         idates[index[ixd]]) * per  # Daily Trends assumes axis 0
    #     counts[mindex] = np.sum(np.isfinite(data.values[index]), axis=0)  # Daily Counts assumes axis 0
    #
    # order = list(data.order)
    # order[order.index(date_dim)] = 'month'
    # dims = data.get_dimension_values()
    # dims.pop(date_dim)
    # dims['month'] = range(1, 13)
    # data.update_values_dims_remove(means, order, dims)
    # data.dims['month'].units = 'month per year'
    # data.dims['month'].axis = 'T'
    # data.attrs['cell_method'] = 'daily trend of anomalies per month' if use_anomalies else 'daily trend per month'
    # data.attrs['standard_name'] += '_monthly_trend'
    # data.name += '_mtrend'
    #
    # counts = data.copy()
    # counts.attrs['standard_name'] += '_counts'
    # counts.name += '_counts'
    # counts.attrs['units'] = '1'
    # counts.attrs['cell_method'] = 'total counts'
    # if return_counts:
    #     return data, counts  # Monthly Trends & Counts
    # return data


def correlate(xdata, ydata, subset=None, method='spearman', return_counts=True):
    from pandas.core.nanops import nancorr

    if not isinstance(xdata, DataArray):
        raise ValueError("Requires a DataArray class object")

    if not isinstance(ydata, DataArray):
        raise ValueError("Requires a DataArray class object")

    xdata = xdata.copy()
    ydata = xdata.align(ydata)
    date_dim = xdata.get_date_dimension()
    dates = xdata.dims[date_dim].values  # Dates
    if subset is not None:
        subset = date_selection(dates, subset)
        index = [slice(None)] * len(xdata.order)
        index[xdata.order.index(date_dim)] = subset
        xdata = xdata.subset(index=index)
        ydata = ydata.subset(index=index)
        dates = xdata.dims[date_dim].values  # Dates

    axis = xdata.order.index(date_dim)  # axis
    shapes = list(xdata.values.shape)  #
    shapes.pop(axis)  # remove date dim
    shapes = tuple(shapes)
    n = []
    k = []
    for i in np.ndindex(shapes):
        idx = list(i)
        idx.insert(axis, slice(None))
        k.append(nancorr(xdata.values[idx], ydata.values[idx], method=method))
        # n.append(np.sum(np.isfinite(xdata.values[idx]) & np.isfinite(ydata.values[idx])) / float(
        #     xdata.values[idx].size))  # relativ
        n.append(np.sum(np.isfinite(xdata.values[idx]) & np.isfinite(ydata.values[idx])))  # absolute

    k = np.array(k).reshape(shapes)
    n = np.array(n).reshape(shapes)
    corr = xdata.copy()
    order = list(corr.order)
    order.remove(date_dim)
    dims = corr.get_dimension_values()
    dims.pop(date_dim)
    corr.update_values_dims_remove(k, order, dims)  # update values
    corr.name += '_corr'
    corr.attrs['period'] = '%d-%d' % nanrange(pd.DatetimeIndex(dates).year.values)
    corr.attrs['standard_name'] += '_corr'
    corr.attrs['units'] = '1'
    corr.attrs['cell_method'] = '%s correlation with %s' % (method, ydata.name)

    counts = corr.copy()
    counts.values = n
    counts.name += '_counts'
    counts.attrs['units'] = '1'
    counts.attrs['standard_name'] += '_counts'
    if return_counts:
        return corr, counts
    return corr


def covariance(xdata, ydata, subset=None, return_counts=True):
    from pandas.core.nanops import nancov

    if not isinstance(xdata, DataArray):
        raise ValueError("Requires a DataArray class object")

    if not isinstance(ydata, DataArray):
        raise ValueError("Requires a DataArray class object")

    xdata = xdata.copy()
    ydata = xdata.align(ydata)
    date_dim = xdata.get_date_dimension()
    dates = xdata.dims[date_dim].values  # Dates
    if subset is not None:
        subset = date_selection(dates, subset)
        index = [slice(None)] * len(xdata.order)
        index[xdata.order.index(date_dim)] = subset
        xdata = xdata.subset(index=index)
        ydata = ydata.subset(index=index)
        dates = xdata.dims[date_dim].values  # Dates

    axis = xdata.order.index(date_dim)  # axis
    shapes = list(xdata.values.shape)  #
    shapes.pop(axis)  # remove date dim
    shapes = tuple(shapes)
    n = []
    k = []
    for i in np.ndindex(shapes):
        idx = list(i)
        idx.insert(axis, slice(None))
        k.append(nancov(xdata.values[idx], ydata.values[idx]))
        # n.append(np.sum(np.isfinite(xdata.values[idx]) & np.isfinite(ydata.values[idx])) / float(
        #     xdata.values[idx].size))
        n.append(np.sum(np.isfinite(xdata.values[idx]) & np.isfinite(ydata.values[idx])))  # absolute

    k = np.array(k).reshape(shapes)
    n = np.array(n).reshape(shapes)
    corr = xdata.copy()
    order = list(corr.order)
    order.remove(date_dim)
    dims = corr.get_dimension_values()
    dims.pop(date_dim)
    corr.update_values_dims_remove(k, order, dims)  # update values
    corr.name += '_cov'
    corr.attrs['period'] = '%d-%d' % nanrange(pd.DatetimeIndex(dates).year.values)
    corr.attrs['standard_name'] += '_cov'
    corr.attrs['units'] = '1'
    corr.attrs['cell_method'] = 'covariance with %s' % ydata.name

    counts = corr.copy()
    counts.values = n
    counts.name += '_counts'
    counts.attrs['units'] = '1'
    counts.attrs['standard_name'] += '_counts'
    if return_counts:
        return corr, counts
    return corr


def day_night_departures(data, standardize=True, start_datetime=None, end_datetime=None, verbose=0):
    """ Calculate Day-Night Departures

    Args:
        data (DataArray) : Input data
        standardize (bool) : Run Standardization
        start_datetime (str) : Start datetime
        end_datetime (str) : End datetime
        verbose (int) : verbosness

    Returns:
        DataArray : day-night departures on noon indexes
    """
    from xData.ops import standardize_datetime
    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object (data)")

    data = data.copy()
    date_dim = data.get_date_dimension()
    idate = data.order.index(date_dim)
    if standardize:
        start_datetime = data.dims[date_dim].values.min() if start_datetime is None else start_datetime
        end_datetime = data.dims[date_dim].values.max() if end_datetime is None else end_datetime
        data = standardize_datetime(data, start_datetime, end_datetime, freq='12h')

    # split
    dates = pd.DatetimeIndex(data.dims[date_dim].values.copy())
    t1 = pd.Series(True, index=dates[dates.hour == 0].to_period('D'), name='night')  # Night Series
    t2 = pd.Series(True, index=dates[dates.hour == 12].to_period('D'), name='noon')  # Day Series
    # figure out what days have both sounding times
    fulldays = pd.concat([t1, t2], axis=1)  # Merge -> creates NAN
    fulldays = fulldays.fillna(False)  # fill NAN with False
    fulldays = fulldays.index[fulldays.all(1)].to_timestamp()  # select only with both night & day -> datetime
    ix = dates.to_period('D').to_timestamp().isin(fulldays)  # which day has both 0 and 12 UTC soundings
    night = [slice(None, None)] * len(data.order)
    noon = [slice(None, None)] * len(data.order)
    with np.errstate(invalid='ignore'):
        night[idate] = np.where(ix & (dates.hour == 0))[0]
        noon[idate] = np.where(ix & (dates.hour == 12))[0]

    dep = data.values[noon] - data.values[night]
    dates = data.dims[date_dim].values[noon[idate]].copy()
    data.update_values_dims(dep, {date_dim: dates})  # center at noon
    data.attrs['standard_name'] += '_day_night_departure'
    data.name += '_dn_dep'
    return data


# def rolling_trend(data, window, freq='M', period=None, use_anomalies=True, min_periods=None):
#     if not isinstance(data, DataArray):
#         raise ValueError("Requires a numpy array (data)")
#
#     date_dim = data.get_date_dimension()
#     if date_dim is None:
#         print("; ".join(["%s: %s" % (i, j) for i, j in zip(data.order, data.axes)]))
#         raise RuntimeError("Requires a datetime dimension with axis T")
#
#     if data.dims[date_dim].attrs.get('freq') != freq:
#         data = data.resample(freq)
#
#     ano = anomaly(data, period=period, min_periods=min_periods)
#     return
    # iterate dims
    # to_pandas().rolling(window=window, center=True, min_periods=).apply(_ptrend)
    # back into DataArray -> done
    # for odim in odims:
    #    for


# def anomalies(data, dates, subset=None, return_means=False):
#     """ Calculates the climatology per month of a timeseries
#
#     Args:
#         data (np.ndarray) : Input timeseries
#         dates (np.ndarray) : Dates
#         subset (np.ndarray) : Indices of Dates for calculation
#         return_means (bool) : Output Monthly Means
#
#     Returns:
#         np.ndarray : Anomalies
#     """
#     if not isinstance(data, np.ndarray):
#         raise ValueError("Requires a numpy array (data)")
#
#     if not isinstance(dates, np.ndarray):
#         raise ValueError("Requires a numpy array (dates)")
#
#     if subset is None:
#         subset = slice(None, None)
#     else:
#         if not isinstance(subset, np.ndarray):
#             raise ValueError("Requires a numpy array (subset)")
#
#     func = getattr(np, 'nanmean')  # could make this flexible with median, var, std
#     data = data.copy()
#     dates = pd.DatetimeIndex(dates.copy())
#     month = dates.month
#     # nyears = int(np.diff(nanrange(dates.year.values)))
#     means = np.full((12,) + data.shape[1:], np.nan)  # Assumes Axis 0
#     counts = np.zeros((12,) + data.shape[1:])
#     # n = 60 * nyears  # about twice obs per day
#     for imon in range(12):
#         ix = (month[subset] == imon + 1)
#         means[imon, ::] = func(data[ix, ::], axis=0)  # Monthly Means assumes axis 0
#         counts[imon, ::] = np.sum(np.isfinite(data[ix, ::]), axis=0)  # Monthly counts assumes axis 0
#         data[month == imon + 1, ::] -= means[imon, ::]  # Anomalies
#
#     # means = np.where(counts < n*ratio_miss, np.nan, means)  # filter
#     if return_means:
#         return means, counts  # Monthly Means & Counts
#     else:
#         return data, counts  # Anomalies from Monthly Means


# def calculate_trends(data, dates, subset=None, use_anomalies=True):
#
#     if not isinstance(data, np.ndarray):
#         raise ValueError("Requires a numpy array (data)")
#     if not isinstance(dates, np.ndarray):
#         raise ValueError("Requires a numpy array (dates)")
#
#     data = data.copy()
#     dates = dates.copy()
#     per = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')  # factor for trends
#     if subset is not None:
#         if not isinstance(subset, np.ndarray):
#             raise ValueError("Requires a numpy array (subset)")
#         data = data[subset,::]
#         dates = dates[subset]
#
#     counts = np.sum(np.isfinite(data), axis=0)
#     if use_anomalies:
#         data, _ = anomalies(data, dates)
#
#     dates = dates.astype(long)  # Nano Seconds
#     dates -= dates[0]  # relative
#     # n = len(dates)
#     itrend = np.apply_along_axis(_trend_helper, 0, data, dates)  # Trends
#     # k = [unit]/time
#     # trend per day (nano s to day)
#     itrend = itrend * per   # Only the trend part
#     # itrend = np.where(counts < n * ratio_miss, np.nan, itrend)  # minimum required
#     return itrend, counts

#
# def fit_trends(data, dates, use_anomalies=True):
#     if not isinstance(data, np.ndarray):
#         raise ValueError("Requires a numpy array (data)")
#     if not isinstance(dates, np.ndarray):
#         raise ValueError("Requires a numpy array (dates)")
#
#     data = data.copy()
#     dates = dates.copy()
#     counts = np.sum(np.isfinite(data), axis=0)
#     # per = np.timedelta64(1, per) / np.timedelta64(1, 'ns')  # factor for trends
#     if use_anomalies:
#         base = data.copy()  # Make a copy of the data
#         data = anomaly(data, dates)
#     dates = dates.astype(long)  # Nano Seconds
#     dates -= dates[0]  # relative
#     # n = len(dates)
#     itrend = np.apply_along_axis(_fit_trend_helper, 0, data, dates)  # Trends
#     # k = [unit]/time
#     # trend per day (nano s to day)
#     # itrend = itrend * per   # Only the trend part
#     if use_anomalies:
#         itrend += base
#     # itrend = np.where(counts < n * ratio_miss, np.nan, itrend)  # minimum required
#     return itrend
#
#
# def _trend_helper(values, dates, nmin=3):
#     ii = np.isfinite(values)
#     if ii.sum() > nmin:
#         # (k,d), residuals, rank, singular values (2), rcond
#         p, _, _, _, _ = np.polyfit(dates[ii], values[ii], deg=1, full=True)
#         return p[0]  # K
#     return np.nan
#
#
# def _trend(dates, values, axis=0, nmin=3):
#     return np.apply_along_axis(_trend_helper, axis, values, dates, nmin=nmin)
#
#
# def _trend_helper_scipy(values, dates):
#     ii = np.isfinite(values)
#     if ii.sum() > 0:
#         # TODO: add possibility for null hypothesis
#         # slope, intercept, r_value, p_value, std_err
#         return linregress(dates[ii], values[ii])[0]
#     return np.nan
#
#
# def _fit_trend_helper(values, dates):
#     ii = np.isfinite(values)
#     if ii.sum() > 0:
#         coef = np.polyfit(dates[ii], values[ii], deg=1, full=False)
#         return np.polyval(coef, dates)
#     return np.full_like(values, np.nan)
#
#
# def _ptrend(data):
#     ii = np.isfinite(data)
#     # (k,d), residuals, rank, singular values (2), rcond
#     p, _, _, _, _ = np.polyfit(np.arange(data.size)[ii], data[ii], deg=1, full=True)
#     return p[0]  # K
