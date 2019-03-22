# -*- coding: utf-8 -*-


__all__ = ['anomaly', 'trend', 'trend_per_month', 'trend_mon_percentile',
           'day_night_departures', 'correlate', 'covariance']
#
# global Variables
#
month_to_season = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                   10: 'SON', 11: 'SON', 12: 'DJF'}


#
# # Counting and Selecting
#


def select_period(data, dim='date', period=None):
    from ..fun import nanrange
    from xarray import DataArray

    if not isinstance(data, DataArray):
        raise ValueError('requires an xarray DataArray')

    if dim not in data.dims:
        raise ValueError("datetime dimension not found")

    if period is None:
        data.attrs['period'] = '%d-%d' % nanrange(data[dim].dt.year.values)
        return data
    else:
        iperiod = '%d-%d' % nanrange(data[dim].to_series()[period].index.year.values)
        data = data.sel(**{dim: period})
        data.attrs['period'] = iperiod
        return data


def count_per(data, dim=None, per='M', keep_attrs=True):
    import numpy as np
    from xarray import DataArray
    if not isinstance(data, DataArray):
        raise ValueError('requires an xarray DataArray')

    if dim is not None:
        idate = None
        if isinstance(data[dim].values[0], np.datetime64):
            idate = dim
        if idate is not None:
            counts = data.resample(**{dim: per}).count(dim, keep_attrs=keep_attrs)
            return counts
    else:
        return data.count(keep_attrs=keep_attrs)


def count_per_times(data, dim=None, keep_attrs=True):
    import numpy as np
    from xarray import DataArray
    if not isinstance(data, DataArray):
        raise ValueError('requires an xarray DataArray')

    if dim is not None:
        idate = None
        if isinstance(data[dim].values[0], np.datetime64):
            idate = dim
        if idate is not None:
            times = data.groupby(data[dim].dt.hour).count(dim, keep_attrs=keep_attrs)
            return times
    else:
        return data.count(keep_attrs=keep_attrs)


def estimate_sample_size(data, ratio=0.6, freq='12h'):
    """ Estimate the sample size from a timeseries, given freq and ratio

    Args:
        data (DataArray): Inputdata
        ratio (float): percentage of data as ratio
        freq (str): Pandas freq str

    Returns:
        int : Sample size according to freq and ratio
    """
    import numpy as np
    import pandas as pd
    from xarray import DataArray
    from ..fun import nanrange, nancount

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

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


#
# Climatology and Anomalies
#


def climatology(data, dim='date', period=None, keep_attrs=True):
    """

    Args:
        data (DataArray): Input Data
        dim (str): datetime dimension
        period (slice): datetime selection
        keep_attrs (bool) : xarray keep attributes
    Returns:
        DataArray : Climate Monthly Means
    """
    from xarray import DataArray
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")
    if dim not in data.dims:
        raise ValueError("datetime dimension not found")

    data = select_period(data, dim=dim, period=period)
    return data.groupby(dim + '.month').mean(dim, keep_attrs=keep_attrs)


def anomaly(data, dim='date', period=None, keep_attrs=True):
    """ Calculates the anomaly from the climatology per month of a time series

    Args:
        data (DataArray) : Inputdata
        dim (str) : datetime dimension
        period (slice, str) : Indices of Dates for calculation
        keep_attrs (bool) : xarray keep attributes
    Returns:
        DataArray : Anomalies
    """
    from xarray import DataArray, set_options
    from ..fun import set_attrs
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")

    if dim not in data.dims:
        raise ValueError("datetime dimension not found")

    data = data.copy()
    # Calculate Climatology
    clim = climatology(data, dim=dim, period=period, keep_attrs=keep_attrs)
    # Calculate Anomaly
    with set_options(keep_attrs=keep_attrs):
        data = data.groupby(dim + '.month') - clim

    data = data.drop('month')
    set_attrs(data.attrs, 'standard_name', add='_ano', default='anomaly')
    data.attrs['period'] = clim.attrs['period']
    return data


#
# Trend Estimation
#


def trend(data, dim='date', use_anomalies=True, period=None, min_periods=3, method='theil_sen',
          alpha=0.95, keep_attrs=True, **kwargs):
    """ Calculate Trend estimates

    Args:
        data (DataArray): input data array
        dim (str): datetime dimension
        use_anomalies (bool): calc. trends from anomalies (climatology removed)
        period (slice): time period for climatology
        min_periods (int): minimum number of values for trend estimate
        method (str): polyfit, theil_sen, linregress, lsq
        alpha (float): get confidence levels for that p value
        keep_attrs (bool): keep DataArray Attributes?

    Returns:
        DataArray : trends
    """
    import numpy as np
    from xarray import DataArray, Dataset
    from ..fun import linear_trend, set_attrs

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

    if dim not in data.dims:
        raise ValueError("datetime dimension not found")

    if method not in ['polyfit', 'theil_sen', 'linregress', 'lsq']:
        raise ValueError("Requires either polyfit, theil_sen, linregress or lsq")

    data = data.copy()
    per = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')  # factor for trends
    axis = data.dims.index(dim)
    coords = {idim: data[idim].copy() for idim in data.dims if idim != dim}
    dimens = list(data.dims[:])
    dimens.remove(dim)
    attrs = data.attrs.copy()

    if use_anomalies:
        data = anomaly(data, dim=dim, period=period, keep_attrs=keep_attrs)
        attrs['period'] = data.attrs['period']  # copy

    # Convert to standard time axis
    idates = data[dim].values.astype('long')  # Nano Seconds
    idates -= idates[0]  # relative Times
    # Trends
    # k = [unit]/time
    params = linear_trend(data.values, idates, method=method, alpha=alpha, nmin=min_periods, axis=axis)
    # slope and intercept
    idx = [slice(None)] * params.ndim
    idx[axis] = 0  # slope
    slope = DataArray(params[tuple(idx)] * per, coords=coords, dims=dimens, name='slope', attrs=attrs)
    set_attrs(slope, 'units', add='/day', default='1/day')
    set_attrs(slope, 'standard_name', add='_trend', default='trend')
    slope.attrs['cell_method'] = 'daily trend of anomalies' if use_anomalies else 'daily trend'

    idx[axis] = 1  # slope
    interc = DataArray(params[tuple(idx)], coords=coords, dims=dimens, name='intercept', attrs=attrs)
    set_attrs(interc, 'standard_name', add='_intercept', default='intercept')

    if params.shape[axis] > 2:
        if method == 'theil_sen':
            idx[axis] = 2  # slope lower
            aslope = DataArray(params[tuple(idx)] * per, coords=coords, dims=dimens, name='slope_min', attrs=attrs)
            set_attrs(aslope, 'units', add='/day', default='1/day')
            set_attrs(aslope, 'standard_name', add='_trend_min', default='trend_min')
            aslope.attrs['alpha'] = alpha
            aslope.attrs['cell_method'] = 'daily trend of anomalies' if use_anomalies else 'daily trend'

            idx[axis] = 3  # slope upper
            bslope = DataArray(params[tuple(idx)] * per, coords=coords, dims=dimens, name='slope_max', attrs=attrs)
            set_attrs(bslope, 'units', add='/day', default='1/day')
            set_attrs(bslope, 'standard_name', add='_trend_max', default='trend_max')
            bslope.attrs['alpha'] = alpha
            bslope.attrs['cell_method'] = 'daily trend of anomalies' if use_anomalies else 'daily trend'
            return Dataset({'slope': slope, 'intercept': interc, 'lower': aslope, 'upper': bslope})

        # r_value, p_value, std_err
        idx[axis] = 2  # R-value
        rslope = DataArray(params[tuple(idx)] ** 2, coords=coords, dims=dimens, name='r_squared', attrs=attrs)
        rslope.attrs['units'] = '1'
        set_attrs(rslope, 'standard_name', add='_r_squared', default='r_squared')

        idx[axis] = 3  # p-value
        bslope = DataArray(params[tuple(idx)], coords=coords, dims=dimens, name='p_value', attrs=attrs)
        bslope.attrs['units'] = '1'
        set_attrs(bslope, 'standard_name', add='_p_value', default='p_value')
        bslope.attrs['cell_method'] = 'p-value for null hypothesis(slope==0)'

        idx[axis] = 4  # std err
        sslope = DataArray(params[tuple(idx)], coords=coords, dims=dimens, name='std_err', attrs=attrs)
        set_attrs(sslope, 'units', add='/day', default='1/day')
        set_attrs(sslope, 'standard_name', add='_std_err', default='std_err')
        sslope.attrs['cell_method'] = 'standard error of slope'

        return Dataset({'slope': slope, 'intercept': interc, 'r_squared': rslope, 'p_value': bslope, 'std_err': sslope})
    return Dataset({'slope': slope, 'intercept': interc})


def trend_mon_percentile(data, dim='date', percentile=None, period=None,
                         min_periods=3, min_per_month=15, method='lsq', **kwargs):
    """ Monthly percentile trends

    Args:
        data (DataArray): input data
        dim (str): datetime dimension
        percentile (list): percentiles, int 1-99
        period (slice): datetime period for climatology
        min_periods (int): minimum values for trend
        min_per_month (int): minimum monthly count
        method (str): trend method
        **kwargs:

    Returns:
        Dataset : slope_perc_XX  for each percentile
    """
    import numpy as np
    from xarray import DataArray, Dataset
    from ..fun import sample_wrapper, xarray_function_wrapper as xfw, set_attrs
    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

    if percentile is None:
        percentile = [25, 50, 75]  # Quartils
    else:
        if any([iq < 1 for iq in percentile]):
            raise ValueError('Percentiles need to be integers [1, 99]')

    data = data.copy()
    axis = data.dims.index(dim)

    def perc(x, nmin=30, axis=0, **kwargs):
        return sample_wrapper(x, np.nanpercentile, nmin=nmin, axis=axis, **kwargs)

    trends = {}
    for iq in percentile:
        tmp = data.resample(**{dim: 'M'}).apply(xfw,
                                                wfunc=perc,
                                                nmin=min_per_month,
                                                q=iq,
                                                axis=axis,
                                                dim=dim)
        # Trend pro percentile
        itrend = trend(tmp, dim=dim, period=period, use_anomalies=False,
                       min_periods=min_periods, method=method, **kwargs)
        itrend = itrend['slope']
        set_attrs(itrend, 'standard_name', add='_perc', default='percentiles')
        itrend.attrs['cell_method'] = 'daily trend of monthly percentiles'
        itrend.attrs['min_per_month'] = min_per_month
        trends['slope_perc_%02d' % iq] = itrend

    return Dataset(trends)


def trend_per_month(data, dim='date', **kwargs):
    """ Trends per month

    Args:
        data (DataArray): input data
        dim (str): datetime dimension
        **kwargs:

    Returns:
        DataArray :
    """
    from xarray import DataArray
    if not isinstance(data, DataArray):
        raise ValueError("Requires a numpy array (data)")

    if dim not in data.dims:
        raise ValueError("datetime dimension not found")

    trends = data.groupby(dim + '.month').apply(trend, dim=dim, **kwargs)
    return trends


#
# Correlations
#


def correlate(x, y, dim='date', period=None, method='spearman', **kwargs):
    """ Correlation between Arrays

    Args:
        x (DataArray): input data
        y (DataArray): input data
        dim (str): datetime dimension
        period (slice): consider only that datetime period
        method (str): either spearman or pearson
        **kwargs:

    Returns:
        DataArray : correlation coefficients
    """
    from xarray import DataArray, align, apply_ufunc
    from ..fun import spearman_correlation, pearson_correlation, set_attrs

    if not isinstance(x, DataArray):
        raise ValueError("Requires a DataArray class object")

    if not isinstance(y, DataArray):
        raise ValueError("Requires a DataArray class object")

    if method not in ['spearman', 'pearson']:
        raise ValueError('Only spearman or pearson allowed')

    if dim not in x.dims or dim not in y.dims:
        raise ValueError('Dimension must be present in both Arrays')

    x = select_period(x, dim=dim, period=period)
    # Align
    x, y = align(x, y, join='left')
    axis = x.dims.index(dim)

    def sp_corr(x, y, dim, axis):
        jdims = list(x.dims)
        jdims.remove(dim)
        return apply_ufunc(spearman_correlation, x, y,
                           input_core_dims=[x.dims, y.dims],
                           output_core_dims=[jdims],
                           output_dtypes=[float],
                           kwargs={'axis': axis},
                           keep_attrs=True)

    def ps_corr(x, y, dim, axis):
        jdims = list(x.dims)
        jdims.remove(dim)
        return apply_ufunc(pearson_correlation, x, y,
                           input_core_dims=[x.dims, y.dims],
                           output_core_dims=[jdims],
                           output_dtypes=[float],
                           kwargs={'axis': axis},
                           keep_attrs=True)

    if method == 'spearman':
        corr = sp_corr(x, y, dim, axis)
    else:
        corr = ps_corr(x, y, dim, axis)

    set_attrs(corr, 'standard_name', add='_corr', default='correlation')
    corr.attrs['units'] = '1'
    corr.attrs['cell_method'] = '%s correlation with %s' % (method, y.name)
    return corr


def covariance(x, y, dim='date', period=None):
    """ Covariance

    Args:
        x:
        y:
        dim:
        period:

    Returns:

    """
    from xarray import DataArray, align, apply_ufunc
    from ..fun import covariance, set_attrs

    if not isinstance(x, DataArray):
        raise ValueError("Requires a DataArray class object")

    if not isinstance(y, DataArray):
        raise ValueError("Requires a DataArray class object")

    if dim not in x.dims or dim not in y.dims:
        raise ValueError('Dimension must be present in both Arrays')

    x = select_period(x, dim=dim, period=period)
    # Align
    x, y = align(x, y, join='left')
    axis = x.dims.index(dim)

    def nancov(x, y, dim, axis):
        jdims = list(x.dims)
        jdims.remove(dim)
        return apply_ufunc(covariance, x, y,
                           input_core_dims=[x.dims, y.dims],
                           output_core_dims=[jdims],
                           output_dtypes=[float],
                           kwargs={'axis': axis},
                           keep_attrs=True)

    corr = nancov(x, y, dim, axis)
    set_attrs(corr, 'standard_name', add='_cov', default='covariance')
    set_attrs(corr, 'units', add='2', default='2')
    set_attrs(corr, 'cell_method', set='covariance with %s' % y.name)
    return corr


#
# Modifications on sounding times
#


def standard_sounding_times(data, dim='date', times=(0, 12), span=6, freq='12h', return_indices=False, fillin=False, **kwargs):
    """ Standardize datetime to times per date, try to fill gaps

    Args:
        data (xarray.DataArray): Input DataArray
        dim (str): datetime dimension
        times (tuple): sounding times
        span (int): plus minus times (smaller than freq/2)
        freq (str): frequency of output times
        return_indices (bool): return indices for alignment
        fillin (bool): use all data to fill gaps

    Returns:
        xarray.DataArray : datetime standardized DataArray
    """
    import numpy as np
    import pandas as pd
    from xarray import DataArray
    from ..fun import message, update_kw, fix_datetime, idx2shp

    kwargs['mname'] = kwargs.get('mname', 'std_hours')

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    dates = data[dim].values.copy()

    # todo: check inconsitencies between times, span and freq
    #
    #  all possible dates
    min_date = pd.Timestamp(dates.min()).replace(hour=np.min(times), minute=0, second=0)
    max_date = pd.Timestamp(fix_datetime(dates.max(), span=span)).replace(hour=np.max(times), minute=0, second=0)
    message("Dates: ", min_date, " to ", max_date, " with ", freq, **kwargs)
    alldates = pd.date_range(min_date, max_date, freq=freq)

    if span > alldates.freq.n//2:
        raise ValueError("Frequency and Span need to be consistent (span < freq/2): ", freq, span)

    message("New Index:", alldates, **kwargs)
    new = data.reindex(**{dim: alldates})  # complete reindex to new dates (implicit copy)
    new['delay'] = (dim, np.zeros(alldates.size))  # new coordinate for delays
    #
    # matching dates (new in old)
    #
    old_logic = np.isin(dates, new[dim].values)
    #
    # old dates not matched to new ones
    #
    jtx = np.where(~old_logic)[0]
    new['delay'].values[~np.isin(new[dim].values, dates)] = np.nan
    message("Indices:", old_logic.sum(), "Candiates:", (~old_logic).sum(), old_logic.size, ">", alldates.size, **kwargs)
    axis = data.dims.index(dim)
    nn = 0
    indices = []
    if jtx.sum() > 0:
        _fix_datetime = np.vectorize(fix_datetime)
        #
        # Is there some data that fits within the given time window to new dates?
        #
        for m, i in enumerate(jtx):
            #
            # make standard date ->
            #
            idate = _fix_datetime(dates[i], span=span)
            j = np.where(alldates == idate)[0]
            if len(j) == 0:
                continue
            #
            # Indices of old and new arrays
            #
            newindex = idx2shp(j[0], axis, new.values.shape)
            oldindex = idx2shp(i, axis, new.values.shape)
            status = False
            #
            # Fill in whenever missing values
            #
            if fillin:
                new.values[newindex] = np.where(
                    (~np.isfinite(new.values[newindex]) & np.isfinite(data.values[oldindex])),
                    data.values[oldindex],
                    new.values[newindex])
                status = True
            #
            # when all are missing (whole profile)
            #
            elif np.isfinite(new.values[newindex]).sum() == 0:
                new.values[newindex] = data.values[oldindex]
                status = True
            else:
                pass

            #
            # keep original sounding times
            #
            if status:
                diff = (dates[i] - idate) / np.timedelta64(1, 'h')
                new['delay'].values[j] = -1 * diff  # pd.Timestamp(dates[j]).hour  # datetime of minimum
                message(m, dates[i], " + ", -1 * diff, " > ", idate, np.sum(np.isfinite(data.values[oldindex]), axis=0),
                        **update_kw('level', 1, **kwargs))

                indices += [(i, j[0])]
                nn += 1
            else:
                message(m, dates[i], " Passed ", idate, np.sum(np.isfinite(new.values[newindex]), axis=0),
                        np.sum(np.isfinite(data.values[oldindex]), axis=0),
                        **update_kw('level', 1, **kwargs))
    #
    # Add delay coordinate
    #
    message('Updated: ', nn, " of ", jtx.size, **kwargs)
    new.attrs['std_times'] = str(times)
    new['delay'].attrs['updated'] = nn
    new['delay'].attrs['missing'] = new['delay'].isnull().sum().values
    new['delay'].attrs['times'] = str(times)
    #
    # return indices for multi array alignment (datasets)
    #
    if return_indices:
        return new, np.array(indices)
    return new


def sel_hours(data, dim='date', times=(0, 12), **kwargs):
    """ Select only given hours

    Args:
        data (xarray.DataArray): Input DataArray
        dim (str): datetime dimension
        times (tuple, list): hours to consider

    Returns:
        xarray.DataArray : DataArray only at selected hours
    """
    from ..fun import message
    from xarray import DataArray

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    message('Selecting times:', times, dim, data.shape, **kwargs)
    return data.sel(**{dim: data[dim].dt.hour.isin(times)}).copy()  # selection


def to_hours(data, dim='date', standardize=True, times=(0, 12), as_dataset=False, hour='hour', **kwargs):
    """ Split Array into separate Arrays by time

    Args:
        data (xarray.DataArray): Input data
        dim (str): datetime dimension
        standardize (bool): apply standardization process or select only
        times (tuple, list): std hours
        as_dataset (bool): return hour dim as variables
        hour (str): name of hour dimension

    Returns:
        xarray.DataArray : datetime dimension split to days and hours
    """
    from ..fun import message, array2dataset, update_kw
    from pandas import Index
    from xarray import DataArray, concat

    kwargs['mname'] = kwargs.get('mname', 'to_hours')

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    data = data.copy()

    if standardize:
        data = standard_sounding_times(data, dim=dim, times=times, **update_kw('level', 1, **kwargs))
    else:
        data = sel_hours(data, dim=dim, times=times, **update_kw('level', 1, **kwargs))  # selection

    data = dict(data.groupby(dim + '.hour'))
    for ikey in data.keys():
        idata = data.get(ikey).copy()
        message(idata.name, ikey, idata.shape, **kwargs)
        #
        # transform datetime to daily freqency
        #
        idata[dim].values = idata[dim].to_index().to_period('D').to_timestamp().values
        data[ikey] = idata

    data = concat(data.values(), dim=Index(data.keys(), name=hour))
    # make sure the shape is as promissed:
    data = data.reindex({hour: list(times)})
    if as_dataset:
        return array2dataset(data, hour, rename={i: 't%02d' % i for i in times})
    return data


def from_hours(data, dim='date', hour='hour', **kwargs):
    """ Combine separate times to one datetime axis

    Args:
        data (DataArray): Inputdata
        dim (str): datetime dimension
        hour (str): time dimension
        **kwargs:

    Returns:
        DataArray : combined datetime axis DataArray
    """
    import pandas as pd
    from xarray import DataArray, concat

    kwargs['mname'] = kwargs.get('mname', 'from_hours')

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray with an hour dimension', type(data))

    if hour not in data.dims:
        raise ValueError('Requires an hour dimension', hour)

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    data = data.copy()
    data = dict(data.groupby(hour))
    for ikey, idata in data.items():
        # transform datetime to daily freqency (inplace?)
        idata[dim].values = (idata[dim].to_index() + pd.DateOffset(hours=int(ikey))).values
    return concat(data.values(), dim=dim).sortby(dim)


def day_night_departures(data, dim='date', standardize=True, **kwargs):
    """ Day-Night departures form data

    Args:
        data (DataArray): input data
        dim (str): datetime dimension
        standardize (bool): standardize to 0, 12 times before departure
        **kwargs:

    Returns:

    """
    from xarray import DataArray

    kwargs['mname'] = kwargs.get('mname', 'dn_dep')

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    data = to_hours(data, dim=dim, standardize=standardize, times=[0, 12], **kwargs)
    attrs = data.attrs.copy()
    name = data.name
    data = data.sel(hours=12) - data.sel(hours=0)
    data.name = name + '_dep'
    data.attrs.update(attrs)
    data.attrs['standard_name'] += '_day_night_dep'
    data.attrs['cell_method'] = 'noon - night'
    return data
