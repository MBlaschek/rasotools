# -*- coding: utf-8 -*-


def vrange(x, axis=0):
    """ Calculate min and max

    Args:
        x (ndarray): input data
        axis (int): axis
    """
    import numpy as np
    return np.min(x, axis=axis), np.max(x, axis=axis)


def nanrange(x, axis=0):
    """ Calculate min and max removing NAN

    Args:
        x (ndarray): input data
        axis (int): axis

    Returns:
        tuple : min, max
    """
    import numpy as np
    return np.nanmin(x, axis=axis), np.nanmax(x, axis=axis)


def nancount(x, axis=0, keepdims=False):
    """

    Args:
        x (ndarray): input data
        axis (int): axis
        keepdims (bool): keep dimensions
    """
    import numpy as np
    return np.sum(np.isfinite(x), axis=axis, keepdims=keepdims)


def nanfunc(data, n=130, axis=0, nmax=1460, borders=0, func=None, flip=False, fargs=()):
    """ Nan omitting function (numpy)

    Args:
        data (np.ndarray): data including NaN
        n (int): minimum sample size
        axis (int): datetime axis
        nmax (int): maximum sample size
        borders (int): border sample to ignore
        func (callable): function to call
        flip (bool): reverse data before applying the function
        args (tuple): function arguments

    Returns:
        np.ndarray : func of values at axis, with sample size, borders and maximum
    """
    import numpy as np
    if func is None:
        func = np.nanmean
    return np.apply_along_axis(sample, axis, data, n, nmax, func, borders=borders, flip=flip, fargs=fargs)


def sample(values, nmin, nmax, func, borders=0, flip=False, fargs=()):
    # variable output, One value or array
    # todo make a numba version of this function
    # make everything loops and stuff
    # find nan
    # find max, min number of data and apply function
    import numpy as np
    itx = np.isfinite(values)
    n = itx.sum()
    j = 0
    if n > nmax:
        if n > (nmax + borders):
            j = borders
        if flip:
            return func(np.flip(values[itx])[j:(nmax + j)], *fargs)  # reversed
        return func(values[itx][j:(nmax + j)], *fargs)  # normal

    elif n < nmin:
        return func(values, *fargs) * np.nan

    else:
        if n > (nmin * 2 + borders):
            j = borders
        if flip:
            return func(np.flip(values[j:]), *fargs)
        return func(values[j:], *fargs)


def rmse(x, y=None, axis=None):
    """ RMSE

    Args:
        x:
        y:
        axis:

    Returns:
        float : RMSE
    """
    import numpy as np
    if y is None:
        y = 0.
    return np.sqrt(np.nanmean((x - y) * (x - y), axis=axis))


def fuzzy_all(x, axis=0, thres=2):
    """ fuzzy all true or not

    Args:
        x (ndarray): input data (bool)
        axis (int): axis
        thres (int): threshold for axis sum

    Returns:
        bool
    """
    import numpy as np
    if np.sum(x, axis=axis) > (np.shape(x)[axis] / np.float(thres)):
        return True
    else:
        return False


def fuzzy_equal(x, y, z):
    """ Fuzzy equal

    Args:
        x (ndarray): input a
        y (ndarray): input b
        z (ndarray): uncertainty of input a

    Returns:
        ndarray : bool array
    """
    return (y < (x + z)) & (y > (x - z))


def distance(lon, lat, ilon, ilat, miles=False):
    """
    Calculates the distance between a point and an array of points
    Distance in kilometers

    Parameters
    ----------
    lon     Longitudes of points
    lat     Latitudes of points
    ilon    Longitude of Position
    ilat    Latitude of Position

    Returns
    -------
    numpy.array / same as input

    Notes
    -----
    Haversine Formula
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    import numpy as np
    rad_factor = np.pi / 180.0  # for trignometry, need angles in radians
    mlat, mlon, jlat, jlon = lat * rad_factor, lon * rad_factor, ilat * rad_factor, ilon * rad_factor
    dlon = mlon - jlon
    dlat = mlat - jlat
    # vector + vector * value * vector
    a = np.sin(dlat / 2) ** 2 + np.cos(mlat) * np.cos(jlat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    if miles:
        r = 3956  # Radius of the earth in miles.
    else:
        r = 6371  # Radius of earth in kilometers.
    return c * r


def linear_trend(y, x, method='polyfit', alpha=None, nmin=3, fit=False, axis=0, **kwargs):
    """ calculate linear trend from data

    Args:
        y (ndarray): values
        x (ndarray): time valus
        method (str): estimation method (polyfit, theil_sen, linregress, lsq)
        alpha:
        nmin:
        fit:
        axis:

    Returns:

    """
    import numpy as np
    if method not in ['polyfit', 'theil_sen', 'linregress', 'lsq']:
        raise ValueError('Requires either polyfit, theil_sen, linregress or lsq')

    if alpha is None:
        alpha = 0.95

    if y.ndim > 1:
        return np.apply_along_axis(linear_trend, axis, y, x, method=method, alpha=alpha, nmin=nmin, fit=fit)

    if method == 'polyfit':
        params = _trend_polyfit_wrapper(y, x, nmin=nmin)
    elif method == 'theil_sen':
        params = _trend_theilslopes_wrapper(y, x, nmin=nmin, alpha=alpha)
    elif method == 'linregress':
        params = _trend_linregress_wrapper(y, x, nmin=nmin)
    else:
        params = _trend_regression_wrapper(y, x, nmin=nmin)

    if fit:
        return params[0] * x + params[1]

    return params


def _trend_polyfit_wrapper(y, x, nmin=3, **kwargs):
    import numpy as np
    ii = np.isfinite(y)
    if ii.sum() > nmin:
        # (k,d), residuals, rank, singular values (2), rcond
        p, _, _, _, _ = np.polyfit(x[ii], y[ii], deg=1, full=True)
        return np.asarray(p)  # slope and intercept
    return np.array([np.nan])


def _trend_theilslopes_wrapper(y, x, nmin=3, **kwargs):
    import numpy as np
    from scipy.stats import theilslopes
    ii = np.isfinite(y)
    if ii.sum() > nmin:
        # k, d, min, max wenn alpha
        return np.asarray(theilslopes(y[ii], x[ii], **kwargs))
    return np.array([np.nan] * 4)


def _trend_linregress_wrapper(y, x, nmin=3):
    import numpy as np
    from scipy.stats import linregress
    ii = np.isfinite(y)
    if ii.sum() > nmin:
        # k, d, min, max wenn alpha
        return np.asarray(linregress(x[ii], y[ii]))
    return np.array([np.nan] * 5)


def _trend_regression_wrapper(y, x, **kwargs):
    import numpy as np
    n = np.size(x)
    xm = np.nanmedian(x)
    ym = np.nanmedian(y)
    ya = y - ym
    xa = x - xm
    # variance and covariances
    xss = np.nansum(xa ** 2) / (n - 1)  # variance of x (with df as n-1)
    # yss = (ya ** 2).sum() / (n - 1)  # variance of y (with df as n-1)
    xys = np.nansum(xa * ya) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    # df = n - 2
    # r = xys / (xss * yss)**0.5
    # t = r * (df / ((1 - r) * (1 + r)))**0.5
    # p = stats.distributions.t.sf(abs(t), df)

    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5
    return np.array([slope, intercept])


def mann_kendall_test(x, alpha=0.05):
    """
    This function is derived from code originally posted by Sat Kumar Tomer
    (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert
    1987) is to statistically assess if there is a monotonic upward or downward
    trend of the variable of interest over time. A monotonic upward (downward)
    trend means that the variable consistently increases (decreases) through
    time, but the trend may or may not be linear. The MK test can be used in
    place of a parametric linear regression analysis, which can be used to test
    if the slope of the estimated linear regression line is different from
    zero. The regression analysis requires that the residuals from the fitted
    regression line be normally distributed; an assumption not required by the
    MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best
    viewed as an exploratory analysis and is most appropriately used to
    identify stations where changes are significant or of large magnitude and
    to quantify these findings.

    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)

    """
    import numpy as np
    from scipy.stats import norm

    n = len(x)

    # calculate S
    s = 0
    for k in range(n - 1):
        s += sum(np.sign(x[k + 1:] - x[k]))
        # for j in range(k+1, n):
        #    s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x, tp = np.unique(x, return_counts=True)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # there are some ties in data
        # tp = np.zeros(unique_x.shape)
        # for i in range(len(unique_x)):
        #     tp[i] = sum(x == unique_x[i])
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:  # s == 0:
        z = 0

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        trend = -1
    elif (z > 0) and h:
        trend = 1
    else:
        trend = 0

    return trend, h, p, z


def num_samples_trend_test(beta, delta, std_dev, alpha=0.05, n=4, num_iter=1000,
                           tol=1e-6, num_cycles=10000, m=5):
    """
    This function is an implementation of the "Calculation of Number of Samples
    Required to Detect a Trend" section written by Sat Kumar Tomer
    (satkumartomer@gmail.com) which can be found at:
    http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    As stated on the webpage in the URL above the method uses a Monte-Carlo
    simulation to determine the required number of points in time, n, to take a
    measurement in order to detect a linear trend for specified small
    probabilities that the MK test will make decision errors. If a non-linear
    trend is actually present, then the value of n computed by VSP is only an
    approximation to the adjustments n. If non-detects are expected in the
    resulting data, then the value of n computed by VSP is only an
    approximation to the adjustments n, and this approximation will tend to be less
    accurate as the number of non-detects increases.

    Input:
        beta: probability of falsely accepting the null hypothesis
        delta: change per sample period, i.e., the change that occurs between
               two adjacent sampling times
        std_dev: standard deviation of the sample points.
        alpha: significance level (0.05 default)
        n: initial number of sample points (4 default).
        num_iter: number of iterations of the Monte-Carlo simulation (1000
                  default).
        tol: tolerance level to decide if the predicted probability is close
             enough to the required statistical power value (1e-6 default).
        num_cycles: Total number of cycles of the simulation. This is to ensure
                    that the simulation does finish regardless of convergence
                    or not (10000 default).
        m: if the tolerance is too small then the simulation could continue to
           cycle through the same sample numbers over and over. This parameter
           determines how many cycles to look back. If the same number of
           samples was been determined m cycles ago then the simulation will
           stop.

        Examples
        --------
          >>> num_samples = check_num_samples(0.2, 1, 0.1)

    """
    import numpy as np

    # Initialize the parameters
    power = 1.0 - beta
    P_d = 0.0
    cycle_num = 0
    min_diff_P_d_and_power = abs(P_d - power)
    best_P_d = P_d
    max_n = n
    min_n = n
    max_n_cycle = 1
    min_n_cycle = 1
    # Print information for user
    print("Delta (gradient): {}".format(delta))
    print("Standard deviation: {}".format(std_dev))
    print("Statistical power: {}".format(power))

    # Compute an estimate of probability of detecting a trend if the estimate
    # Is not close enough to the specified statistical power value or if the
    # number of iterations exceeds the number of defined cycles.
    while abs(P_d - power) > tol and cycle_num < num_cycles:
        cycle_num += 1
        # print("Cycle Number: {}".format(cycle_num))
        count_of_trend_detections = 0

        # Perform MK test for random sample.
        # could use range here
        for i in range(num_iter):
            r = np.random.normal(loc=0.0, scale=std_dev, size=n)
            x = r + delta * np.arange(n)
            trend, h, p, z = mann_kendall_test(x, alpha)
            if h:
                count_of_trend_detections += 1
        P_d = float(count_of_trend_detections) / num_iter

        # Determine if P_d is close to the power value.
        if abs(P_d - power) < tol:
            # print("P_d: {}".format(P_d))
            # print("{} samples are required".format(n))
            return n

        # Determine if the calculated probability is closest to the statistical
        # power.
        if min_diff_P_d_and_power > abs(P_d - power):
            min_diff_P_d_and_power = abs(P_d - power)
            best_P_d = P_d

        # Update max or min n.
        if n > max_n and abs(best_P_d - P_d) < tol:
            max_n = n
            max_n_cycle = cycle_num
        elif n < min_n and abs(best_P_d - P_d) < tol:
            min_n = n
            min_n_cycle = cycle_num

        # In case the tolerance is too small we'll stop the cycling when the
        # number of cycles, n, is cycling between the same values.
        elif (abs(max_n - n) == 0 and
              cycle_num - max_n_cycle >= m or
              abs(min_n - n) == 0 and
              cycle_num - min_n_cycle >= m):
            # print("Number of samples required has converged.")
            # print("P_d: {}".format(P_d))
            # print("Approximately {} samples are required".format(n))
            return n

        # Determine whether to increase or decrease the number of samples.
        if P_d < power:
            n += 1
            print("P_d: {}".format(P_d))
            print("Increasing n to {}".format(n))
            print("")
        else:
            n -= 1
            print("P_d: {}".format(P_d))
            print("Decreasing n to {}".format(n))
            print("")
            if n == 0:
                raise ValueError("Number of samples = 0. This should not happen.")


def sample_wrapper(data, func, nmin=130, axis=0, **kwargs):
    import numpy as np
    nn = np.isfinite(data).sum(axis=axis)
    nn = np.where(nn < nmin, np.nan, 1.)
    return func(data, axis=axis, **kwargs) * nn


def xarray_function_wrapper(x, wfunc=None, **kwargs):
    """ Map a numpy function that is not currently in xarray to xarray with apply_ufunc

    Args:
        x (DataArray): Input data
        func (callable): function to call, e.g.: np.nanpercentile
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


def covariance(x, y, axis=0):
    import numpy as np
    return np.nanmean((x - np.nanmean(x, axis=axis, keepdims=True))
                      * (y - np.nanmean(y, axis=axis, keepdims=True)), axis=axis)


def pearson_correlation(x, y, axis=0):
    import numpy as np
    return covariance(x, y, axis=axis) / (np.nanstd(x, axis=axis) * np.nanstd(y, axis=axis))


def spearman_correlation(x, y, axis=0):
    import bottleneck as bn
    x_ranks = bn.nanrankdata(x, axis=axis)
    y_ranks = bn.nanrankdata(y, axis=axis)
    return pearson_correlation(x_ranks, y_ranks, axis=axis)
