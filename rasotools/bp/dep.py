# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['mean', 'quantile']


def mean(data, sample1, sample2, sampleout=None, axis=0, sample_size=130, bounded=None, ratio=True,
         median=False, **kwargs):
    """ Adjustment method using mean differences or ratios

    ratio=False
    data[sampleout]  + (MEAN(data[sample1]) - MEAN(data[sample2]))

    ratio=True
    data[sampleout]  * (MEAN(data[sample1]) / MEAN(data[sample2]))

    Args:
        data (np.ndarray): input data
        sample1 (list, slice): slice reference
        sample2 (list, slice): slice sample
        sampleout (list, slice): slice output sample
        axis (int): date axis
        sample_size (int): minimum sample size
        bounded (tuple): allowed variable range
        ratio (bool): use ratio or difference?
        median (bool): use median instead of mean?

    Returns:
        np.ndarray : mean adjusted data
    """

    lbound, ubound = None, None
    if bounded is not None:
        lbound, ubound = bounded

    if sampleout is None:
        sampleout = sample2

    if median:
        sample1 = nanmedian(data[sample1], axis=axis, n=sample_size)
        sample2 = nanmedian(data[sample2], axis=axis, n=sample_size)
    else:
        sample1 = nanmean(data[sample1], axis=axis, n=sample_size)
        sample2 = nanmean(data[sample2], axis=axis, n=sample_size)

    if bounded is not None:
        tmp = data[sampleout].copy()

    if ratio:
        dep = sample1 / sample2
        dep = np.where(np.isfinite(dep), dep, 1.)  # replace NaN with 1
        sampleout = data[sampleout] * dep
    else:
        dep = sample1 - sample2
        sampleout = data[sampleout] + dep

    if bounded is not None:
        with np.errstate(invalid='ignore'):
            sampleout = np.where((sampleout < lbound) | (sampleout > ubound), tmp, sampleout)

    return sampleout


def quantile(data, sample1, sample2, quantiles, sampleout=None, axis=0, sample_size=130, bounded=None, ratio=True,
             **kwargs):
    """ Adjustment method using quantile differences or ratios

    ratio=False
    data[sample1] + ( percentiles(data[sample1]) - percentiles(data[sample2]) )

    ratio=True
    data[sample1] * ( percentiles(data[sample1]) / percentiles(data[sample2]) )

    Args:
        data (np.ndarray): input data
        sample1 (list): slice reference
        sample2 (list): slice sample
        quantiles (list): percentiles to use
        sampleout (list): slice output sample
        axis (int): date axis
        sample_size (int): minimum sample size
        bounded (tuple): allowed variable range
        ratio (bool): use ratio or difference?

    Returns:
        np.ndarray : percentile adjusted data
    """
    lbound, ubound = None, None
    if bounded is not None:
        lbound, ubound = bounded

    if sampleout is None:
        sampleout = sample2

    # Add 0 and 100, and remove them
    quantiles = np.unique(np.concatenate([[0], quantiles, [100]]))
    quantiles = quantiles[1:-1]  # remove 0 and 100

    # Sample sizes are enough?
    nsample1 = np.isfinite(data[sample1]).sum(axis=axis) > sample_size
    nsample2 = np.isfinite(data[sample2]).sum(axis=axis) > sample_size

    # Percentiles of the samples
    sample1 = np.nanpercentile(data[sample1], quantiles, axis=axis)
    sample2 = np.nanpercentile(data[sample2], quantiles, axis=axis)

    sampleout = data[sampleout]
    if bounded is not None:
        tmp = sampleout.copy()

    if ratio:
        dep = np.where(sample2 != 0., sample1 / sample2, 1.)
        dep = np.where(nsample1 & nsample2, dep, 1.)  # apply sample size
        dep = np.where(np.isfinite(dep), dep, 1.)  # replace NaN
    else:
        dep = sample1 - sample2
        dep = np.where(nsample1 & nsample2, dep, 0.)  # apply sample size

    # Interpolate adjustments to sampleout shape and data
    dep = apply_quantile_adjustments(sampleout, sample2, dep, axis=axis)

    if ratio:
        sampleout = sampleout * dep
    else:
        sampleout = sampleout + dep

    if bounded is not None:
        with np.errstate(invalid='ignore'):
            sampleout = np.where((sampleout < lbound) | (sampleout > ubound), tmp, sampleout)

    return sampleout


def quantile_reference(xdata, ydata, sample1, sample2, quantiles, sampleout=None, axis=0, sample_size=130, bounded=None,
                       ratio=True, **kwargs):
    """ Adjustment method using quantile differences or ratios

    ratio=False
    xdata[sample1] + ( percentiles(xdata[sample1]) - percentiles(ydata[sample2]) )

    ratio=True
    xdata[sample1] * ( percentiles(xdata[sample1]) / percentiles(ydata[sample2]) )

    Args:
        xdata (np.ndarray): input data
        ydata (np.ndarray): reference data
        sample1 (list, slice): slice reference
        sample2 (list, slice): slice sample
        quantiles (list): percentiles to use
        sampleout (list, slice): slice output sample
        axis (int): date axis
        sample_size (int): minimum sample size
        bounded (tuple): allowed variable range
        ratio (bool): use ratio or difference?

    Returns:
        np.ndarray : percentile adjusted data
    """
    lbound, ubound = None, None
    if bounded is not None:
        lbound, ubound = bounded

    if sampleout is None:
        sampleout = sample2

    # Add 0 and 100, and remove them
    quantiles = np.unique(np.concatenate([[0], quantiles, [100]]))
    quantiles = quantiles[1:-1]  # remove 0 and 100

    # Sample sizes are enough?
    nsample1 = np.isfinite(xdata[sample1]).sum(axis=axis) > sample_size
    nsample2 = np.isfinite(ydata[sample2]).sum(axis=axis) > sample_size

    # Percentiles of the samples
    sample1 = np.nanpercentile(xdata[sample1], quantiles, axis=axis)
    sample2 = np.nanpercentile(ydata[sample2], quantiles, axis=axis)

    sampleout = xdata[sampleout]
    if bounded is not None:
        tmp = sampleout.copy()

    if ratio:
        dep = np.where(sample2 != 0., sample1 / sample2, 1.)
        dep = np.where(nsample1 & nsample2, dep, 1.)  # apply sample size
        dep = np.where(np.isfinite(dep), dep, 1.)  # replace NaN
    else:
        dep = sample1 - sample2
        dep = np.where(nsample1 & nsample2, dep, 0.)  # apply sample size

    # Interpolate adjustments to sampleout shape and data
    dep = apply_quantile_adjustments(sampleout, sample2, dep, axis=axis)

    if ratio:
        sampleout = sampleout * dep
    else:
        sampleout = sampleout + dep

    if bounded is not None:
        with np.errstate(invalid='ignore'):
            sampleout = np.where((sampleout < lbound) | (sampleout > ubound), tmp, sampleout)

    return sampleout


#
# Helper functions
#


def nanmean(data, n=130, axis=0):
    """ Nan omitting Mean

    Args:
        data (np.ndarray): data including NaN
        n (int): minimum sample size
        axis (int): datetime axis

    Returns:
        np.ndarray : mean values at axis
    """
    nn = np.isfinite(data).sum(axis=axis)
    nn = np.where(nn < n, np.nan, nn)
    return np.nansum(data, axis=axis) / nn


def nanmedian(data, n=130, axis=0):
    """ Nan omitting Median

    Args:
        data (np.ndarray): data including NaN
        n (int): minimum sample size
        axis (int): datetime axis

    Returns:
        np.ndarray : median values at axis
    """
    nn = np.isfinite(data).sum(axis=axis)
    nn = np.where(nn < n, np.nan, 1.)
    return np.nanmedian(data, axis=axis) * nn


def apply_quantile_adjustments(data, quantiles, adjustment, axis=0):
    """ Helper Function for applying quantile adjustments

    Args:
        data (np.ndarray): data
        quantiles (np.ndarray): quantiles, points of adjustments
        adjustment (np.ndarray): adjustments to be interpolated
        axis (int): axis of datetime

    Returns:
        np.ndarray : interpolated adjustment, same shape as data
    """
    in_dims = list(range(data.ndim))
    # last dim == axis, Last dim should be time/date
    data = np.transpose(data, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    quantiles = np.transpose(quantiles, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    adjustment = np.transpose(adjustment, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    adjusts = np.zeros(data.shape)
    # Indices for iteration + expand
    inds = np.ndindex(data.shape[:-1])  # iterate all dimensions but last
    inds = (ind + (Ellipsis,) for ind in inds)  # add last as ':' == Ellipsis == all
    for ind in inds:
        # INTERP -> Xnew, Xpoints, Fpoints
        adjusts[ind] = np.interp(data[ind], quantiles[ind], adjustment[ind])

    # Transform back to original shape
    return np.transpose(adjusts, in_dims[:axis] + in_dims[axis + 1:] + [axis])

#
#  Old stuff
#


# def quantile_reference(xdata, ydata, subsample, sampleout, sample_size, quantiles, bounded=None,
#                  ratio=True, verbose=0):
#     lbound, ubound = None, None
#     if bounded is not None:
#         lbound, ubound = bounded
#
#     itx = [slice(None)] * xdata.ndim
#     jtx = [slice(None)] * xdata.ndim
#     itx[0] = sampleout
#     tmp_qad = ydata[itx].copy()  # piece of sample
#     if xdata.ndim > 1:
#         for i in range(xdata.shape[1]):
#             itx[1] = i
#             jtx[1] = i
#             if ratio:
#                 dep = quantile_era_ratio_1d(xdata[itx], ydata[itx], subsample, quantiles, sample_size)
#                 # dep = dep + (1 - dep) / factor if factor > 1 else dep
#                 tmp_qad[jtx] = ydata[itx] * dep
#             else:
#                 dep = quantile_era_1d(xdata[itx], ydata[itx], subsample, quantiles, sample_size)
#                 tmp_qad[jtx] = ydata[itx] + dep  #/ factor  # one value per level (might be negativ)
#     else:
#         if ratio:
#             dep = quantile_era_ratio_1d(xdata[itx], ydata[itx], subsample, quantiles, sample_size)
#             # dep = dep + (1 - dep) / factor if factor > 1 else dep
#             tmp_qad[jtx] = ydata[itx] * dep
#         else:
#             dep = quantile_era_1d(xdata[itx], ydata[itx], subsample, quantiles, sample_size)
#             tmp_qad[jtx] = ydata[itx] + dep  #/ factor  # one value per level (might be negativ)
#
#     if bounded is not None:
#         itx = [slice(None)] * xdata.ndim
#         itx[0] = sampleout
#         tmp_qad = np.where((tmp_qad < lbound) | (tmp_qad > ubound), ydata[itx], tmp_qad)
#
#     # print_stuff('QE', xdata, subsample, subsample, sampleout, tmp_qad, np.nanmedian(dep, axis=0), verbose)
#     # broken because dep does not have level information!!!
#     return tmp_qad

#
# def quantile_1d(x, sample1, sample2, meinequantilen, sample_size, sample3=None, return_mean=False, linear=True,
#                 verbose=0):
#     from support import qstats
#     #
#     s1d = x[sample1]  # truth (sample1)
#     s2d = x[sample2]  # biased (sample2)
#     ok1 = np.isfinite(s1d).sum() > sample_size
#     ok2 = np.isfinite(s2d).sum() > sample_size
#     # Enough Data to calculate ?
#     if not np.any(ok1 & ok2):
#         if sample3 is not None:
#             return np.zeros(x[sample3].shape)  # return only zeros
#         else:
#             return np.zeros(s2d.shape)
#     #
#     # add 0 and 100 to remove them afterwards
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))
#     # Mean of quantile boxes( not 0 and 100 )
#     count1, m1 = qstats(s1d, meinequantilen[1:-1], counts=3)
#     count2, m2 = qstats(s2d, meinequantilen[1:-1], counts=3)
#     #
#     if verbose > 1:
#         print "Quantiles:", meinequantilen
#         print "Sample 1: ", count1
#         print "Sample 2: ", count2
#     #
#     qb = np.nanpercentile(s1d, meinequantilen)  # truth
#     qa = np.nanpercentile(s2d, meinequantilen)  # biased
#     #
#     diffs = qb - qa  # Difference of quantiles (1st and lst for interp)
#     xp = qa  # x Punkte der Interpolation (Quantile Edges)
#     xp[:-1] = m2  # x Punkte der Interpolation
#     diffs[:-1] = m1 - m2  # y punkte der interpolation
#     if return_mean:
#         return m1, m2
#
#     # interpolate quantile differences
#     # we know the difference at quantile values and linearly interpolate this to all values
#     #
#     # how to handle end-point ?
#     # if not extrapolate:
#     #     diffs = diffs[1:-1] # trim
#     #     xp = xp[1:-1]       # trim
#     #
#     # Spline or linear interpolation
#     if not linear:
#         tck = interpolate.splrep(xp, diffs, s=0)
#         if sample3 is not None:
#             out = interpolate.splev(x[sample3], tck, der=0)  # does this retain nan ?
#         else:
#             out = interpolate.splev(s2d, tck, der=0)
#     #
#     else:
#         # to all Data in sample / but not when missing!
#         if sample3 is not None:
#             out = np.interp(x[sample3], xp, diffs)
#         else:
#             out = np.interp(s2d, xp, diffs)
#
#     # turn missing into zero
#     return np.where(np.isfinite(out), out, 0.)  # size of sample 2 or sample 3 # no adjustment
#
#
# def quantile_ratio_1d(x, sample1, sample2, meinequantilen, sample_size, sample3=None, verbose=0):
#     """ Quantile matching between Samples using ratio of quantiles
#     adjustment = sample2 * interp. ( Q. of sample 1 / Q. of sample 2 )
#     """
#     s1d = x[sample1]  # truth (sample1)
#     s2d = x[sample2]  # biased (sample2)
#     ok1 = np.isfinite(s1d).sum() > sample_size
#     ok2 = np.isfinite(s2d).sum() > sample_size
#
#     if sample3 is not None:
#         out = x[sample3]
#     else:
#         out = s2d
#
#     if not ok1 or not ok2:
#         return np.ones(out.shape)  # only 1
#
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))  # add 0, 1 and 99, 100
#     qb = np.nanpercentile(s1d, meinequantilen)  # [1:-1], interpolation='linear')       # truth   (1 to 99 %)
#     qa = np.nanpercentile(s2d, meinequantilen)  # [1:-1], interpolation='linear')       # biased
#     # ratio of quantiles (scale truth to biased)
#     with np.errstate(invalid='ignore', divide='ignore'):
#         qdiff = np.where(qa > 0, qb / qa, qb)  # remove 0  Exception NullDivide
#         qdiff = np.where(qb == 0, 1, qdiff)  # no 0 allowed
#     return np.interp(out, qa, qdiff)  # interpolate quantile diff to all Data
#
#
# def quantile_era_1d(x, y, sample1, meinequantilen, sample_size, verbose=0):
#     from support import qstats
#     # Match ERA to RASO
#     # Sampling Period:
#     s1d = x[sample1]  # truth  (sample1) RASO
#     s2d = y[sample1]  # biased (sample2) ERA
#     #
#     ok1 = np.isfinite(s1d).sum() > sample_size
#     ok2 = np.isfinite(s2d).sum() > sample_size
#     if not ok1 or not ok2:
#         return y
#     #
#     # add 0 and 100
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))
#     # Be sure to remove 0,100 now
#     # Mean of quantile boxes( not 0 and 100 )
#     count1, m1 = qstats(s1d, meinequantilen[1:-1])
#     count2, m2 = qstats(s2d, meinequantilen[1:-1])
#     # ok1 = count1[:-1] > sample_size
#     # ok2 = count2[:-1] > sample_size
#     # # Enough Data to calculate ?
#     # if not np.any(ok1 & ok2):
#     #     return y  # np.zeros(y.shape)
#     #
#     if verbose > 1:
#         print "Quantiles:", meinequantilen
#         print "Sample 1: ", count1
#         print "Sample 2: ", count2
#     #
#     qb = np.nanpercentile(s1d, meinequantilen)  # truth
#     qa = np.nanpercentile(s2d, meinequantilen)  # biased
#     #
#     diffs = qb - qa  # Difference of quantiles (1st and lst for interp)
#     xp = qa
#     xp[:-1] = m2  # x punkte der interpolation ( ? NAN )
#     diffs[:-1] = m1 - m2  # y punkte der interpolation
#     out = np.interp(y, xp, diffs)  # new, old, old values
#     # turn missing into zero
#     out = np.where(np.isfinite(out), out, 0.)
#     # add ontop of variable
#     return out  # size of y
#
#
# def quantile_era_ratio_1d(x, y, sample1, meinequantilen, sample_size, verbose=0):
#     s1d = x[sample1]  # truth (sample1)
#     s2d = y[sample1]  # biased (sample2)
#     # ? sample_size ?
#     ok1 = np.isfinite(s1d).sum() > sample_size
#     ok2 = np.isfinite(s2d).sum() > sample_size
#     if not ok1 or not ok2:
#         return np.ones(y.shape)  # only 1
#
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))  # add 0, 1 and 99, 100
#     qb = np.nanpercentile(s1d, meinequantilen)  # [1:-1], interpolation='linear')       # truth   (1 to 99 %)
#     qa = np.nanpercentile(s2d, meinequantilen)  # [1:-1], interpolation='linear')       # biased
#     qdiff = np.where(qa > 0, qb / qa, qb)  # remove 0  Exception NullDivide
#     qdiff = np.where(qb == 0, 1, qdiff)  # no 0 allowed
#     return np.interp(y, qa, qdiff)  # interpolate quantile diff to all Data
#
#
# def print_stuff(name, nref, nsam, m1, m2, m3, dep, test, sample_size, verbose):
#     from raso.support import message
#     try:
#         z = 0
#         for i, j, k, l, m, n in zip(nsam, nref, dep, m1, m2, m3):
#             message("%s %2d [%6d] %6d <> %6d (+*%6.3f) [%5r] >> %9.4f %9.4f %9.4f" % (
#             name, z, sample_size, i, j, k, k != test, l, m, n),
#                           verbose)
#             z +=1
#     except:
#         message("%s [%6d] %6d <> %6d (+*%6.3f) [%5r] >> %9.4f %9.4f %9.4f" % (
#             name, sample_size, nsam, nref, dep, dep != test, m1, m2, m3), verbose)

# Works terribly
# rank like difference
# def qmap(x, sample1, sample2):
#     s1d = x[sample1]  # truth (sample1)
#     s2d = x[sample2]  # biased (sample2)
#     m1 = np.isfinite(s1d)
#     m2 = np.isfinite(s2d)
#     # sorts / interpolates Data to second sample:
#     q1 = np.nanpercentile(s1d[m1], np.linspace(0, 100, len(s2d[m2])))
#     # Sort, but keep order
#     ix = np.argsort(s1d[m2])
#     new = s2d.copy()  # copy
#     new[m2[ix]] = q1 - new[m2[ix]]  # Fill in Difference
#     return new


# works terribly
# interpolate bin-means, difference then
# def qmap_mean_departure(x, sample1, sample2, meinequantilen, sample_size,
#                         return_mean=False, linear=True):
#     from support import qstats
#
#     s1d = x[sample1]  # truth (sample1)
#     s2d = x[sample2]  # biased (sample2)
#
#     # add 0 and 100
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))
#
#     qb = np.nanpercentile(s1d, meinequantilen)  # truth
#     qa = np.nanpercentile(s2d, meinequantilen)  # biased
#     mean1 = np.copy(qb)
#     mean2 = np.copy(qa)
#
#     # Mean of quantile boxes( not 0 and 100 )
#     count1, m1 = qstats(s1d, meinequantilen[1:-1], counts=sample_size)
#     count2, m2 = qstats(s2d, meinequantilen[1:-1], counts=sample_size)
#     # only missing ?
#     mean1[:-1] = m1
#     mean2[:-1] = m2
#     # interpolation of bin-means
#     if linear:
#         m1d = np.interp(s2d, qb[1:], mean1[:-1])  # interpoliere Mittelwerte zu Daten
#         m2d = np.interp(s2d, qa[1:], mean2[:-1])
#     else:
#         tck = interpolate.splrep(qb[1:], mean1[:-1], s=0)
#         m1d = interpolate.splev(s2d, tck, der=0)
#         tck = interpolate.splrep(qa[1:], mean2[:-1], s=0)
#         m2d = interpolate.splev(s2d, tck, der=0)
#     # difference
#     if return_mean:
#         return m1, m2
#
#     return m1d - m2d  # one value


# def qmap_var_departure(x, y, sample1, sample2, meinequantilen, sample_size, verbose=0):
#     """ Quantile matching with secondary variable
#
#     Parameters
#     ----------
#     x               Variable to match quantiles
#     y               Variable to calc. quantiles
#     sample1
#     sample2
#     meinequantilen
#     sample_size
#     verbose
#
#     Returns
#     -------
#
#     """
#     s1d = y[sample1]  # truth (sample1)
#     s1dx = x[sample1]
#
#     s2d = y[sample2]  # biased (sample2)
#     s2dx = x[sample2]
#
#     q1 = np.nanpercentile(s1d, meinequantilen)  # truth
#     q2 = np.nanpercentile(s2d, meinequantilen)  # biased
#
#     nq = len(meinequantilen) + 1
#
#     # index of Data in quantile ranges
#     index_s1 = np.digitize(s1d, q1)
#     index_s2 = np.digitize(s2d, q2)
#
#     # check if there is enough Data inside the bins
#     s1_counts = np.bincount(index_s1)
#     s2_counts = np.bincount(index_s2)
#
#     # output counts ?
#     if verbose > 1:
#         print s1_counts
#         print s2_counts
#
#     diffs = np.zeros(nq)
#     xp = np.zeros(nq)
#
#     for ibin in np.arange(nq):
#         m2 = np.nanmean(s2dx[index_s2 == ibin])  # biased
#         if (s2_counts[ibin] > sample_size) & (s1_counts[ibin] > sample_size):
#             # Average(func) of that Quantile Range
#             m1 = np.nanmean(s1dx[index_s1 == ibin])  # truth
#
#             # new[ index_s3 == ibin ] = m1 - m2 # apply difference to all
#             diffs[ibin] = m1 - m2
#         xp[ibin] = m2
#
#     out = np.interp(s2d, xp, diffs)
#     return np.where(np.isfinite(out), out, 0.)  # size of sample 2 or sample 3 # no adjustment
#


#
# def qmap_departure_mod(x, sample1, sample2, meinequantilen, sample_size,
#                        sample3=None, return_mean=False, linear=True,
#                        verbose=0):
#     # check if there is an anomaly like dpd30
#     # remove these values -> make sure we adjustments the histogram for it
#     #
#     # Differenz between quantiles -> estimate if there is huge bias
#     # or have a lot of Data in one class ?
#     # adjust only comparable samples
#     from support import qstats
#     #
#     s1d = x[sample1]  # truth (sample1)
#     s2d = x[sample2]  # biased (sample2)
#     #
#     # add 0 and 100
#     meinequantilen = np.unique(np.concatenate([[0], meinequantilen, [100]]))
#     # Be sure to remove 0,100 now
#     # Mean of quantile boxes( not 0 and 100 )
#     count1, m1 = qstats(s1d, meinequantilen[1:-1], counts=sample_size)
#     count2, m2 = qstats(s2d, meinequantilen[1:-1], counts=sample_size)
#     ok1 = count1[:-1] > sample_size
#     ok2 = count2[:-1] > sample_size
#     # Enough Data to calculate ?
#     if not np.any(ok1 & ok2):
#         if sample3 is not None:
#             return np.zeros(x[sample3].shape)  # return only zeros
#         else:
#             return np.zeros(s2d.shape)
#     #
#     if verbose > 1:
#         print "Quantiles:", meinequantilen
#         print "Sample 1: ", count1
#         print "Sample 2: ", count2
#     #
#     qb = np.nanpercentile(s1d, meinequantilen)  # truth
#     qa = np.nanpercentile(s2d, meinequantilen)  # biased
#     #
#     diffs = qb - qa  # Difference of quantiles (1st and lst for interp)
#     xp = qa
#     xp[:-1] = m2  # x punkte der interpolation ( ? NAN )
#     diffs[:-1] = m1 - m2  # y punkte der interpolation
#     if return_mean:
#         return m1, m2
#     # interpolate quantile differences
#     # how to handle end-point ?
#     # if not extrapolate:
#     #     diffs = diffs[1:-1] # trim
#     #     xp = xp[1:-1]       # trim
#     # Spline or linear interpolation
#     if not linear:
#         tck = interpolate.splrep(xp, diffs, s=0)
#         if sample3 is not None:
#             out = interpolate.splev(x[sample3], tck, der=0)  # does this retain nan ?
#         else:
#             out = interpolate.splev(s2d, tck, der=0)
#     #
#     else:
#         # to all Data in sample / but not when missing!
#         if sample3 is not None:
#             out = np.interp(x[sample3], xp, diffs)
#         else:
#             out = np.interp(s2d, xp, diffs)
#
#     # turn missing into zero
#     return np.where(np.isfinite(out), out, 0.)  # size of sample 2 or sample 3 # no adjustment
#     #
#     #
