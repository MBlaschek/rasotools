# -*- coding: utf-8 -*-
import numpy as np

from . import dep
from ..fun import message, nancount, kwc, kwu

__all__ = ['mean', 'quantile', 'quantile_reference']


def mean(data, breaks, axis=0, sample_size=130, borders=30, max_sample=1460, bounded=None, recent=False, ratio=True,
         **kwargs):
    """ Mean Adjustment of breakpoints

    Args:
        data (array): radiosonde data
        breaks (list): breakpoint indices
        axis (int): axis of datetime
        sample_size (int): minimum sample size
        borders (int): adjust breaks with borders
        max_sample (int): maximum sample size
        bounded (tuple): variable allowed range
        recent (bool): use full reference period
        ratio (bool): calculate ratio, instead of difference
        **kwargs:

    Returns:
        array : adjusted data
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("requires a numpy array")
    if not isinstance(breaks, (np.ndarray, list)):
        raise ValueError('requires a numpy array')

    data = data.copy()
    dshape = data.shape  # Shape of data (date x levs)
    imax = dshape[axis]  # maximum index
    breaks = np.sort(np.asarray(breaks))  # sort
    breaks = np.append(np.insert(breaks, 0, 0), imax)  # 0 ... ibreaks ... None
    nb = breaks.size

    for i in reversed(range(1, nb - 1)):
        # Indices
        im = breaks[i - 1]  # earlier
        ib = breaks[i]  # current breakpoint
        if recent:
            ip = imax
        else:
            ip = breaks[i + 1]  # later

        # Slices all axes
        iref = slice(ib, ip)
        isample = slice(im, ib)
        isample = idx2shp(isample, axis, dshape)
        iref = idx2shp(iref, axis, dshape)
        # Before Adjustments
        before = np.nanmean(data[isample], axis=axis)
        # Apply Adjustments
        data[isample] = dep.mean(data, iref, isample, axis=axis, sample_size=sample_size, max_sample=max_sample,
                                 borders=borders, bounded=bounded, ratio=ratio)
        # Debug infos
        if kwc('verbose', value=2, **kwargs):
            sdata = stats(data, iref, isample, axis=axis, a=before)
            sdata = np.array_str(sdata, precision=2, suppress_small=True)
            message(sdata, **kwu('level', 1, **kwargs))
    return data


def quantile(data, breaks, axis=0, quantilen=None, sample_size=130, borders=30, max_sample=1460, bounded=None,
             recent=False, ratio=True, **kwargs):
    """ Percentile Adjustment of breakpoints

    Args:
        data (array): radiosonde data
        breaks (list): breakpoint indices
        axis (int): axis of datetime dimension
        quantilen (list): percentile ranges
        sample_size (int): minimum sample size
        borders (int): adjust breaks with borders
        max_sample (int): maximum sample size
        bounded (tuple): variable allowed range
        recent (bool): use full reference period
        ratio (bool): calculate ratio, instead of difference
        **kwargs:

    Returns:
        array : adjusted data
    """

    if not isinstance(data, np.ndarray):
        raise ValueError("requires a numpy array")
    if not isinstance(breaks, (list, np.ndarray)):
        raise ValueError('requires a numpy array of list')

    data = data.copy()
    if quantilen is None:
        quantilen = np.arange(0, 101, 10)

    dshape = data.shape
    imax = dshape[axis]
    breaks = np.sort(np.asarray(breaks))  # sort
    breaks = np.append(np.insert(breaks, 0, 0), imax)  # 0 ... ibreaks ... None
    nb = breaks.size

    for i in reversed(range(1, nb - 1)):
        # Indices
        im = breaks[i - 1]  # earlier
        ib = breaks[i]  # current breakpoint
        if recent:
            ip = imax
        else:
            ip = breaks[i + 1]  # later

        # slices all axes
        ibiased = slice(im, ib)
        isample = slice(im, ib)
        isample = idx2shp(isample, axis, dshape)
        iref = idx2shp(iref, axis, dshape)

        before = np.nanmean(data[isample], axis=axis)
        # Apply Adjustments
        data[ibiased] = dep.quantile(data, iref, isample, quantilen, axis=axis, sample_size=sample_size,
                                     max_sample=max_sample, borders=borders, bounded=bounded, ratio=ratio)
        # Debug infos
        if kwc('verbose', value=2, **kwargs):
            sdata = stats(data, iref, isample, axis=axis, a=before)
            sdata = np.array_str(sdata, precision=2, suppress_small=True)
            message(sdata, **kwu('level', 1, **kwargs))

    return data


def quantile_reference(xdata, ydata, breaks, axis=0, quantilen=None, sample_size=365, borders=30, max_sample=1460,
                       bounded=None, recent=False, ratio=True, ref_period=None, adjust_reference=True, **kwargs):
    # xdata is RASO
    # ydata is Reference
    if not isinstance(xdata, np.ndarray):
        raise ValueError("requires a numpy array")
    if not isinstance(ydata, np.ndarray):
        raise ValueError("requires a numpy array")
    if not isinstance(breaks, (list, np.ndarray)):
        raise ValueError('requires a numpy array of list')

    xdata = xdata.copy()
    ydata = ydata.copy()
    if quantilen is None:
        quantilen = np.arange(0, 101, 10)

    breaks = np.sort(np.asarray(breaks))
    nb = len(breaks)
    imax = xdata.shape[axis]
    # Last Breakpoint
    if (breaks[-1] + sample_size) > imax:
        message(Warning("Reference Data set is shorter than 1 year"))

    if adjust_reference:
        all_period = [slice(None)] * xdata.ndim
        # 1. Adjust Reference to match distribution of unbiased period
        if ref_period is None:
            ref_period = all_period[:]  # copy
            ref_period[axis] = slice(breaks[-1], None)
        else:
            if isinstance(ref_period, (slice, list)):
                iref_period = all_period[:]  # copy
                iref_period[axis] = ref_period
                ref_period = iref_period
            else:
                raise ValueError('Reference period needs to be a list or slice')

        # Apply Dist. from xdata[ref_period] to all ydata  (Match dists.)
        ydata = dep.quantile_reference(ydata, xdata, tuple(all_period), tuple(ref_period), quantilen, axis=axis,
                                       sampleout=slice(None), sample_size=sample_size, bounded=bounded, ratio=ratio)

    # 2. Loop Breakpoints and adjust backwards using adjusted ydata as reference
    for ib in reversed(range(nb)):
        ibiased, isample, iref = index_samples(breaks, ib, axis, xdata.shape, recent=recent, sample_size=sample_size,
                                               borders=borders, max_sample=max_sample)

        # Use same time sample for both data
        xdata[ibiased] = dep.quantile_reference(xdata, ydata, isample, isample, quantilen, axis=axis, sampleout=ibiased,
                                                sample_size=sample_size, bounded=bounded, ratio=ratio)

    return xdata, ydata


def idx2shp(idx, axis, shape):
    index = [slice(None)] * len(shape)
    index[axis] = idx
    return tuple(index)


def stats(data, ref, sample, axis=0, a=None):
    # print counts, means, dep
    sn = nancount(data[sample], axis=axis)
    s = np.nanmean(data[sample], axis=axis)
    rn = nancount(data[ref], axis=axis)
    r = np.nanmean(data[ref], axis=axis)
    if a is not None:
        return np.array([sn, a, s, r - s, r, rn]).T
    return np.array([sn, s, r - s, r, rn]).T

# def index_samples(breaks, ibreak, axis, dshape, recent=False, sample_size=130, borders=30, max_sample=1460):
#     """ Apply Breakpoints to data shape, return index lists
#
#     Args:
#         breaks (list, np.ndarray): Breakpoints
#         ibreak (int): index of current Breakpoint
#         axis (int): datetime axis
#         dshape (tuple): shape of data array
#         recent (bool): always use the whole recent part
#         sample_size (int): minimum sample size
#         borders (int): Breakpoint borders/uncertainty
#         max_sample: maximum samples
#
#     Returns:
#         list, list, list : Biased , Sample, Reference
#     """
#     imax = dshape[axis]
#     biased, ref = sample_indices(breaks, ibreak, imax, recent=recent)
#     sample, ref = adjust_samples(biased, ref, sample_size, borders=borders, max_sample=max_sample)
#     return idx2shp(biased, axis, dshape), idx2shp(sample, axis, dshape), idx2shp(ref, axis, dshape)
# def sample_indices(breaks, ibreak, imax, recent=False):
#     """ Indices of Samples before and after a bp
#
#     Args:
#         breaks (list, np.ndarray) :        Breakpoints
#         ibreak (int) :          current Breakpoint
#         imax (int) :            maximum index
#         recent (bool) :         use all newest Data
#
#     Returns:
#         tuple: sample indices
#     """
#     n = len(breaks)
#
#     if ibreak > 0:
#         anfang = breaks[ibreak - 1]
#     else:
#         anfang = 0  # takes all the stuff? or only sometime after the break?
#
#     mitte = breaks[ibreak]  # Mittelpunkt ist der Bruchpunkt
#
#     if ibreak == (n - 1) or recent:
#         ende = imax  # most recent
#     else:
#         ende = breaks[ibreak + 1]  # bp before
#
#     sample1 = slice(anfang, mitte)  # erste Teil (indices niedriger)
#     sample2 = slice(mitte, ende)  # Zweite Teil (indices höher)
#     return sample1, sample2

#
# def adjust_samples(ibiased, iref, sample_size, borders, max_sample):
#     # start -> kleiner index (nächster Bruchpunkt, früher)
#     # stop -> grosser index (Bruchpunkt)
#     n = ibiased.stop - ibiased.start  # sample size
#     isample = slice(ibiased.start, ibiased.stop)  # isample == ibiased
#     if n - 2 * borders > sample_size:
#         isample = slice(ibiased.start + borders, ibiased.stop - borders)  # ohne Borders
#         if n - 2 * borders > max_sample:
#             isample = slice(ibiased.stop - borders - max_sample, ibiased.stop - borders)  # nur max_sample
#
#     n = iref.stop - iref.start
#     if n - 2 * borders > sample_size:
#         iref = slice(iref.start + borders, iref.stop - borders)
#         if n - 2 * borders > max_sample:
#             iref = slice(iref.start, iref.start + max_sample)
#
#     return isample, iref
#
# def breakpoint_zone(x, thres, axis=0, k=200, recent=False, target=None):
#     n = x.shape[axis]
#
#     if target is not None:
#         m1 = target
#     else:
#         m1 = np.nanmean(x, axis=axis)  # total median as backup
#
#     dep = x - m1
#     j = None
#
#     for i in range(n - k, 0, -k):
#         if not recent:
#             j = i + k if i + k < n else n
#         itx = idx2shp(slice(i, j), axis, x.shape)
#         s1 = np.where(np.isfinite(dep[itx]).sum(axis=axis) > 0, np.nanmean(dep[itx], axis=axis), m1)
#         l = i - k if i - k > 0 else 0
#         jtx = idx2shp(slice(l, i), axis, x.shape)
#         s2 = np.where(np.isfinite(dep[jtx]).sum(axis=axis) > 0, np.nanmean(dep[jtx], axis=axis), m1)
#         dep[jtx] += np.where(np.abs(s1 - s2) > thres, (s1 - s2), 0.)
#     return m1 + dep
