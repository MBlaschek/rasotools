# -*- coding: utf-8 -*-
import numpy as np

from . import dep
from ..fun import message

__all__ = ['mean', 'quantile', 'quantile_reference', 'index_samples']


def mean(data, breaks, axis=0, sample_size=130, borders=30, max_sample=1460, bounded=None, recent=False, ratio=True,
         **kwargs):
    """ Mean Adjustment of breakpoints

    Args:
        data (array): radiosonde data
        breaks (list, slice): breakpoint indices
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
    if not isinstance(breaks, (list, np.ndarray)):
        raise ValueError('requires a numpy array of list')

    data = data.copy()
    breaks = np.sort(np.asarray(breaks))
    nb = breaks.size
    imax = data.shape[axis]
    # Last Breakpoint
    if (breaks[-1] + sample_size) > imax:
        message(Warning("Reference Dataset is shorter than 1 year"), **kwargs)

    for ib in reversed(range(nb)):
        ibiased, isample, iref = index_samples(breaks, ib, axis, data.shape, recent=recent, sample_size=sample_size,
                                               borders=borders, max_sample=max_sample)

        data[ibiased] = dep.mean(data, iref, isample, axis=axis, sampleout=ibiased, sample_size=sample_size,
                                 bounded=bounded, ratio=ratio)

    return data


def quantile(data, breaks, axis=0, quantilen=None, sample_size=180, borders=30, max_sample=1460, bounded=None,
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

    breaks = np.sort(np.asarray(breaks))
    nb = len(breaks)
    imax = data.shape[axis]
    # Last Breakpoint
    if (breaks[-1] + sample_size) > imax:
        message(Warning("Reference Data set is shorter than 1 year"), **kwargs)

    for ib in reversed(range(nb)):
        ibiased, isample, iref = index_samples(breaks, ib, axis, data.shape, recent=recent, sample_size=sample_size,
                                               borders=borders, max_sample=max_sample)

        data[ibiased] = dep.quantile(data, iref, isample, quantilen, axis=axis, sampleout=ibiased,
                                     sample_size=sample_size, bounded=bounded, ratio=ratio)

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
        ydata = dep.quantile_reference(ydata, xdata, all_period, ref_period, quantilen, axis=axis,
                                       sampleout=slice(None), sample_size=sample_size, bounded=bounded, ratio=ratio)

    # 2. Loop Breakpoints and adjust backwards using adjusted ydata as reference
    for ib in reversed(range(nb)):
        ibiased, isample, iref = index_samples(breaks, ib, axis, xdata.shape, recent=recent, sample_size=sample_size,
                                               borders=borders, max_sample=max_sample)

        # Use same time sample for both data
        xdata[ibiased] = dep.quantile_reference(xdata, ydata, isample, isample, quantilen, axis=axis, sampleout=ibiased,
                                                sample_size=sample_size, bounded=bounded, ratio=ratio)

    return xdata, ydata


def index_samples(breaks, ibreak, axis, dshape, recent=False, sample_size=130, borders=30, max_sample=1460):
    """ Apply Breakpoints to data shape, return index lists

    Args:
        breaks (list, np.ndarray): Breakpoints
        ibreak (int): index of current Breakpoint
        axis (int): datetime axis
        dshape (tuple): shape of data array
        recent (bool): always use the whole recent part
        sample_size (int): minimum sample size
        borders (int): Breakpoint borders/uncertainty
        max_sample: maximum samples

    Returns:
        list, list, list : Biased , Sample, Reference
    """
    imax = dshape[axis]
    biased, ref = sample_indices(breaks, ibreak, imax, recent=recent)
    sample, ref = adjust_samples(biased, ref, sample_size, borders=borders, max_sample=max_sample)
    return idx2shp(biased, axis, dshape), idx2shp(sample, axis, dshape), idx2shp(ref, axis, dshape)


def idx2shp(period, axis, shape):
    index = [slice(None)] * len(shape)
    index[axis] = period
    return index


def sample_indices(breaks, ibreak, imax, recent=False):
    """ Indices of Samples before and after a bp

    Args:
        breaks (list, np.ndarray) :        Breakpoints
        ibreak (int) :          current Breakpoint
        imax (int) :            maximum index
        recent (bool) :         use all newest Data

    Returns:
        tuple: sample indices
    """
    n = len(breaks)

    if ibreak > 0:
        anfang = breaks[ibreak - 1]
    else:
        anfang = 0  # takes all the stuff? or only sometime after the break?

    mitte = breaks[ibreak]  # Mittelpunkt ist der Bruchpunkt

    if ibreak == (n - 1) or recent:
        ende = imax  # most recent
    else:
        ende = breaks[ibreak + 1]  # bp before

    sample1 = slice(anfang, mitte)  # erste Teil (indices niedriger)
    sample2 = slice(mitte, ende)    # Zweite Teil (indices höher)
    return sample1, sample2


def adjust_samples(ibiased, iref, sample_size, borders, max_sample):
    # start -> kleiner index (nächster Bruchpunkt, früher)
    # stop -> grosser index (Bruchpunkt)
    n = ibiased.stop - ibiased.start   # sample size
    isample = slice(ibiased.start, ibiased.stop)  # isample == ibiased
    if n - 2*borders > sample_size:
        isample = slice(ibiased.start + borders, ibiased.stop - borders)  # ohne Borders
        if n - 2*borders > max_sample:
            isample = slice(ibiased.stop - borders - max_sample, ibiased.stop - borders)  # nur max_sample

    n = iref.stop - iref.start
    if n - 2*borders > sample_size:
        iref = slice(iref.start + borders, iref.stop - borders)
        if n - 2*borders > max_sample:
            iref = slice(iref.start, iref.start + max_sample)

    return isample, iref


def break_iterator(breaks, axis, dshape, left_align=True, borders=30, max_sample=1460, **kwargs):
    """ break point *iterator* (list for iteration)

    Parameters
    ----------
    breaks : list
        list of breakpoints
    axis : int
        axis of datetime dim
    dshape : tuple
        shape of array
    left_align : bool
        period is between breakpoint and more recent part if True
    borders : int
        number of obs to remove near breakpoint
    max_sample : int
        maximum number of allowed obs
    kwargs : dict

    Returns
    -------
    list
        list of indices
    """
    breaks = list(breaks).copy()
    breaks.append(0)
    breaks = sorted(breaks, reverse=True)  # from present (large) to past (small)
    out = []
    jb = dshape[axis]   # maximum date index

    for i, ib in enumerate(breaks):
        sample = slice(ib, jb)  # smaller to larger index
        n = jb - ib
        if n > 2*borders:
            sample = slice(ib + borders, jb - borders)
            if n - 2*borders > max_sample:
                if left_align:
                    sample = slice(ib + borders, ib + borders + max_sample)
                else:
                    sample = slice(jb - borders - max_sample, jb - borders)

        out.append(tuple(idx2shp(sample, axis, dshape)))
        jb = ib
    return out
