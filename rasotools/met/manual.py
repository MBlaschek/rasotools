# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ..fun import message

__all__ = ['remove_spurious_values']


def remove_spurious_values(dates, data, axis=0, num_years=10, value=30, bins=None, **kwargs):
    """ Specifc Function to remove a certain value from the Histogram (DPD)

    Args:
        dates (ndarray): dates
        data (ndarray): values
        axis (int): axis of dates
        num_years (int): groupby number of years
        value (float): value
        bins (list): histogram bins

    Returns:
        ndarray, int, ndarray : data, counts of occurence, mask
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Requires a numpy array")

    if bins is None:
        bins = np.arange(0, 60)

    years = pd.DatetimeIndex(dates).year.values
    y_min = np.min(years)  # 1st year

    # Groupby (same years)
    # years = (years - y_min) / int(num_years)  # integer truncation -> groups of 10 years

    message("from %d , %d years" % (y_min, np.size(np.unique(years))), **kwargs)

    # Indices for array
    jtx = [slice(None)] * data.ndim

    count = 0

    mask = np.full(data.shape, False, dtype=bool)

    for iyear in np.unique(years)[::-1]:
        # 10 year running statistics
        jtx[axis] = np.where((years <= iyear) & (years >= (iyear - num_years)), True, False)
        # Apply over date dimension, for all other
        new = np.apply_along_axis(_myhistogram, axis, data[jtx], value=value, bins=bins, **message_level(kwargs))
        data[jtx] = np.where(new, np.nan, data[jtx])  # Set nan or leave it
        mask[jtx] = new
        message("%d %d [%d]" % (iyear, np.sum(np.isfinite(data[jtx])), np.sum(new)), **kwargs)
        count += np.sum(new)

    return data, count, mask


def _myhistogram(data, value=30, bins=10, normed=True, excess=5, thres=1, **kwargs):
    """ Customized histogram

    Args:
        data:
        value:
        bins:
        normed:

    Returns:
        ndarray : boolean mask
    """
    itx = np.isfinite(data)  # remove missing
    counts, divs = np.histogram(data[itx], bins=bins, normed=normed)

    n = len(divs) + 1
    if np.sum(itx) < n * 2:
        return np.full(data.shape, False)  # not enough data

    # mark values with anomalous high distribution! add 10% or 5%
    jtx = np.argmax(np.histogram(value, bins=bins, normed=normed)[0])  # box of value in the current histogram
    crit = (excess / 100.) * (n / 100.)  # 5%  depends on sample size

    # Value is above critical
    if counts[jtx] > crit:
        message("[%f] %f [%f] %f > %f" % (value, counts[jtx - 1], counts[jtx], counts[jtx + 1], crit), **kwargs)
        # Before and After are below critical
        if (counts[jtx - 1] < crit) & (counts[jtx + 1] < crit):
            # Any data ?
            # if np.any(counts[slice(jtx - 1, jtx + 2)] > 0):
            with np.errstate(invalid='ignore'):
                #
                return itx & (data > (value - thres)) & (data < (value + thres))

    return np.full(data.shape, False)
