# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from . import message

__all__ = ['standard_dates_times']


########################################################################################################################
#
#  DATETIME SUBROUTINES
#
########################################################################################################################

# todo add message logging
def standard_dates_times(data, hours=[0, 6, 12, 18], span=None, freq=None, **kwargs):
    """ Fix datetime index to standard sounding times (0, 6, 12, 18) pm 3h

    Parameters
    ----------
    data : ndarray
        Dates to be standardized
    kwargs : dict
        opther options, ignored

    Returns
    -------
    out : ndarray
        Array with dates conform to standard times

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> random_minutes = [pd.Timedelta(i, unit='m') for i in np.random.randint(0,60,size=100)]
    >>> dates = pd.date_range('1-1-1999 00:00:00', periods=100, freq='2h')
    >>> dates += pd.Series(random_minutes)
    >>> newdates = standard_dates_times(dates)
    >>> data = pd.DataFrame({'original': dates, 'new': newdates})
    >>> data['departure'] = data['original'] - data['new']
    >>> data.where(data.departure != pd.Timedelta(0)).dropna()
    """
    if not isinstance(data, np.ndarray) or not isinstance(data[0], np.datetime64):
        raise ValueError("Requires a numpy array with datetime64 values")

    hours = np.sort(hours)
    if span is None:
        span = np.unique(np.diff(hours))
    else:
        span = np.array(span)

    if span.size > 1:
        raise ValueError("Hours not equally divided")

    for i, ihour in enumerate(hours):
        if np.any((ihour + span) > hours[slice(i + 1, None)]):
            raise ValueError('Overlapping time spans')

    data = data.copy()  # numpy
    pdata = pd.DatetimeIndex(data)
    itime = pdata.hour.values + np.round(pdata.minute.values / 60.)
    itx = np.isin(itime, hours)  # already conform to hours
    if all(itx):
        return data  # nothing to do

    rx = data[~itx]  # subset, only non hours times [0,6,12,18]
    for ihour in hours:
        message(ihour, span, rx.shape, mname='STD', **kwargs)
        rx = fix_datetime_mod(rx, ihour, span)

    newdata = data.copy()
    newdata[~itx] = rx
    # check for duplicates
    only, counts = np.unique(newdata, return_counts=True)
    itimes = only[counts > 1]  # any duplicates ?
    if len(itimes) > 0:
        message(len(itimes), mname='STD', **kwargs)
        for idate in itimes:
            itx = (newdata == idate)  # index of duplicates
            trdiff = (newdata[itx] - data[itx]) / np.timedelta64(1, 'h')  # time difference normalized to hours
            tdiff = np.abs(trdiff)  # Absolute time differences
            imin = np.min(tdiff)  # What is the minimum for each duplicate
            if np.sum(tdiff == imin) > 1:
                imin = np.min(trdiff)  # use the minimum closest to hour
                newdata[itx] = np.where(trdiff == imin, newdata[itx], data[itx])  # Take the minimum and set back
            else:
                newdata[itx] = np.where(tdiff == imin, newdata[itx], data[itx])  # Take the abs minimum and set back
        message(newdata[itx], data[itx], mname='STD', **kwargs)
    # There might be duplicates
    return newdata


@np.vectorize
def fix_datetime_mod(itime, hour, span):
    itime = pd.Timestamp(itime)
    istd = itime.replace(hour=hour, minute=0, second=0, microsecond=0)
    imin = istd - pd.offsets.Hour(span)
    imax = istd + pd.offsets.Hour(span)
    offset = pd.offsets.Day(1)

    # print(imin, itime, imax)
    if imin <= itime < imax:
        return istd.to_datetime64()
    elif (imin + offset) <= itime < (imax + offset):
        return (istd + offset).to_datetime64()
    else:
        return itime.to_datetime64()
