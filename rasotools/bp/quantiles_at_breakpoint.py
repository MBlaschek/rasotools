# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

__all__ = ['quantiles_at_breakpoint']


################################################
#
# Calculate Quantiles at breakpoint
#
#
################################################


# Make a plot for each break point with before and after breakpoint !?
# can we use that as well after the correction !?
# 
# TODO: convert to a function that can be used to understand what is different
# either calculate quantiles between samples and show
# or mean differences
#

def quantiles_at_breakpoint(data, var, dvar=None, quantilen=None, ibreak=None, sample_size=730, borders=180, verbose=0):
    """Calculate Quantiles at the breakpoints
    """
    from .dep import quantile
    from ..tools import sample_indices
    funcid = '[QAB] '

    if not isinstance(var, str):
        raise ValueError(funcid + "var Requires  a string")

    if dvar is not None and not isinstance(dvar, str):
        raise ValueError(funcid + "dvar Requires  a string")

    if dvar is None:
        dvar = var

    print funcid + "Data from Variable: ", dvar
    if not isinstance(data, (pd.DataFrame, pd.Panel)):
        raise ValueError("Require a DataFrame or Panel as input")

    if quantilen is None:
        quantilen = np.arange(0, 101, 10)

    quantilen = quantilen[(quantilen < 100) & (quantilen > 0)]  # drop 0 and 100
    qss = sample_size / (len(quantilen) + 1) / 2  # sample size per quantile
    print funcid + "Quantilen: ", quantilen
    print funcid + "Global Sample size: %d , per quantile(%d): %d" % (sample_size, len(quantilen), qss)
    mlabels = ["Q%d" % i for i in quantilen]
    mlabels.append(">")

    if isinstance(data, pd.DataFrame):
        if not data.columns.isin([var, '%s_breaks' % var]).sum() == 2:
            raise ValueError(funcid + "Variable not found: %s or %s_breaks in %s" % (var, var, str(data.columns)))
        # convert to panel
        if 'p' not in data.columns:

            out = {}
            #  get Breakpoints
            int_breaks = np.where((data['%s_breaks' % var] > 0))[0]
            breaks = data.index[int_breaks]
            nb = len(breaks)
            if nb == 0:
                raise RuntimeError(funcid + "No Breakpoints found in %s and %s_breaks" % (var, var))

            print "Found Breaks: ", nb
            print str(breaks)
            if (int_breaks[-1] + sample_size) > data.shape[0]:
                print funcid + "Reference Data set is shorter than 1 year"

            for ib in reversed(range(nb)):
                if ibreak is not None and ibreak != ib:
                    print funcid + "Looking for: ", breaks[ibreak], " at ", breaks[ib]
                    continue
                # ibiased is everything between breakpoints
                # isample is minus the borders -> used to calculate
                ibiased, isample, iref = sample_indices(int_breaks, ib, data.index,
                                                        sample_size=sample_size,
                                                        borders=borders,
                                                        recent=False,
                                                        verbose=verbose - 1)
                # Quantiles at the breakpoint
                b1, c1, quants1 = qstats(data[dvar].values[iref], quantilen, qss)
                b2, c2, quants2 = qstats(data[dvar].values[isample], quantilen, qss)

                if verbose > 0:
                    print funcid + " %s : %s " % (dvar, breaks[ib])
                    print funcid + " Qs(B): ", quants1
                    print funcid + " Qs(#): ", c1
                    print funcid + " Qs(B): ", quants2
                    print funcid + " Qs(#): ", c2

                out[str(breaks[ib])] = pd.DataFrame({'Ref': quants1.tolist(), 'Bias': quants2.tolist()}, index=mlabels)
            return out

        # when there are pressure levels
        data = data.reset_index().set_index(['date', 'p']).to_panel()

    else:
        if not data.items.isin([var, '%s_breaks' % var]).sum() == 2:
            raise ValueError(funcid + "Variable not found: %s or %s_breaks in %s" % (var, var, str(data.items)))

    # per level
    #  get Breakpoints
    int_breaks = np.where((data['%s_breaks' % var] > 0).any(1))[0]
    breaks = data.major_axis[int_breaks]
    nb = len(breaks)
    if nb == 0:
        raise RuntimeError(funcid + "No Breakpoints found in %s and %s_breaks" % (var, var))

    print "Found Breaks: ", nb
    print str(breaks)
    if (int_breaks[-1] + sample_size) > data.shape[0]:
        print funcid + "Reference Data set is shorter than 1 year"

    out = {}

    for ib in reversed(range(nb)):
        if ibreak is not None and ibreak != ib:
            print funcid + "Looking for: ", breaks[ibreak], " at ", breaks[ib]
            continue
        # ibiased is everything between breakpoints
        # isample is minus the borders -> used to calculate
        ibiased, isample, iref = sample_indices(int_breaks, ib, data.major_axis,
                                                sample_size=sample_size,
                                                borders=borders,
                                                recent=False,
                                                verbose=verbose - 1)

        # Quantiles at the breakpoint
        def myqstats(x, quantilen, sample_size):
            c, y = qstats(x, quantilen, sample_size)
            return y

        quants1 = np.apply_along_axis(myqstats,
                                      0,
                                      data[dvar].values[iref],
                                      quantilen,
                                      qss)

        quants2 = np.apply_along_axis(myqstats,
                                      0,
                                      data[dvar].values[isample],
                                      quantilen,
                                      qss)
        out[str(breaks[ib])] = pd.Panel({'Ref': quants1, 'Bias': quants2}, major_axis=mlabels,
                                        minor_axis=data.minor_axis)

    return out




def qstats(x, quantilen, counts=0, func=np.nanmean):
    x = np.asarray(x)
    n = len(quantilen)
    n += 1
    out = np.full(n, np.nan)
    miss = np.isfinite(x)  # if all are missing
    if np.sum(miss) > 0:
        qs = np.nanpercentile(x, quantilen)  # can be the same value for two different quantiles ? > needs to be sorted!
        qs = np.sort(qs)  # makes sure it is increasingly sorted!
        index = np.digitize(x, qs)  # convert to indices
        dcount = np.bincount(index[miss], minlength=n)  # only good
        for ibin in np.arange(n):
            if dcount[ibin] > counts:
                out[ibin] = func(x[index == ibin])
        return dcount, out
    return np.zeros(n), out