# -*- coding: utf-8 -*-
import numpy as np

from ..fun import nanfunc

__all__ = ['mean', 'percentile']


def mean(sample1, sample2, axis=0, sample_size=130, borders=0, max_sample=1460, ratio=True,
         median=False, **kwargs):
    """ Adjustment method using mean differences or ratios

    ratio=False
    data[sampleout]  + (MEAN(data[sample1]) - MEAN(data[sample2]))

    ratio=True
    data[sampleout]  * (MEAN(data[sample1]) / MEAN(data[sample2]))

    Args:
        sample1 (np.ndarray): reference
        sample2 (np.ndarray): sample
        axis (int): date axis
        sample_size (int): minimum sample size
        ratio (bool): use ratio or difference?
        median (bool): use median instead of mean?
        borders (int): around breakpoint
        max_sample (int): maximum sample size

    Returns:
        np.ndarray : mean adjusted data
    """
    # minimum sample size, maximum sample size
    if median:
        s1 = nanfunc(sample1,
                     axis=axis,
                     n=sample_size,
                     nmax=max_sample,
                     func=np.nanmedian,
                     borders=borders)
        s2 = nanfunc(sample2,
                     axis=axis,
                     n=sample_size,
                     nmax=max_sample,
                     func=np.nanmedian,
                     borders=borders,
                     flip=True)
    else:
        s1 = nanfunc(sample1,
                     axis=axis,
                     n=sample_size,
                     nmax=max_sample,
                     func=np.nanmean,
                     borders=borders)
        s2 = nanfunc(sample2,
                     axis=axis,
                     n=sample_size,
                     nmax=max_sample,
                     func=np.nanmean,
                     borders=borders,
                     flip=True)

    if ratio:
        # Todo factor amplifies extreme values
        dep = s1 / s2
        dep = np.where(np.isfinite(dep), dep, 1.)  # replace NaN with 1
        sample2 *= dep
    else:
        dep = s1 - s2
        sample2 += dep
    return sample2


def meanvar(sample1, sample2, axis=0, sample_size=130, borders=0, max_sample=1460, **kwargs):
    """ Adjustment method using mean differences or ratios

    data[sampleout]  + (MEAN(data[sample1]) - MEAN(data[sample2]))

    Args:
        sample1 (np.ndarray): reference
        sample2 (np.ndarray): sample
        axis (int): date axis
        sample_size (int): minimum sample size
        borders (int): around breakpoint
        max_sample (int): maximum sample size

    Returns:
        np.ndarray : mean adjusted data
    """
    s1 = nanfunc(sample1,
                 axis=axis,
                 n=sample_size,
                 nmax=max_sample,
                 func=np.nanmean,
                 borders=borders)
    s2 = nanfunc(sample2,
                 axis=axis,
                 n=sample_size,
                 nmax=max_sample,
                 func=np.nanmean,
                 borders=borders,
                 flip=True)
    s1v = nanfunc(sample1,
                  axis=axis,
                  n=sample_size,
                  nmax=max_sample,
                  func=np.nanvar,
                  borders=borders)
    s2v = nanfunc(sample2,
                  axis=axis,
                  n=sample_size,
                  nmax=max_sample,
                  func=np.nanvar,
                  borders=borders,
                  flip=True)

    # MEAN
    dep = s1 - s2
    # VAR
    fac = np.divide(s1v, s2v, out=np.ones(s2v.shape), where=s2v != 0)
    sample2 += (dep * fac)
    return sample2


def percentile(sample1, sample2, percentiles, axis=0, sample_size=130, borders=0, max_sample=1460, ratio=True,
               apply=None, **kwargs):
    """ Adjustment method using percentile differences or ratios

    ratio=False
    data[sample1] + ( percentiles(data[sample1]) - percentiles(data[sample2]) )

    ratio=True
    data[sample1] * ( percentiles(data[sample1]) / percentiles(data[sample2]) )

    Args:
        sample1 (np.ndarray): reference
        sample2 (np.ndarray): sample
        percentiles (list): percentiles to use
        axis (int): date axis
        sample_size (int): minimum sample size
        ratio (bool): use ratio or difference?
        borders (int): around breakpoint
        max_sample (int): maximum sample size

    Returns:
        np.ndarray : percentile adjusted data
    """
    # Add 0 and 100, and remove them
    percentiles = np.unique(np.concatenate([[0], percentiles, [100]]))
    percentiles = percentiles[1:-1]  # remove 0 and 100

    # Sample sizes are enough?
    # nsample1 = np.isfinite(data[sample1]).sum(axis=axis) > sample_size
    # nsample2 = np.isfinite(data[sample2]).sum(axis=axis) > sample_size

    # Percentiles of the samples
    # s1 = np.nanpercentile(data[sample1], percentiles, axis=axis)
    # s2 = np.nanpercentile(data[sample2], percentiles, axis=axis)
    s1 = nanfunc(sample1,
                 axis=axis,
                 n=sample_size,
                 nmax=max_sample,
                 func=np.nanpercentile,
                 borders=borders,
                 fargs=(percentiles,))

    s2 = nanfunc(sample2,
                 axis=axis,
                 n=sample_size,
                 nmax=max_sample,
                 func=np.nanpercentile,
                 borders=borders,
                 fargs=(percentiles,),
                 flip=True)
    if ratio:
        # dep = np.where(sample2 != 0., sample1 / sample2, 1.)
        dep = np.divide(s1, s2, where=(s2 != 0), out=np.full(s2.shape, 1.))
        # dep = np.where(nsample1 & nsample2, dep, 1.)  # apply sample size
        dep = np.where(np.isfinite(dep), dep, 1.)  # replace NaN
    else:
        dep = s1 - s2
        # dep = np.where(nsample1 & nsample2, dep, 0.)  # apply sample size
        dep = np.where(np.isfinite(dep), dep, 0.)

    # Interpolate adjustments to sampleout shape and data
    if apply is None:
        dep = apply_percentile_adjustments(sample2, s2, dep, axis=axis)

        if ratio:
            dep = np.where(np.isfinite(dep), dep, 1.)
            sample2 *= dep
        else:
            dep = np.where(np.isfinite(dep), dep, 0.)
            sample2 += dep

        return sample2

    dep = apply_percentile_adjustments(apply, s2, dep, axis=axis)

    if ratio:
        dep = np.where(np.isfinite(dep), dep, 1.)
        apply *= dep
    else:
        dep = np.where(np.isfinite(dep), dep, 0.)
        apply += dep

    return apply


#
# Helper functions
#


def apply_percentile_adjustments(data, percentiles, adjustment, axis=0):
    """ Helper Function for applying percentile adjustments

    Args:
        data (np.ndarray): data
        percentiles (np.ndarray): percentiles, points of adjustments
        adjustment (np.ndarray): adjustments to be interpolated
        axis (int): axis of datetime

    Returns:
        np.ndarray : interpolated adjustment, same shape as data
    """
    in_dims = list(range(data.ndim))
    # last dim == axis, Last dim should be time/date
    data = np.transpose(data, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    percentiles = np.transpose(percentiles, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    adjustment = np.transpose(adjustment, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    adjusts = np.zeros(data.shape)
    # Indices for iteration + expand
    inds = np.ndindex(data.shape[:-1])  # iterate all dimensions but last
    inds = (ind + (Ellipsis,) for ind in inds)  # add last as ':' == Ellipsis == all
    for ind in inds:
        # INTERP -> Xnew, Xpoints, Fpoints
        adjusts[ind] = np.interp(data[ind], percentiles[ind], adjustment[ind], left=np.nan, right=np.nan)

    # Transform back to original shape
    return np.transpose(adjusts, in_dims[:axis] + in_dims[axis + 1:] + [axis])
