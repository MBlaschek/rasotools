# -*- coding: utf-8 -*-
from ._helpers import *
from . import map
from . import profile
from . import time


def fig_horizont(n=2, ratios=(1, 4), figsize=None, **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(n, 1, sharex='col', gridspec_kw={'height_ratios': ratios}, figsize=figsize)


def fig_vertical(n=2, ratios=(1, 3), figsize=None, **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(1, n, sharey='row', gridspec_kw={'width_ratios': ratios}, figsize=figsize)


def snht(data, snht_var, break_var, dim='date', lev='pres', thres=50, **kwargs):
    f, ax = fig_horizont(**kwargs)
    _, cs = time.threshold(data[snht_var], lev=lev, ax=ax[1], logy=False, legend=False, **kwargs)
    ax[1].set_title('')
    time.breakpoints(data[break_var], ax=ax[1], color='k')
    time.summary(data[snht_var], dim=dim, thres=thres, ax=ax[0], xlabel='', ylabel='Sum SNHT', **kwargs)
    time.breakpoints(data[break_var], ax=ax[0], color='k')
    f.get_axes()[2].set_ylabel('Sum sign. Levs')
    f.subplots_adjust(right=0.9)
    cax = f.add_axes([0.91, 0.15, 0.015, 0.5])
    f.colorbar(cs, cax=cax)
    f.subplots_adjust(hspace=0.05)
    return f, ax
