# -*- coding: utf-8 -*-
from ._helpers import *
from . import map
from . import profile
from .time import *
from .analysis import *


def init_fig_vertical(n=2, ratios=(1, 4), figsize=None, sharex='row', **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(n, 1, sharex=sharex, gridspec_kw={'height_ratios': ratios}, figsize=figsize, **kwargs)


def init_fig_horizontal(n=2, ratios=(1, 3), figsize=None, sharey='col', **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(1, n, sharey=sharey, gridspec_kw={'width_ratios': ratios}, figsize=figsize, **kwargs)

