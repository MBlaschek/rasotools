# -*- coding: utf-8 -*-
from ._helpers import *
from . import map
from . import profile
from . import time


def fig_horizont(n=2, ratios=(1, 4)):
    import matplotlib.pyplot as plt
    return plt.subplots(n, 1, sharex='col', gridspec_kw={'height_ratios': ratios})


def fig_vertical(n=2, ratios=(1, 3)):
    import matplotlib.pyplot as plt
    return plt.subplots(1, n, sharey='row', gridspec_kw={'width_ratios': ratios})

