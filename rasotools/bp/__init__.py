"""
Breakpoint Module of RASO

Structure
- detect
--- test (x, window, missing)
- adj
--- mean (data, breaks, sample_size=730, borders=180, bounded=None, recent=False, ratio=True, verbose=0)
--- quantile (data, breaks, quantilen=None, sample_size=730, borders=180, bounded=None, recent=False, ratio=True, verbose=0)
--- quantile_reference (xdata, ydata, breaks, quantilen=None, sample_size=730, borders=180, bounded=None, recent=False, ratio=True, verbose=0)
- dep
--- mean (x, sample1, sample2, sample_size)
--- mean_ratio (x, sample1, sample2, sample_size)
--- quantile (x, sample1, sample2, meinequantilen, sample_size, sample3=None, return_mean=False, linear=True, verbose=0)
--- quantile_ratio (x, sample1, sample2, meinequantilen, sample_size, sample3=None, verbose=0)
--- quantile_reference (x, y, sample1, meinequantilen, sample_size, verbose=0)
--- quantile_era_ratio (x, y, sample1, meinequantilen, sample_size, verbose=0)
"""

from . import det
from . import dep
from . import adj
from . import meta
# from quantile_count import quantile_count
# from quantiles_at_breakpoint import quantiles_at_breakpoint

from .__wrapper__ import *
# from .perz import *
