# -*- coding: utf-8 -*-

__all__ = ['breakpoints_histograms']


def breakpoints_histograms(data, name, adjname, breakname, dim='time', levdim='plev', level=None, bins=None, borders=0,
                           nmax=1470,
                           other_var=None, **kwargs):
    """
    Calculate breakpoint stats from raw and adjusted
    make histograms

    Returns:

    """
    import numpy as np
    from xarray import Dataset
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from ..bp import breakpoint_statistics, get_breakpoints
    from .time import var as plotvar, breakpoints as plotbreaks
    from ..fun import update_kw

    if not isinstance(data, Dataset):
        raise ValueError("Requires a Dataset class object", type(data))

    if dim not in data.coords:
        raise ValueError("Requires a datetime dimension", data.coords)

    if name not in data.data_vars:
        raise ValueError("Variable breakname not present", name, data.data_vars)

    if adjname not in data.data_vars:
        raise ValueError("Variable breakname not present", adjname, data.data_vars)

    if breakname not in data.data_vars:
        raise ValueError("Variable breakname not present", breakname, data.data_vars)

    variables = [name, adjname, breakname]
    nplotlevels = 2  # timeseries, hists
    if other_var is not None:
        if other_var not in data.data_vars:
            raise ValueError("Variable breakname not present", breakname, data.data_vars)
        variables += [other_var]
        nplotlevels = 3  # timeseries, timeseries, hists

    if bins is None:
        bins = np.arange(0, 60)

    if level is not None:
        data = data[variables].sel(**{levdim: level})

    if len(data[name].shape) > 1:
        raise RuntimeError("Too many dimensions for plotting, set lev or select before", data[name].shape)

    #
    # Private histogram function
    #
    myhist = lambda x: np.histogram(x, bins=bins)[0]
    #
    # calculate histograms before and after breakpoints
    #
    ibreaks = get_breakpoints(data[breakname], dim=dim, **kwargs)
    if len(ibreaks) == 0:
        return

    xyhists = breakpoint_statistics(data[variables], breakname, dim=dim, inbetween=False, verbose=1,
                                    borders=borders, nmax=nmax,
                                    ffunc=myhist, add_dim='bins', **kwargs)
    xyhists = xyhists.assign_coords({'bins': bins[:-1]})  # np.mean([bins[:-1], bins[1:]], axis=0)})
    nbreaks = len(ibreaks)
    f = plt.figure(figsize=kwargs.pop('figsize', (12, nplotlevels * 3)))
    gs = gridspec.GridSpec(nplotlevels, nbreaks, height_ratios=list(np.ones(nplotlevels - 1)) + [2],
                           hspace=0.4)  # zwischen oben und unten
    #
    # timeseries
    #
    ax = f.add_subplot(gs[0, :])
    ax = plotvar(data[name].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=ax, **kwargs)
    ax = plotvar(data[adjname].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=ax, **kwargs)
    ax = plotbreaks(data[breakname], dim=dim, ax=ax, **update_kw('lw', 3, **kwargs))
    ax.set_title("%s / %s Breakpoint Histograms %s" % (name, adjname, data[name]._title_for_slice()))
    ax.set_xlabel('')
    ax.set_ylabel("[%s]" % (xyhists[name].units))
    j = 1
    ax = [ax]
    if other_var is not None:
        az = f.add_subplot(gs[j, :])
        az = plotvar(data[other_var].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=az, **kwargs)
        az.set_title(other_var)
        az.set_xlabel('')
        az.set_ylabel("[%s]" % (data[other_var].units))
        ax += [az]
        j = 2
    #
    # Histograms
    #

    ibreaks = list(data[dim].values[ibreaks].astype('M8[D]').astype('str'))
    for i, ibreak in enumerate(ibreaks):
        # Plot
        ay = f.add_subplot(gs[j, i], sharey=ax[j] if i > 0 else None)
        # +++++ A | B ++++++
        # B adjusted
        xyhists[adjname].sel(region='B%s' % ibreak).plot.step(ax=ay, label='B ' + adjname, c='lightgray')
        # A unadjusted
        xyhists[name].sel(region='A%s' % ibreak).plot.step(ax=ay, label=name)
        # A adjusted
        xyhists[adjname].sel(region='A%s' % ibreak).plot.step(ax=ay, label=adjname)
        # Other var
        if other_var is not None:
            xyhists[other_var].sel(region='A%s' % ibreak).plot.step(ax=ay, label=other_var)

        ay.set_title("%s #%d" % (ibreak, xyhists[name].sel(region='A%s' % ibreak).sum()))
        ay.set_xlabel("%s [%s]" % (xyhists[name].standard_name, xyhists[name].units))
        if i == 0:
            ay.set_ylabel('#')
            ay.legend()
        else:
            ay.set_ylabel('')
        ay.grid()
        ax += [ay]
    return f, ax
