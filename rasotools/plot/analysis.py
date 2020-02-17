# -*- coding: utf-8 -*-

__all__ = ['breakpoints_histograms']


def breakpoints_histograms(data, name, adjname, breakname, dim='time', levdim='plev', level=None, bins=None, borders=0,
                           nmax=1470, other_var=None, other_hist=False, annotate=False, **kwargs):
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
        if not isinstance(other_var, list):
            other_var = [other_var]
        for iovar in other_var:
            if iovar not in data.data_vars:
                raise ValueError("Variable not present", iovar, data.data_vars)
            variables += [iovar]
        nplotlevels = 3  # timeseries, timeseries, hists
        if other_hist:
            nplotlevels = 4

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
    gs = gridspec.GridSpec(nplotlevels, nbreaks, height_ratios=kwargs.pop('height_ratios', [1]*nplotlevels),
                           hspace=0.4)  # zwischen oben und unten
    #
    # timeseries
    #
    j = 0
    ax = f.add_subplot(gs[0, :])
    ax = plotvar(data[name].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=ax, **kwargs)
    ax = plotvar(data[adjname].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=ax, **kwargs)
    ax = plotbreaks(data[breakname], dim=dim, ax=ax, ls=':', **update_kw('lw', 2, **kwargs))
    ax.set_title("%s / %s Breakpoint Histograms %s" % (name, adjname, data[name]._title_for_slice()))
    ax.set_xlabel('')
    ax.set_ylabel("[%s]" % (xyhists[name].units))
    anno_labels = 'abcdefghijklmnopqrstuvwxyz'
    if annotate:
        ax.annotate(anno_labels[j]+')',xy=(-0.06, 1), xycoords='axes fraction')
    j = 1
    ax = [ax]
    if other_var is not None:
        az = f.add_subplot(gs[j, :], sharex=ax[0])
        for iovar in other_var:
            az = plotvar(data[iovar].rolling(**{dim: 30}, min_periods=10, center=True).mean(), dim=dim, ax=az, **kwargs)
        az = plotbreaks(data[breakname], dim=dim, ax=az, ls=':', **update_kw('lw', 2, **kwargs))
        az.set_title(" / ".join(other_var))
        az.set_xlabel('')
        az.set_ylabel("[%s]" % data[other_var[0]].units)
        if annotate:
            az.annotate(anno_labels[j] + ')', xy=(-0.06, 1), xycoords='axes fraction')
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
        # ay.step('bins', 'values')
        xyhists[adjname].sel(region='B%s' % ibreak).plot.step(ax=ay, label='B ' + adjname, c='lightgray')
        # A unadjusted
        xyhists[name].sel(region='A%s' % ibreak).plot.step(ax=ay, label=name)
        # A adjusted
        xyhists[adjname].sel(region='A%s' % ibreak).plot.step(ax=ay, label=adjname)
        # Other var
        if other_var is not None:
            if other_hist:
                az = f.add_subplot(gs[j + 1, i], sharey=ax[j + 1] if i > 0 else None)
                for iovar in other_var:
                    xyhists[iovar].sel(region='A%s' % ibreak).plot.step(ax=az, label=iovar)
            else:
                for iovar in other_var:
                    xyhists[iovar].sel(region='A%s' % ibreak).plot.step(ax=ay, label=iovar)

        ay.set_title("%s #%d" % (ibreak, xyhists[name].sel(region='A%s' % ibreak).sum()))
        if other_hist:
            ay.set_xlabel('')
            az.set_title("%s #%d" % (ibreak, xyhists[name].sel(region='A%s' % ibreak).sum()))
            az.set_xlabel("%s [%s]" % (xyhists[name].standard_name, xyhists[name].units))
            if i == 0:
                az.set_ylabel('#')
                az.legend()
            else:
                az.set_ylabel('')
            az.grid()
            if annotate:
                az.annotate(anno_labels[(j+len(ibreaks))+i] + ')', xy=(-0.1, 1), xycoords='axes fraction')
            ax += [az]
        else:
            ay.set_xlabel("%s [%s]" % (xyhists[name].standard_name, xyhists[name].units))

        if i == 0:
            ay.set_ylabel('#')
            ay.legend()
        else:
            ay.set_ylabel('')
        ay.grid()
        if annotate:
            ay.annotate(anno_labels[j+i] + ')', xy=(-0.1, 1), xycoords='axes fraction')
        ax += [ay]
    return f, ax


def departures(var1, var2, data=None, dim='time', lev=None, colorlevels=None, logy=False, yticklabels=None,
               legend=True, ax=None, **kwargs):
    """

    Args:
        var1 (str, DataArray):
        var2 (str, DataArray):
        data (Dataset):
        dim (str):
        lev (str):
        colorlevels (list):
        logy (bool):
        yticklabels (list):
        legend (bool):
        ax (axis):
        **kwargs:

    Returns:

    """
    from xarray import Dataset, DataArray
    from ..met.time import correlate, statistics
    from ._helpers import line, contour, get_info, set_labels, plot_levels as pl, plot_arange as pa

    if data is not None:
        if not isinstance(data, Dataset):
            raise ValueError('Requires a Dataset', type(data))
        if var1 not in data.data_vars or var2 not in data.data_vars:
            raise ValueError("Variables not found: ", var1, var2, "<>", data.data_vars)
        var1 = data[var1]
        var2 = data[var2]

    if not isinstance(var1, DataArray):
        raise ValueError('Requires a DataArray', type(var1))

    if not isinstance(var2, DataArray):
        raise ValueError('Requires a DataArray', type(var2))

    if dim not in var1.dims or dim not in var2.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if lev is not None and lev not in var1.dims and lev not in var2.dims:
        raise ValueError('Requires a level dimension', lev)

    if colorlevels is not None:
        if isinstance(colorlevels, str):
            colorlevels = eval(colorlevels)  # plot_levels, plot_arange

    diff = var1 - var2
    diff.name = '{}_{}'.format(var1.name, var2.name)
    diff.attrs['standard_name'] = 'dep_' + var1.attrs.get('standard_name', diff.name)
    diff.attrs['units'] = var1.attrs.get('units', '1')
    cor = correlate(var1, var2, dim=dim, **kwargs).median().values
    rmse = statistics(var1, f='rmse', y=var2, dim=dim, **kwargs).median().values
    med = diff.median().values
    kwargs['title'] += " Dep({}-{}) R: {:.2f} RMSE: {:6.2f} M: {:6.2f}".format(var1.name, var2.name, cor, rmse, med)

    if lev is None:
        set_labels(kwargs, xlabel=get_info(diff[dim]), title=get_info(diff), ylabel=get_info(diff))
        return line(diff[dim].values, diff.values, ax=ax, **kwargs)
    else:
        set_labels(kwargs, extend='both', xlabel=get_info(diff[dim]), ylabel=get_info(diff[lev]),
                   title=get_info(diff), clabel=get_info(diff))
        if 'units' in diff[lev].attrs:
            units = diff[lev].attrs['units']
            if units == 'hPa':
                kwargs.update({'levfactor': 1})

        return contour(ax, diff[dim].values, diff[lev].values, diff.values, logy=logy, colorlevels=colorlevels,
                       yticklabels=yticklabels, legend=legend, **kwargs)
