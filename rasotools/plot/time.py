# -*- coding: utf-8 -*-


def threshold(data, dim='date', lev=None, thres=50, colorlevels=None, legend=True, logy=False,
              yticklabels=None, ax=None, **kwargs):
    """

    Args:
        data:
        dim:
        lev:
        thres:
        colorlevels:
        legend:
        logy:
        yticklabels:
        ax:
        **kwargs:

    Returns:

    """
    from xarray import DataArray
    from ._helpers import line, contour, set_labels, get_info, plot_levels as pl, plot_arange as pa

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if lev is not None and lev not in data.dims:
        raise ValueError('Requires a level dimension', lev)

    if colorlevels is not None:
        if isinstance(colorlevels, str):
            colorlevels = eval(colorlevels)
    else:
        colorlevels = [thres, 2 * thres, 4 * thres, 10 * thres, 20 * thres]

    if lev is None:
        set_labels(kwargs, xlabel=get_info(data[dim]),
                   title=get_info(data), ylabel='above')

        return line(data[dim].values, data.values >= thres, **kwargs)
    else:
        set_labels(kwargs, extend='max', xlabel=get_info(data[dim]), ylabel=get_info(data[lev]),
                   title=get_info(data), clabel='above')

        return contour(ax, data[dim].values, data[lev].values, data.values, logy=logy, colorlevels=colorlevels,
                       yticklabels=yticklabels, legend=legend, **kwargs)


def summary(data, dim='date', thres=None, ax=None, **kwargs):
    """

    Args:
        data:
        dim:
        thres:
        ax:
        **kwargs:

    Returns:

    """
    import numpy as np
    from xarray import DataArray
    from ._helpers import line, set_labels, get_info

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if data.ndim > 1:
        idims = list(data.dims[:])
        idims.remove(dim)
        set_labels(kwargs, xlabel=get_info(data[dim]),
                   title=get_info(data), ylabel='Sum (' + ",".join([get_info(data[i]) for i in idims]) + ')')

        ax = line(data[dim].values, data.sum(idims).values, ax=ax, **kwargs)
        ax.set_ylim(0, 5000)
        if thres is not None:
            ay = ax.twinx()
            ay = line(data[dim].values, (data >= thres).sum(idims).values, ax=ay, color=kwargs.pop('color', 'r'))
            ay.set_yticks(np.linspace(ay.get_yticks()[0], ay.get_yticks()[-1], len(ax.get_yticks())))
            return ax, ay
    else:
        set_labels(kwargs, xlabel=get_info(data[dim]),
                   title=get_info(data), ylabel=get_info(data))

        ax = line(data[dim].values, data.values, ax=ax, **kwargs)
    return ax


def var(data, dim='date', lev=None, colorlevels=None, logy=False, yticklabels=None, legend=True,
        ax=None, **kwargs):
    """

    Args:
        data:
        dim:
        lev:
        colorlevels:
        logy:
        yticklabels:
        legend:
        ax:
        **kwargs:

    Returns:

    """
    from xarray import DataArray
    from ._helpers import line, contour, get_info, set_labels, plot_levels as pl, plot_arange as pa

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if lev is not None and lev not in data.dims:
        raise ValueError('Requires a level dimension', lev)

    if colorlevels is not None:
        if isinstance(colorlevels, str):
            colorlevels = eval(colorlevels)

    if lev is None:
        set_labels(kwargs, xlabel=get_info(data[dim]), title=get_info(data), ylabel=get_info(data))
        return line(data[dim].values, data.values, ax=ax, **kwargs)
    else:
        set_labels(kwargs, extend='both', xlabel=get_info(data[dim]), ylabel=get_info(data[lev]),
                   title=get_info(data), clabel=get_info(data))

        return contour(ax, data[dim].values, data[lev].values, data.values, logy=logy, colorlevels=colorlevels,
                       yticklabels=yticklabels, legend=legend, **kwargs)


def breakpoints(data, dim='date', thres=2, startend=False, borders=None, filled=False, ax=None, **kwargs):
    """

    Args:
        data (DataArray): Breakpoint data
        dim (str): datetime dimension
        thres (int, float): threshold to id breaks
        startend (bool): start end thresholds
        borders (int): breakpoint borders
        ax: Axes
        **kwargs:

    Returns:
        Axes
    """
    import numpy as np
    from xarray import DataArray
    import matplotlib.pyplot as plt
    from ..bp import get_breakpoints

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    dates = data[dim].values
    indices = get_breakpoints(data, thres, dim=dim)
    e = []
    s = []
    axis = data.dims.index(dim)
    if data.ndim > 1:
        summe = data.values.sum(axis=1 if axis == 0 else 0)
    else:
        summe = data.values

    for k in indices:
        l = np.where(summe[:k][::-1] == 0)[0][0]
        m = np.where(summe[k:] == 0)[0][0]
        e += [k - l]
        s += [k + m]

    if ax is None:
        f, ax = plt.subplots()  # 1D SNHT PLOT

    j = 0
    for i, ib in zip(indices, dates[indices]):
        ax.axvline(x=ib,
                   ls=kwargs.get('ls', '-'),
                   lw=kwargs.get('lw', 1),
                   label=kwargs.get('label', None) if j == 0 else None,
                   marker=kwargs.get('marker', None),
                   alpha=kwargs.get('alpha', 1),
                   color=kwargs.get('color', 'k'))  # Line Plot
        if startend:
            ax.axvline(x=dates[s[j]], color='r', ls='--', alpha=0.5)
            ax.axvline(x=dates[e[j]], color='b', ls='--', alpha=0.5)

        elif borders is not None:
            if filled:
                ax.axvspan(dates[i - borders], dates[i + borders], color=kwargs.get('color', None), alpha=0.5)
            else:
                ax.axvline(x=dates[i - borders], color=kwargs.get('color', None), ls='--', alpha=0.5)
                ax.axvline(x=dates[i + borders], color=kwargs.get('color', None), ls='--', alpha=0.5)
        else:
            pass
        j += 1

    return ax
