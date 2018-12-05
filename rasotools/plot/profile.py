# -*- coding: utf-8 -*-


def var(data, dim='pres', ax=None, logy=False, yticklabels=None, showna=True, **kwargs):
    import numpy as np
    from xarray import DataArray
    from ._helpers import line, set_labels, get_info
    from ..fun import message

    if not isinstance(data, DataArray):
        raise ValueError('Requires a DataArray', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if data.ndim > 1:
        raise ValueError('Too many dimensions', data.dims, data.shape)

    values = data.values
    levels = data[dim].values.copy()
    lev_units = data[dim].attrs.get('units', 'Pa')
    if lev_units == 'Pa':
        levels /= 100.
        message('Converting', lev_units, 'to', 'hPa', levels, **kwargs)
        lev_units = 'hPa'

    itx = np.isfinite(values)

    set_labels(kwargs, xlabel=get_info(data),
               title=get_info(data), ylabel=dim + ' [%s]' %lev_units)
    ax = line(values[itx], levels[itx], ax=ax, **kwargs)
    if np.sum(itx) != np.size(levels) and showna:
        itmp = ax.get_xlim()
        ax.plot([itmp[1]]*np.sum(~itx), levels[~itx], marker=kwargs.get('marker','x'), c='red')

    if logy:
        ax.set_yscale('log')

    if np.diff(ax.get_ylim())[0] > 0:
        ax.invert_yaxis()
    ax.set_ylim(*kwargs.get('ylim', (None, None)))  # fixed
    ax.set_xlim(*kwargs.get('xlim', (None, None)))
    ax.set_yticks(levels, minor=True)
    if yticklabels is not None:
        yticklabels = np.asarray(yticklabels)  # can not calc on list
        ax.set_yticks(yticklabels)
        ax.set_yticklabels(np.int_(yticklabels))
    else:
        ax.set_yticks(levels[::2])
        ax.set_yticklabels(np.int_(levels[::2]))
    return ax


def winds(data, u='u', v='v', dim='pres', barbs=True, ax=None, logy=False, yticklabels=None, showna=True, **kwargs):
    import numpy as np
    from xarray import Dataset
    import matplotlib.pyplot as plt
    from ._helpers import set_labels, line
    from ..fun import message

    if not isinstance(data, Dataset):
        raise ValueError('Requires a Dataset', type(data))

    if dim not in data.dims:
        raise ValueError('Requires a datetime dimension', dim)

    if u not in data.data_vars:
        raise ValueError('Requires a u-wind component', u)

    if v not in data.data_vars:
        raise ValueError('Requires a v-wind component', v)

    if data[u].ndim > 1 or data[v].ndim > 1:
        raise ValueError('Too many dimensions', data.dims, data.shape)

    uvalues = data[u].values
    vvalues = data[v].values
    levels = data[dim].values.copy()
    lev_units = data[dim].attrs.get('units', 'Pa')
    if lev_units == 'Pa':
        levels /= 100.
        message('Converting', lev_units, 'to', 'hPa', levels, **kwargs)
        lev_units = 'hPa'

    itx = np.isfinite(uvalues) & np.isfinite(vvalues)

    set_labels(kwargs, xlabel='Winds ['+data[u].attrs.get('units','m/s')+']',
               title='Winds', ylabel=dim + ' [%s]' %lev_units)

    if barbs:
        if ax is None:
            f, ax = plt.subplots()  # 1D SNHT PLOT
        speed = np.sqrt(uvalues * uvalues + vvalues * vvalues)
        ax.barbs(np.zeros_like(levels[itx]), levels[itx], uvalues[itx], vvalues[itx], speed[itx],
                 alpha=kwargs.get('alpha', 1))
    else:
        ax = line(uvalues[itx], levels[itx], label='u-wind', ax=ax, **kwargs)
        ax = line(vvalues[itx], levels[itx], label='v-wind', ax=ax, **kwargs)
        ax.legend()

    ax.grid('gray', ls='--')
    ax.set_title(kwargs.get('title'))
    ax.set_ylabel(kwargs.get('ylabel'))
    ax.set_xlabel(kwargs.get('xlabel'))

    if logy:
        ax.set_yscale('log')

    if np.diff(ax.get_ylim())[0] > 0:
        ax.invert_yaxis()

    ax.set_ylim(*kwargs.get('ylim', (None, None)))  # fixed
    ax.set_xlim(*kwargs.get('xlim', (None, None)))
    ax.set_yticks(levels, minor=True)
    if yticklabels is not None:
        yticklabels = np.asarray(yticklabels)  # can not calc on list
        ax.set_yticks(yticklabels)
        ax.set_yticklabels(np.int_(yticklabels))
    else:
        ax.set_yticks(levels[::2])
        ax.set_yticklabels(np.int_(levels[::2]))
    return ax


def filled():
    pass
