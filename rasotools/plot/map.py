# -*- coding: utf-8 -*-

__all__ = ['points', 'costfunction', 'station']


def station(lon, lat, label=None, marker='o', markersize=20, ax=None, filename=None, **kwargs):
    # plot a map of location, if lon, lat are vector data,
    # plot a polar
    # locs = pd.DataFrame({'lon':lon,'lat':lat})
    # nfo = locs.groupby(by=['lon','lat']).size().reset_index().rename(columns={0:'counts'})
    #          lon    lat      counts
    # 0  16.35  48.23    376
    # 1  16.36  48.25   3915
    # 2  16.37  48.23   9602
    # 3  16.37  48.25  39373
    # -> index gibt die farbe, größe durch counts
    # zuerst die größeren plotten
    # markersize > dem größten und die anderen kleiner, aber nie unter 1
    pass


def points(lon, lat, labels=None, values=None, markersize=80, ax=None, ocean=True, land=True, coastlines=True,
           grid=True, posneg=False, extent=None, lloffset=0.2, showcost=False, clabel=None, cbars={}, **kwargs):
    import numpy as np
    import cartopy as cpy
    import matplotlib.pyplot as plt
    from ._helpers import cost

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.size != lat.size:
        raise ValueError("Lon and Lat need same size")

    if values is not None:
        values = np.asarray(values)
        if lon.size != lat.size or lon.size != values.size:
            raise ValueError("Lon, Lat and Values need same size", lon.size, lat.size, values.size)

    projection = kwargs.get('projection', cpy.crs.PlateCarree())
    if ax is None:
        ax = plt.axes(projection=projection)

    if ocean:
        ax.add_feature(cpy.feature.OCEAN, zorder=0)

    if land:
        ax.add_feature(cpy.feature.LAND, zorder=0)

    if coastlines:
        ax.coastlines()

    if labels is not None:
        labels = np.asarray(labels)

    if values is None:
        ax.scatter(lon, lat, s=markersize, c=kwargs.get('color', 'r'), transform=projection, zorder=10,
                   edgecolor='k')  # ontop
    else:
        if posneg:
            kwargs['marker'] = np.where(values < 0, 'd', 'o')

        cs = ax.scatter(lon, lat, s=markersize, c=values, transform=projection, zorder=10,
                        cmap=kwargs.get('cmap', None), edgecolor='k', marker=kwargs.get('marker', 'o'))
        cb = plt.colorbar(cs, ax=ax, **cbars)
        if clabel is not None:
            cb.set_label(clabel)
        if showcost:
            tcost = cost(lon, lat, values)

        if np.isfinite(values).sum() != np.size(values):
            itx = ~np.isfinite(values)
            ax.scatter(lon[itx], lat[itx], s=markersize, marker='s', c='w', transform=projection, zorder=10,
                       edgecolor='k')

    if labels is not None:
        for i, j, l in zip(lon, lat, labels):
            # bbox=dict(facecolor='white', alpha=0.40, edgecolor='none'),
            ax.text(i + lloffset, j, str(l), horizontalalignment='left', verticalalignment='top',
                    transform=projection, fontsize=kwargs.get('fontsize', 8), zorder=12)

    if grid:
        try:
            gl = ax.gridlines(draw_labels=True, xlocs=kwargs.get('xlocs', None), ylocs=kwargs.get('ylocs', None),
                              linewidth=0.5, linestyle='--', color='k')
            gl.xformatter = cpy.mpl.gridliner.LONGITUDE_FORMATTER
            gl.yformatter = cpy.mpl.gridliner.LATITUDE_FORMATTER
            gl.xlabels_top = False
            gl.ylabels_right = False
        except:
            ax.gridlines(draw_labels=False)

    if values is not None:
        nn = np.sum(np.isfinite(values))
        title = '(# %d / %d)' % (nn, np.size(values))
        # COST Summary
        if showcost:
            tscost = np.nansum(tcost) / np.sum(np.isfinite(values))
            title += ' Cost: %5.2f' % tscost
    else:
        title = 'Stations # %d' % np.size(lon)

    ax.set_title(kwargs.get('title', '') + title)

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs.get('xlabel'))

    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs.get('ylabel'))

    if extent is not None:
        ax.set_extent(extent, crs=projection)

    return ax


def costfunction(lon, lat, values, **kwargs):
    import numpy as np
    from ._helpers import cost
    # plot cost values instead
    tcost = cost(lon, lat, values)
    values = np.where(np.isfinite(values), tcost, np.nan)  # Plot Cost Values instead of real values
    return points(lon, lat, values=values, **kwargs)


def values_zonal_meridional(lon, lat, values, zonal=True, ax=None, label=None, lat_bands=10, lon_bands=10, func=None,
                            fkwargs={}, **kwargs):
    import numpy as np
    import cartopy as cpy
    import matplotlib.pyplot as plt

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    values = np.asarray(values)

    if lon.size != lat.size or lon.size != values.size:
        raise ValueError("Lon, Lat and Values need same size", lon.size, lat.size, values.size)

    projection = kwargs.get('projection', cpy.crs.PlateCarree())
    if ax is None:
        ax = plt.axes(projection=projection)

    # transform lon to -180, 180
    nx = int(360 / lon_bands)
    ny = int(180 / lat_bands)
    lon_bins = np.arange(-180, 181, lon_bands)
    lat_bins = np.arange(-90, 91, lat_bands)
    grid = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            logic = (lon >= lon_bins[i] & lon < lon_bins[i+1]) & (lat >= lat_bins[j] & lat < lat_bins[j+1])
            grid[i, j] = func(values[logic], **fkwargs)
    # average zonally

    # plot meridionally ?
    return ax
