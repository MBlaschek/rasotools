# -*- coding: utf-8 -*-

__all__ = ['points', 'costfunction', 'station']

# todo add grid definitions of reanalysis
# for plotting
era5_grid = {'lon': None, 'lat': None}
erai_grid = None
jra55_grid = None
cera_grid = None


def station_class(sonde, **kwargs):
    from .. import Radiosonde
    if not isinstance(sonde, Radiosonde):
        raise ValueError('Requires a Radiosonde class object')
    for iatt in sonde.attrs:
        if 'lon' in iatt:
            lon = sonde.attrs[iatt]
            if isinstance(lon, str):
                for i in lon.split(' '):
                    try:
                        lon = float(i)
                    except:
                        pass

        if 'lat' in iatt:
            lat = sonde.attrs[iatt]
            if isinstance(lat, str):
                for i in lat.split(' '):
                    try:
                        lat = float(i)
                    except:
                        pass
    print(lon, lat)
    return station(lon, lat, **kwargs)


def station(lon, lat, label=None, marker='o', markersize=20, bounds=1, ax=None, data=None, **kwargs):
    import numpy as np
    import cartopy as cpy
    import matplotlib.pyplot as plt
    from ..fun import check_kw
    # from cartopy.feature import NaturalEarthFeature
    # coast = NaturalEarthFeature(category='physical', scale='10m', facecolor='none', name='coastline')

    # have a time component? -> plot as time series
    if data is not None:
        if lon in data:
            lon = data[lon].values
        if lat in data:
            lat = data[lat].values

    else:
        lon = np.asarray(lon)
        lat = np.asarray(lat)

    itx = np.isfinite(lon) & np.isfinite(lat)
    lon = lon[itx]
    lat = lat[itx]

    if lon.size != lat.size:
        raise ValueError("Lon and Lat need same size")

    # todo add number of points plotted / median distance
    ilon = np.median(lon)
    ilat = np.median(lat)
    projection = kwargs.get('projection', cpy.crs.PlateCarree())
    if ax is None:
        ax = plt.axes(projection=projection)

    ax.set_extent((ilon - bounds, ilon + bounds, ilat - bounds, ilat + bounds), crs=projection)
    if check_kw('ocean', value=True, **kwargs):
        ax.add_feature(cpy.feature.OCEAN.with_scale('10m'), zorder=0)
    if check_kw('land', value=True, **kwargs):
        ax.add_feature(cpy.feature.LAND.with_scale('10m'), zorder=0)
    if check_kw('coastline', value=True, **kwargs):
        ax.add_feature(cpy.feature.COASTLINE.with_scale('10m'), zorder=0)
    # ax.coastlines()
    if check_kw('rivers', value=True, **kwargs):
        ax.add_feature(cpy.feature.RIVERS.with_scale('10m'), zorder=1)
    if check_kw('lakes', value=True, **kwargs):
        ax.add_feature(cpy.feature.LAKES.with_scale('10m'), zorder=1)
    if check_kw('borders', value=True, **kwargs):
        ax.add_feature(cpy.feature.BORDERS.with_scale('10m'), zorder=1)
    if check_kw('states', value=True, **kwargs):
        ax.add_feature(cpy.feature.STATES.with_scale('10m'), zorder=1)

    ax.scatter(lon, lat, s=markersize, c=kwargs.get('color', 'r'), transform=projection, zorder=10,
               edgecolor='k')  # ontop
    try:
        gl = ax.gridlines(draw_labels=True, xlocs=kwargs.get('xlocs', None), ylocs=kwargs.get('ylocs', None),
                          linewidth=0.5, linestyle='--', color='k')
        gl.xformatter = cpy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cpy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabels_top = False
        gl.ylabels_right = False
    except:
        ax.gridlines(draw_labels=False)

    ax.set_extent((ilon - bounds, ilon + bounds, ilat - bounds, ilat + bounds), crs=projection)
    left = 0.5
    bottom = 0.13
    width = 0.3
    height = 0.2
    rect = [left, bottom, width, height]
    ax2 = plt.axes(rect, projection=cpy.crs.PlateCarree())
    ax2.set_extent((ilon - 30, ilon + 30, ilat - 30, ilat + 30))
    # ax2.set_global()  #will show the whole world as context

    ax2.coastlines(resolution='110m', zorder=2)
    ax2.add_feature(cpy.feature.LAND)
    ax2.add_feature(cpy.feature.OCEAN)

    ax2.gridlines()

    lon0, lon1, lat0, lat1 = ax.get_extent()
    box_x = [lon0, lon1, lon1, lon0, lon0]
    box_y = [lat0, lat0, lat1, lat1, lat0]

    plt.plot(box_x, box_y, color='red', transform=cpy.crs.Geodetic())

    return ax


def points(lon, lat, labels=None, values=None, markersize=80, ocean=True, land=True, coastlines=True, rivers=False,
           grid=True, posneg=False, extent=None, lloffset=0.2, showcost=False, clabel=None, cbars={}, colorlevels=None,
           **kwargs):
    import numpy as np
    import cartopy as cpy
    from matplotlib.colors import BoundaryNorm
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
    ax = plt.axes(projection=projection)

    if ocean:
        ax.add_feature(cpy.feature.OCEAN, zorder=0)

    if land:
        ax.add_feature(cpy.feature.LAND, zorder=0)

    if coastlines:
        ax.coastlines()

    if rivers:
        ax.add_feature(cpy.feature.LAKES, zorder=0)
        ax.add_feature(cpy.feature.RIVERS, zorder=1)

    if labels is not None:
        labels = np.asarray(labels)

    if values is None:
        ax.scatter(lon, lat, s=markersize, c=kwargs.get('color', 'r'), transform=projection, zorder=10,
                   edgecolor='k')  # ontop
    else:
        if posneg:
            kwargs['marker'] = np.where(values < 0, 'd', 'o')

        cmap = plt.get_cmap(kwargs.pop('cmap', None))
        norm = None
        if colorlevels is not None:
            norm = BoundaryNorm(colorlevels, cmap.N)

        cs = ax.scatter(lon, lat, s=markersize, c=values,
                        transform=projection,
                        zorder=10,
                        cmap=cmap,
                        edgecolor='k',
                        marker=kwargs.get('marker', 'o'),
                        norm=norm)

        cb = plt.colorbar(cs, ax=ax,
                          fraction=cbars.get('fraction', 0.01),
                          aspect=cbars.get('aspect', 50),
                          shrink=cbars.get('shrink', 0.8),
                          extend=cbars.get('extend', 'both'))

        if clabel is not None:
            cb.set_label(clabel)

        if showcost:
            tcost = cost(lon, lat, values)

        if np.isfinite(values).sum() != np.size(values):
            itx = ~np.isfinite(values)
            ax.scatter(lon[itx], lat[itx], s=markersize, marker='s', c='w', transform=projection, zorder=9,
                       edgecolor='k')

    if labels is not None:
        if not hasattr(lloffset, '__iter__'):
            lloffset = [lloffset] * len(labels)

        for i, j, l, k in zip(lon, lat, labels, lloffset):
            ax.text(i + k, j, str(l), horizontalalignment='left', verticalalignment='top',
                    transform=projection, fontsize=kwargs.get('fontsize', 8), zorder=12,
                    clip_on=True)

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

    ax.set_title(kwargs.get('title', '') +' '+ title)

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
            logic = (lon >= lon_bins[i] & lon < lon_bins[i + 1]) & (lat >= lat_bins[j] & lat < lat_bins[j + 1])
            grid[i, j] = func(values[logic], **fkwargs)
    # average zonally

    # plot meridionally ?
    return ax
