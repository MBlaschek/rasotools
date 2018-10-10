# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__all__ = ['values', 'points', 'values_cost']


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


def points(lon, lat, color=None, labels=None, symbols='o', markersize=20, ax=None, filename=None,
           title="Radiosondes", fontsize=8, **kwargs):
    """ Plot a map with station points labeled

    Args:
        lon:
        lat:
        color:
        labels:
        symbols:
        markersize:
        ax:
        filename:
        title:
        fontsize:
        **kwargs (dict): Keyword Arguments to Basemap

    Returns:

    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if lon.size != lat.size:
        raise ValueError("Lon and Lat need same size")

    if filename is not None:
        plt.ioff()

    if ax is None:
        _, ax = plt.subplots()

    x, y, m = init_map(lon, lat, ax, **kwargs)
    if labels is not None:
        labels = np.array(labels)

    # Subsettting
    ix = np.full(lon.size, True, np.bool)
    ix = ix & ((lat >= m.llcrnrlat) & (lat <= m.urcrnrlat))  # outside of map ?
    ix = ix & ((lon >= m.llcrnrlon) & (lon <= m.urcrnrlon))

    if color is None:
        color = ['r']

    m.scatter(x[ix], y[ix], s=markersize, marker=symbols, c=color, zorder=10)
    if labels is not None:
        for i, j, l in zip(x[ix], y[ix], labels[ix]):
            if l is not None:
                ax.text(i, j, str(l), horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                        bbox=dict(facecolor='white', alpha=0.40, edgecolor='none'), zorder=12)

    nn = len(x)
    title += ' (# %d)' % nn
    ax.set_title(title, fontsize=12)

    if filename is not None:
        plt.savefig(filename)
        plt.close(ax.get_figure())
        plt.ion()
        print(filename)


def values(lon, lat, value, label=None, bins=None, title=None, labels=None, labels_fontsize=8, filename=None,
           markersize=10, vmin=None, vmax=None, ax=None, plothist=True, posneg=False, levels=None, showcost=False,
           showmissing=True, verbose=0, **kwargs):
    import matplotlib.gridspec as gridspec

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    value = np.asarray(value)
    if lon.size != lat.size or lon.size != value.size:
        raise ValueError("Lon, Lat and Value need same size")

    if label is None:
        label = 'var'
    #
    if filename is not None:
        plt.ioff()  # non interactive plotting

    if showcost:
        tcost = cost(lon, lat, value)

    cmap = plt.get_cmap(kwargs.pop('cmap', None))

    if ax is None:
        if plothist:
            f = plt.figure()
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
            ax = [plt.subplot(gs[1]), plt.subplot(gs[0])]
            x, y, m = init_map(lon, lat, ax[0], **kwargs)
        else:
            f, ax = plt.subplots()
            x, y, m = init_map(lon, lat, ax, **kwargs)
    else:
        plothist = False
        x, y, m = init_map(lon, lat, ax, **kwargs)

    # Subsettting
    ix = np.full(value.size, True, np.bool)
    ix = ix & ((lat >= m.llcrnrlat) & (lat <= m.urcrnrlat))  # outside of map ?
    ix = ix & ((lon >= m.llcrnrlon) & (lon <= m.urcrnrlon))
    # check for vmin / vmax
    if vmin is not None:
        ix = ix & (value > vmin)
    if vmax is not None:
        ix = ix & (value < vmax)

    ix = ix & np.isfinite(value)

    if verbose > 0:
        print(np.nanmin(value), np.nanmax(value), " V: ", vmin, vmax)

    if verbose > 1:
        print(value[ix])

    if levels is not None:
        norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
        vmin = None
        vmax = None
    else:
        norm = None

    if posneg:
        # give other symbols for positve and negative
        iy = value[ix] > 0
        if np.sum(iy) > 0:
            im = m.scatter(x[ix][iy], y[ix][iy], s=markersize, marker='o', c=value[ix][iy], cmap=cmap,
                           zorder=10, vmin=vmin, vmax=vmax, norm=norm, edgecolor='k')
        if np.sum(~iy) > 0:
            im = m.scatter(x[ix][~iy], y[ix][~iy], s=markersize, marker='d', c=value[ix][~iy], cmap=cmap,
                           zorder=10, vmin=vmin, vmax=vmax, norm=norm, edgecolor='k')
    else:
        im = m.scatter(x[ix], y[ix], s=markersize, marker='o', c=value[ix], cmap=cmap, zorder=10,
                       vmin=vmin, vmax=vmax, norm=norm, edgecolor='k')

    # show missing as empty circles
    if np.sum(~ix) > 0 and showmissing:
        inp = m.scatter(x[~ix], y[~ix], s=markersize, marker='o', c='w', zorder=9, vmin=vmin, vmax=vmax, norm=norm,
                        edgecolor='k')

    cb = m.colorbar(im, 'right', size="2%", pad='2%', extend='both')
    cb.set_label(label)

    if title is None:
        title = 'Radiosonde Locations'

    nn = np.sum(np.isfinite(value))
    title += '(# %d / %d)' % (nn, np.size(value))
    # COST Summary
    if showcost:
        tscost = np.nansum(tcost) / np.sum(np.isfinite(value))
        title += ' Cost: %5.2f' % tscost

    if labels is not None:
        labels = np.asarray(labels)
        if plothist:
            axm = ax[0]
        else:
            axm = ax

        for label, xpt, ypt in zip(labels[ix], x[ix], y[ix]):
            # add a text Label (Station IDS)
            axm.text(xpt, ypt, str(label), horizontalalignment='left', verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.40, edgecolor='none'), fontsize=labels_fontsize, zorder=12)

    if plothist:
        ax[0].set_title(title, fontsize=12)
        if bins is None:
            bins = np.linspace(np.nanmin(value), np.nanmax(value), 11, endpoint=True)
        else:
            bins = bins[:]
            if bins[-1] < np.max(value[ix]):
                bins[-1] = np.max(value[ix])
            if bins[0] > np.min(value[ix]):
                bins[0] = np.min(value[ix])

        plot_bins = bins[:]
        counts, bins = np.histogram(value[ix], bins)
        centers = np.mean([plot_bins[:-1], plot_bins[1:]], axis=0)
        width = np.abs(np.diff(plot_bins))
        ax[1].bar(centers, counts, width=width, align='center')
        ax[1].set_xticks(plot_bins)
        ax[1].set_yticks(ax[1].get_yticks())
        ax[1].tick_params(axis='both', labelsize='small')
        ax[1].grid(True)

    else:
        ax.set_title(title, fontsize=12)
    #
    if filename is not None:
        plt.savefig(filename)
        if isinstance(ax, list):
            plt.close(ax[0].get_figure())
        else:
            plt.close(ax.get_figure())
        plt.ion()
        print(filename)


def values_cost(lon, lat, value, label=None, **kwargs):
    if label is None:
        label = 'var'
    # plot cost values instead
    tcost = cost(lon, lat, value)
    value = np.where(np.isfinite(value), tcost, np.nan)  # Plot Cost Values instead of real values
    label = 'Cost-Func. ' + label
    values(lon, lat, value, label=label, **kwargs)


def values_zonal_meridional(lon, lat, value, label=None, lat_bands=10, lon_bands=10, func=None, **kwargs):
    import matplotlib.gridspec as gridspec
    # maps + zonal and meridional averages in bands
    # todo values_zonal_meridional
    f = plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 6], width_ratios=[6, 1])
    ax = [plt.subplot(gs[1]), plt.subplot(gs[0])]
    x, y, m = init_map(lon, lat, ax[0], **kwargs)
    # legend bottom
    return


def cost(lon, lat, values):
    """ Estimate Cost between Points

    Parameters
    ----------
    lon         array/list      Longitudes
    lat         array/list      Latitudes
    values      array/list      Values

    Returns
    -------
    float   Cost
    """
    n = lon.shape[0]
    cost = np.zeros((n))
    for i in range(n):
        # Distance of all points * difference of values
        #
        cost[i] = np.nansum((distance(lon[i], lat[i], lat, lon) * (values[i] - values)) ** 2)

    return cost  # np.nansum(cost)/np.sum(np.isfinite(values))


def distance(ilon, ilat, lats, lons):
    """ Calculate Distance between one point and others

    Parameters
    ----------
    ilon
    ilat
    lats
    lons

    Returns
    -------
    array   Distances
    """
    ix = np.cos(ilat * np.pi / 180.) * np.cos(ilon * np.pi / 180.)
    iy = np.cos(ilat * np.pi / 180.) * np.sin(ilon * np.pi / 180.)
    iz = np.sin(ilat * np.pi / 180.)
    x = np.cos(lats * np.pi / 180.) * np.cos(lons * np.pi / 180.)
    y = np.cos(lats * np.pi / 180.) * np.sin(lons * np.pi / 180.)
    z = np.sin(lats * np.pi / 180.)
    dists = ix * x + iy * y + iz * z
    return np.arccos(dists * 0.999999)


def init_map(lon, lat, ax, maplimits=None, drawcountries=False, drawrelief=False, drawrivers=False, **kwargs):
    from mpl_toolkits.basemap import Basemap
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    # todo add option for center=10, 10 degrees around

    if maplimits is None:
        llcrnrlat = -90
        urcrnrlat = 90
        llcrnrlon = -180
        urcrnrlon = 180
        dlon = 30.
        dlat = 15.
    else:
        llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon = maplimits
        nlat = np.trunc(np.log10((urcrnrlat - llcrnrlat))) * 5.
        if nlat < 1:
            nlat = 5.  # minimum of 5
        dlat = np.round((urcrnrlat - llcrnrlat) / nlat, 2)
        if dlat <= 0:
            dlat = 0.5  # minium 0.5 degrees
        nlon = np.trunc(np.log10((urcrnrlon - llcrnrlon))) * 5.
        if nlon < 1:
            nlon = 5.
        dlon = np.round((urcrnrlon - llcrnrlon) / nlon, 2)
        if dlon <= 0:
            dlon = 0.5

    llcrnrlat = kwargs.pop('llcrnrlat', llcrnrlat)
    urcrnrlat = kwargs.pop('urcrnrlat', urcrnrlat)
    llcrnrlon = kwargs.pop('llcrnrlon', llcrnrlon)
    urcrnrlon = kwargs.pop('urcrnrlon', urcrnrlon)
    # Meridians and Parallels for Maps
    dlon = kwargs.pop('meridians', dlon)
    dlat = kwargs.pop('parallels', dlat)

    if llcrnrlat == urcrnrlat:
        urcrnrlat += 0.5
        llcrnrlat -= 0.5

    if llcrnrlon == urcrnrlon:
        urcrnrlon += 0.5
        llcrnrlon -= 0.5

    lon_0 = kwargs.pop('lon_0', np.mean([llcrnrlon, urcrnrlon]))
    print(llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, lon_0, dlon, dlat)
    resolution = kwargs.pop('resolution', 'l')

    if drawrelief:
        resolution = 'l'

    area_thresh = kwargs.pop('area_thresh', 10000)

    if kwargs.get('projection', None) is not None:
        m = Basemap(ax=ax, resolution=resolution, area_thresh=area_thresh, **kwargs)
    else:
        m = Basemap(ax=ax,
                    llcrnrlat=llcrnrlat,
                    urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon,
                    resolution=resolution,
                    area_thresh=area_thresh,
                    **kwargs)
    #
    if drawrelief:
        m.shadedrelief()

    x, y = m(lon, lat)
    x = np.asarray(x)
    y = np.asarray(y)
    water_color = 'white'  # [.47, .60, .81] # '#99ffff'
    m.drawmapboundary(fill_color=water_color)  # ? ocean
    if not drawrelief:
        m.fillcontinents(color='#d3d3d3', lake_color=water_color)  # land, lakes '#cc9966'

    if drawcountries:
        m.drawcountries()

    if drawrivers:
        m.drawrivers(color='blue')

    m.drawcoastlines()
    m.drawparallels(np.arange(llcrnrlat, urcrnrlat, dlat), labels=[1, 0, 0, 0], fontsize=kwargs.get('fontsize', 10))
    m.drawmeridians(np.arange(llcrnrlon, urcrnrlon, dlon), labels=[0, 0, 0, 1], fontsize=kwargs.get('fontsize', 10))
    return x, y, m


def estimate_limits(data, is_lat=False, interval=10):
    """ Lon-Lat Limits for Maps

    Parameters
    ----------
    data
    is_lat
    interval

    Returns
    -------

    """
    if is_lat:
        imin = -90
        imax = 90
    else:
        imin = -180
        imax = 180

    if not is_lat:
        data_180 = False
        data = np.where(data < 0, data + 360., data)  # fix -180; 180 -> 0-360
        jmin = np.min(data)
        jmax = np.max(data)
        if (jmax - jmin) > 350:
            # Europe
            data = np.where(data > 180, data - 360., data)  # fix -180; 180 -> 0-360
            jmin = np.min(data)
            jmax = np.max(data)
            data_180 = True
    else:
        jmin = np.min(data)
        jmax = np.max(data)

    mid = np.nanmedian(data)

    add = 0.1 * np.arange(0, imax, interval / 2)[
        np.digitize(jmax - jmin, bins=np.arange(interval / 2, imax, interval / 2))]
    jmin -= add
    jmax += add

    if np.any(data < jmin):
        jmin = np.min(data) - add

    if np.any(data > jmax):
        jmax = np.max(data) + add

    if is_lat:
        if jmin < imin:
            jmin = imin
        if jmax > imax:
            jmax = imax
    else:
        print(jmin, jmax, data_180)
        if data_180:
            if (jmax - jmin) > 270:
                jmin = -180  # Global ?
                jmax = 180
                mid = 0
        else:
            if jmin < 10 and jmax > 350:
                jmin = -180  # global ?
                jmax = 180
                mid = 0
                data = np.where(data > 180, data - 360., data)  # reset again to -180 180
    return jmin, jmax, mid, data
