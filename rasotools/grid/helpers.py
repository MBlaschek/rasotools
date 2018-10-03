import numpy as np


def _get_lonlat(dims):
    "Look for lon lat -> return keys"
    jlon = None
    jlat = None
    for i in dims:
        if i.lower() in ['longitude', 'long', 'lon']:
            jlon = i

        if i.lower() in ['latitude', 'lati', 'lat']:
            jlat = i

    return jlon, jlat


def _get_pos(ilon, ilat, lon, lat):
    # should be also valid for points in the range of -180 to 180
    if np.argmin([np.min(np.abs(lon - ilon)), np.min(np.abs(lon + 360. - ilon))]) == 0:
        return np.argmin(np.abs(lon - ilon)), np.argmin(np.abs(lat - ilat))
    else:
        return np.argmin(np.abs(lon + 360. - ilon)), np.argmin(np.abs(lat - ilat))


def _rectangle(ilon, ilat, lons, lats, n=2):
    import itertools
    ix, iy = _get_pos(ilon, ilat, lons, lats)
    if ix < 2 or ix > lons.size - 2:
        # wrap around for lon
        ilon = ilon if ilon <= 180. else ilon - 360.
        jx = np.abs(np.where(lons > 180., lons - 360., lons) - ilon).argsort()[slice(None, n)]
    else:
        jx = np.abs(lons - ilon).argsort()[slice(None, n)]
    jy = np.abs(lats - ilat).argsort()[slice(None, n)]
    jy = np.unique(np.where(jy > lats.size, lats.size, jy))  # upper limit
    return list(itertools.product(jx, jy)), list(itertools.product(lons[jx], lats[jy]))


def _bilinear_weights(x, y, p, lons, lats):
    """Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.
    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    p = sorted(p)  # order points by x, then by y
    #
    (x1, y1), (_x1, y2), (x2, _y1), (_x2, _y2) = p
    # (lon1,lat1), (lon2,lat1), (lon1,lat2), (lon2,lat2)
    w = np.array([(lons[x2] - x) * (lats[y2] - y),
                  (x - lons[x1]) * (lats[y2] - y),
                  (lons[x2] - x) * (y - lats[y1]),
                  (x - lons[x1]) * (y - lats[y1])]) / ((lons[x2] - lons[x1]) * (lats[y2] - lats[y1]) + 0.0)
    if (w < 0).any():
        lons = np.where(lons > 180., lons - 360., lons)
        x = x if x <= 180. else x - 360.
        w = np.array([(lons[x2] - x) * (lats[y2] - y),
                      (x - lons[x1]) * (lats[y2] - y),
                      (lons[x2] - x) * (y - lats[y1]),
                      (x - lons[x1]) * (y - lats[y1])]) / ((lons[x2] - lons[x1]) * (lats[y2] - lats[y1]) + 0.0)

    # Make sure it is 1
    return np.array([x1, x2, x1, x2]), np.array([y1, y1, y2, y2]), w / w.sum()


def _distance_weights(lons, lats, ilon, ilat):
    ix, iy = _get_pos(ilon, ilat, lons, lats)
    ny = len(lats)
    nx = len(lons)
    dx = np.abs(lons - ilon)
    dy = np.abs(lats - ilat)
    if ix + 1 < nx:
        if dx[ix - 1] < dx[ix + 1]:
            idx = [ix - 1, ix]
        else:
            idx = [ix, ix + 1]
    else:
        if dx[ix - 1] < dx[0]:
            idx = [ix - 1, ix]
        else:
            idx = [ix, 0]

    if iy + 1 < ny:
        if dy[iy - 1] < dy[iy + 1]:
            idy = [iy - 1, iy]
            if iy - 1 < 0:
                idy = [iy, iy + 1]
        else:
            idy = [iy, iy + 1]
            if iy + 1 >= ny:
                idy = [iy - 1, iy]
    else:
        idy = [iy - 1, iy]

    dist = np.full((2, 2), 1, dtype=np.float)
    for k, i in enumerate(idx):
        for l, j in enumerate(idy):
            dist[l, k] = 1 / _distance(ilon, ilat, lons[i], lats[j])
    return idx, idy, (dist / np.sum(dist))


def _distance(lon, lat, lon0, lat0):
    """
    Calculates the distance between a point and an array of points

    Parameters
    ----------
    lon     Longitude Vector
    lat     Latitude Vector
    lon0    Longitude of Position
    lat0    Latitude of Position

    Returns
    -------
    numpy.array
    """
    lonvar, latvar = np.meshgrid(lon, lat)
    rad_factor = np.pi / 180.0  # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar * rad_factor
    lonvals = lonvar * rad_factor
    # ny,nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Compute numpy arrays for all values, no loops
    clat, clon = np.cos(latvals), np.cos(lonvals)
    slat, slon = np.sin(latvals), np.sin(lonvals)
    delX = np.cos(lat0_rad) * np.cos(lon0_rad) - clat * clon
    delY = np.cos(lat0_rad) * np.sin(lon0_rad) - clat * slon
    delZ = np.sin(lat0_rad) - slat;
    dist_sq = delX ** 2 + delY ** 2 + delZ ** 2  # Distance
    return np.squeeze(dist_sq)




#
#  DEBUGGING
#

def _plot_weights(idx, idy, weights, data, lons, lats, jlon, jlat, lonlat=True, dx=0, dy=0):
    import matplotlib
    import matplotlib.pyplot as plt

    if not lonlat:
        data = data.T

    ix, iy = idx[0], idy[0]
    tmp = data[ix - 2:ix + 4, iy - 2:iy + 4]

    norm = matplotlib.colors.Normalize(vmin=tmp.min(), vmax=tmp.max())
    plt.figure()
    plt.pcolormesh(lons[ix - 2:ix + 4] + dx, lats[iy - 2:iy + 4] + dy, tmp.T, norm=norm)

    for i, j, k in zip(idx, idy, weights):
        plt.scatter(lons[i], lats[j], c=data[i, j], norm=norm, edgecolors='k')
        plt.text(lons[i], lats[j], "%6.2f\n(%6.2f %6.2f)\n%6.5f" % (data[i, j], lons[i], lats[j], k),
                 horizontalalignment='right')

    plt.plot(jlon, jlat, 'o', c='k')
    plt.colorbar()

