# -*- coding: utf-8 -*-

__all__ = ['location_change']


def location_change(data=None, lon='lon', lat='lat', ilon=None, ilat=None, **kwargs):
    import numpy as np
    from xarray import Dataset, DataArray, full_like
    from ..fun import distance, message

    # count occurence of each coordinate pair
    # use the most common (also the most recent?)
    # to estimate distance from,
    # era-interim has a distance of about 80km so only if larger it would make sense to split?

    if data is not None:
        if not isinstance(data, Dataset):
            raise ValueError('requires a Dataset', type(data))

        if lon not in data.data_vars or lat not in data.data_vars:
            raise ValueError('Longitude or Latitude not found ?', lon, lat)
        lon = data[lon]
        lat = data[lat]

    else:
        if not isinstance(lon, DataArray) and not isinstance(lat, DataArray):
            raise ValueError('requires a DataArray for lon, lat')

    dist = full_like(lon, 0, dtype=float)
    dist.name = 'distance'
    dist.attrs['units'] = 'km'
    fdistance = np.vectorize(distance)

    if ilon is None and ilat is None:
        # distance between more recent and less recent
        tmp = fdistance(lon.values[1:], lat.values[1:], lon.values[:-1], lat.values[:-1])
        tmp = np.append(tmp, tmp[-1])
        dist.values = tmp
        dist.attrs['method'] = 'Backwards'
    else:
        dist.values = fdistance(lon, lat, ilon, ilat)
        dist.attrs['method'] = 'Point(%f E, %f N)' % (ilon, ilat)

    if data is None:
        return dist
    data['distance'] = dist


def sondetype(data, var='sondetype', **kwargs):
    import numpy as np
    from xarray import DataArray, Dataset, full_like
    from ..fun import message

    if not isinstance(data, (Dataset, DataArray)):
        raise ValueError('requires a DataArray, Dataset', type(data))

    if isinstance(data, Dataset):
        if var not in data.data_vars:
            raise ValueError('SondeType not found ?', var)
        idata = data[var]
    else:
        idata = data

    stype = full_like(idata, 0, dtype=float)
    stype.name = 'stype'
    stype.values = np.append(np.where(np.diff(idata.values) !=0, 1, 0), 0)
    return stype

