# -*- coding: utf-8 -*-

__all__ = ['location_change', 'sondetype', 'metadata']


def location_change(lon, lat, dim='date', ilon=None, ilat=None, **kwargs):
    """ Convert location change to breakpoint series

    Args:
        lon (DataArray): longitudes
        lat (DataArray): latitudes
        dim (str): datetime dimension
        ilon (float): location longitude
        ilat (float): location latitude
        **kwargs:

    Returns:
        DataArray : distances between locations
    """
    import numpy as np
    from xarray import DataArray, full_like
    from ..fun import distance

    # count occurence of each coordinate pair
    # use the most common (also the most recent?)
    # to estimate distance from,
    # era-interim has a distance of about 80km so only if larger it would make sense to split?
    if not isinstance(lon, DataArray):
        raise ValueError('requires a DataArray', type(lon))
    if not isinstance(lat, DataArray):
        raise ValueError('requires a DataArray', type(lat))

    lon = lon.copy()
    lat = lat.copy()
    lon = lon.bfill(dim)
    lat = lat.bfill(dim)

    dist = full_like(lon, 0, dtype=float)
    dist.name = 'distance'
    dist.attrs['units'] = 'km'
    fdistance = np.vectorize(distance)
    ishape = lon.values.shape
    lon = lon.values.flatten()
    lat = lat.values.flatten()
    if ilon is None and ilat is None:
        # distance between more recent and less recent
        tmp = fdistance(lon[1:], lat[1:], lon[:-1], lat[:-1])
        tmp = np.append(tmp, tmp[-1])
        dist.values = tmp.reshape(ishape)
        dist.attrs['method'] = 'Backwards'
    else:
        tmp = fdistance(lon, lat, ilon, ilat)
        dist.values = tmp.reshape(ishape)
        dist.attrs['method'] = 'Point(%f E, %f N)' % (ilon, ilat)

    return dist


def sondetype(data, dim='date', missing=None, **kwargs):
    """ Make breakpoint series from sondetype changes

    Args:
        data (DataArray): Sondetype values
        dim (str): datetime dimension
        thres (int, float): threshold for breakpoint
        **kwargs:

    Returns:
        DataArray : sondetype changes
    """
    import numpy as np
    from .adj import idx2shp
    from xarray import DataArray, full_like

    if not isinstance(data, DataArray):
        raise ValueError('requires a DataArray', type(data))

    data = data.copy()
    if missing is not None:
        if not isinstance(missing, (list, tuple)):
            missing = [missing]

        # Adjust values
        for im in missing:
            data.values = np.where(data.values == im, np.nan, data.values)

    # replace NAN
    data = data.ffill(dim=dim).bfill(dim=dim)
    #
    axis = data.dims.index(dim)
    stype = full_like(data, 0, dtype=float)
    stype.name = 'event'
    idx = idx2shp(slice(1, None), axis, data.values.shape)
    stype.values[idx] = np.apply_along_axis(np.diff, axis, data.values)
    stype = stype.fillna(0)
    stype.values = (stype.values != 0).astype(int)
    return stype


def metadata(ident, dates, igra=None, wmo=None, return_dataframes=False, **kwargs):
    """ Read IGRA and WMO _metadata along a given datetime axis

    Args:
        ident:
        dates:
        igra:
        wmo:
        return_dataframes:
        **kwargs:

    Returns:

    """
    import numpy as np
    import pandas as pd
    from .. import get_data
    from ..fun import message

    if not isinstance(ident, (str, int)):
        raise ValueError("Requires a string or int Radiosonde ID")
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("Requires a pandas.DatetimeIndex")

    igra_id = False
    if isinstance(ident, str):
        if len(ident) > 6:
            igra_id = True
    else:
        ident = "%06d" % ident

    # time IGRA, WMO vars
    #      1     0

    if igra is None:
        igra = pd.read_json(get_data('igrav2_metadata.json'))

    if igra_id:
        igra = igra[igra.id == ident]
    else:
        igra = igra[igra.wmoid == int(ident)]

    message('IGRA Events:', igra.shape, mname='META', **kwargs)
    event = igra.drop_duplicates('date').set_index('date')
    event = pd.Series(1, index=event.index).reindex(dates).fillna(0)

    if wmo is None:
        wmo = pd.read_json(get_data('wmo_metadata.json'))

    wmo = wmo[wmo.id.str.contains(ident)]
    message('WMO Type changes', wmo.shape, mname='META', **kwargs)
    iwmo = pd.Series(0, index=dates)
    for row in wmo.iterrows():
        iwmo.loc[slice(row[1]['start'], row[1]['stop'])] = row[1]['c']

    print(iwmo.sum())
    data = pd.concat([event, iwmo], axis=1, keys=['event_igra', 'sondetype_wmo'])
    data['sondetype_wmo'] = data['sondetype_wmo'].replace(0, np.nan)
    data['sondetype_wmo'] = data['sondetype_wmo'].replace(-1, np.nan)
    data['sondetype_wmo'] = data['sondetype_wmo'].bfill().ffill()
    data['event_wmo'] = 0
    data.loc[1:, 'event_wmo'] = (np.diff(data.sondetype_wmo.values) != 0).astype(int)
    data.index.name = 'date'
    data = data.to_xarray()
    if return_dataframes:
        return data, igra, wmo
    return data


def wmo_code_table():
    """ Metadata from WMO Common Code Tables to binary and alphanumeric codes
    Table 3685
    Version 7.11.2018

    Returns:
        DataFrame : Radiosonde Code Table C2
    """
    import pandas as pd
    from .. import get_data

    return pd.read_csv(get_data('Common_C02_20181107_en.txt'))
