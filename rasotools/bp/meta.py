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


def open_igra_metadata(filename):
    """ Read IGRAv2 _metadata file according to readme

    igra2-_metadata-readme.txt

    Documentation for IGRA Station History Information
    Accompanying IGRA Version 2.0.0b1
    August 2014

    Args:
        filename (str):  igra2-_metadata.txt

    Returns:
        DataFrame
    """
    import pandas as pd
    infos = """
    IGRAID         1- 11   Character
    WMOID         13- 17   Integer
    NAME          19- 48   Character
    NAMFLAG       50- 50   Character
    LATITUDE      52- 60   Real
    LATFLAG       62- 62   Character
    LONGITUDE     64- 72   Real
    LONFLAG       74- 74   Character
    ELEVATION     76- 81   Real
    ELVFLAG       83- 83   Character
    YEAR          85- 88   Integer
    MONTH         90- 91   Integer
    DAY           93- 94   Integer
    HOUR          96- 97   Integer
    DATEIND       99- 99   Integer
    EVENT        101-119   Character
    ALTIND       121-122   Character
    BEFINFO      124-163   Character
    BEFFLAG      164-164   Character
    LINK         166-167   Character
    AFTINFO      169-208   Character
    AFTFLAG      209-209   Character
    REFERENCE    211-235   Character
    COMMENT      236-315   Character
    UPDCOM       316-346   Character
    UPDDATE      348-354   Character
    """
    import numpy as np
    colspecs = []
    header = []
    types = {}
    for iline in infos.splitlines():
        if iline == '':
            continue
        ih = iline[0:11].strip().lower()
        header.append(ih)
        ii = int(iline[13:16]) - 1
        ij = int(iline[17:20])
        colspecs.append((ii, ij))
        it = iline[22:].strip()
        if it == 'Character':
            it = 'str'
        elif it == 'Real':
            it = 'float'
        else:
            it = 'int'
        types[ih] = it

    data = pd.read_fwf(filename, colspecs=colspecs, header=None, dtype=types, names=header)
    data = data.replace('nan', '')
    data['date'] = pd.to_datetime((data.year * 1000000 +
                                   np.where(data.month.values == 99, 6, data.month.values) * 10000 +
                                   np.where(data.day.values == 99, 15, data.day.values) * 100 +
                                   np.where(data.hour.values == 99, 0, data.hour.values)).apply(str), format='%Y%m%d%H')
    return data
