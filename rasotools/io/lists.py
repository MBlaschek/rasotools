# -*- coding: utf-8 -*-

__all__ = ['read_radiosondelist']


def read_radiosondelist(filename=None, minimal=True, with_igra=False, **kwargs):
    import pandas as pd
    from .. import get_data

    if filename is None:
        filename = get_data('radiosondeslist.csv')

    if '.csv' not in filename:
        raise ValueError("Unknown Radiosondelist")

    table = pd.read_csv(filename, sep=";", index_col=0)
    for icol in table.columns:
        if table[icol].dtype == 'object':
            table.loc[table[icol].isnull(), icol] = ''

    if minimal:
        return table[['lon', 'lat', 'alt', 'name']]
    elif with_igra:
        return table[['lon', 'lat', 'alt', 'name', 'id_igra']]
    else:
        return table


def igra(filename=None):
    """Read IGRA Radiosondelist

        or download

        Parameters
        ----------
        new         bool
        filename    str
        verbose     int

        Returns
        -------
        DataFrame
    """
    import numpy as np
    import pandas as pd
    from .. import get_data

    if filename is None:
        filename = get_data('igra2-station-list.txt')

    try:
        infile = open(filename)
        tmp = infile.read()
        data = tmp.splitlines()

    except IOError as e:
        print("File not found: " + filename)
        raise e
    else:
        infile.close()

    out = pd.DataFrame(columns=['id', 'wmo', 'lat', 'lon', 'alt', 'state', 'name', 'start', 'end', 'total'])

    for i, line in enumerate(data):
        id = line[0:11]

        try:
            id2 = "%06d" % int(line[5:11])  # substring

        except Exception as e:
            id2 = ""

        lat = float(line[12:20])
        lon = float(line[21:30])
        alt = float(line[31:37])
        state = line[38:40]
        name = line[41:71]
        start = int(line[72:76])
        end = int(line[77:81])
        count = int(line[82:88])
        out.loc[i] = (id, id2, lat, lon, alt, state, name, start, end, count)

    out.loc[out.lon <= -998.8, 'lon'] = np.nan  # repalce missing values
    out.loc[out.alt <= -998.8, 'alt'] = np.nan  # repalce missing values
    out.loc[out.lat <= -98.8, 'lat'] = np.nan  # replace missing values
    out['name'] = out.name.str.strip()
    out = out.set_index('id')
    return out


def Obstype(data, typ):
    import pandas as pd
    out = []
    for i in data.values:
        if ',' in i:
            j = i.split(',')
            status = False
            for k in j:
                if k.strip() == typ:
                    status = True
            out += [status]
        else:
            if i.strip() == typ:
                out += [True]
            else:
                out += [False]
    return pd.Series(out, index=data.index)


def wmolist(ifile, minimal=True, only_raso=True):
    """ Read WMO Radiosonde Station List

    ANTON(T)    : Antarctic Observing Network upper-air station (TEMP)
    GUAN        : GCOS Upper-Air Network station
    RBSN(T)     : Regional Basic Synoptic Network upper-air station (TEMP)
    RBSN(P)     : Regional Basic Synoptic Network upper-air station (PILOT)
    RBSN(ST)    : Regional Basic Synoptic Network surface and upper-air station (SYNOP/TEMP)
    RBSN(SP)    : Regional Basic Synoptic Network surface and upper-air station (SYNOP/PILOT)
    WN          : Upper-wind observations made by using navigation aids (NAVAID)
    WR          : Upper-wind observations made by radar
    WT          : Upper-wind observations made by radiotheodolite
    WTR         : Upper-wind observations made by radiotheodolite/radar composite method

    Args:
        ifile (str): filename to read (wmo-stations.txt)
        minimal (bool): subset of columns
        only_raso (bool): only radiosonde stations

    Returns:
        pd.DataFrame : station list
    """
    import numpy as np
    import pandas as pd
    from .. import get_data

    if ifile is None:
        ifile = get_data('wmo-stations.txt')

    try:
        wd = pd.read_csv(ifile, sep='\t')
    except IOError as e:
        print("Error missing file: ", ifile)
        raise e

    sign = np.where(wd.Longitude.apply(lambda x: 'E' in x).values, 1, -1)
    wd.loc[:, 'Longitude'] = wd.Longitude.apply(
        lambda x: np.sum(np.float_(x[:-1].split()) / [1., 60., 3600.])).values * sign

    sign = np.where(wd.Latitude.apply(lambda x: 'S' in x).values, -1, 1)
    wd.loc[:, 'Latitude'] = wd.Latitude.apply(
        lambda x: np.sum(np.float_(x[:-1].split()) / [1., 60., 3600.])).values * sign

    # wd.ix[:, 'CountryArea'] = wd.CountryArea.apply(lambda x: x.split('/')[0])
    # wd.ix[:, 'RegionName'] = wd.RegionName.apply(lambda x: x.split('/')[0])
    wd = wd.rename(columns={'Longitude': 'lon', 'Latitude': 'lat', 'IndexNbr': 'id', 'StationName': 'name',
                            'Hp': 'alt', 'CountryArea': 'area', 'RegionName': 'region', 'StationId': 'wigos'})

    # require variables named: id, lon,lat,alt,name,count
    wd['id'] = wd.id.map('{:06.0f}'.format)
    rasotypes = ['RBSN(T)', 'RBSN(P)', 'RBSN(ST)', 'RBSN(SP)', 'GUAN', 'ANTON(T)', 'R']
    status = pd.concat([Obstype(wd.ObsRems, ityp) for ityp in rasotypes], axis=1, keys=rasotypes)

    print(status.sum())

    wd['raso'] = np.any([Obstype(wd.ObsRems, ityp) for ityp in rasotypes], axis=0)
    if only_raso:
        wd = wd[wd.raso]

    if minimal:
        wd = wd.set_index('id')
        wd = wd[['lon', 'lat', 'name', 'alt', 'area', 'region', 'wigos']].sort_index().drop_duplicates()

    return wd


def dist_array(data, lon='lon', lat='lat'):
    import numpy as np
    import pandas as pd
    from ..fun import distance
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Requires a DataFrame with lon, lat columns and index WMO')

    matrix = []

    for irow in data.shape[0]:
        matrix += [distance(data[lon], data[lat], data[irow, lon], data[irow, lat])]

    matrix = pd.DataFrame(np.array(matrix), index=data.index, columns=data.index)

    return matrix
