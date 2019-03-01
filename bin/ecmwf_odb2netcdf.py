#!/usr/bin/env python
# -*- coding: utf-8 -*-


__doc__ = """
Read ECMWF ODB file and convert it to netcdf and standard pressure levels

"""


def open_mars_odb(ident, variables=None, filename=None, directory=None, force_read_ascii=False, levels=None, **kwargs):
    """ Read ECMWF MARS ODB file/dump into Dataset

    Parameters
    ----------
    ident : str
    variables : str / list
    filename : str
    directory : str
    force_read_ascii : bool
    levels : list
    kwargs : dict

    Returns
    -------
    Dataset : MARS
    """
    from ..fun import message
    from .. import config

    if directory is None:
        directory = config.rasodir

    if filename is None and not force_read_ascii:
        filename = directory + '/%s/MARS_ODB.nc' % ident
        message(filename, **kwargs)

    data = to_xarray(ident, filename=filename, levels=levels, force=force_read_ascii, **kwargs)
    if variables is not None:
        avail = list(data.data_vars.keys())
        if not isinstance(variables, list):
            variables = [variables]
        variables = [iv for iv in variables if iv in avail]
        if len(variables) > 0:
            data = data[variables]  # subset
    return data


def to_xarray(ident, filename=None, save=True, levels=None, force=False, **kwargs):
    """ Convert MARS ODB to xArray

    Parameters
    ----------
    ident
    filename
    save
    levels
    force
    kwargs

    Returns
    -------

    """
    import os

    import numpy as np
    import xarray as xr

    from ..fun import message
    from .. import config
    from ..fun.interp import dataframe

    interp_new = False

    if levels is not None:
        if not np.isin(levels, config.era_plevels).all():
            interp_new = True
            save = False
            message("Not all levels in ERA-I levels, interpolating again", levels, **kwargs)

    if not interp_new and os.path.isfile(config.rasodir + '/%s/MARS_ODB.nc' % ident) and not force:
        #
        # READ NetCDF
        #
        data = xr.open_dataset(config.rasodir + '/%s/MARS_ODB.nc' % ident)  # DataSet
        if levels is not None:
            message(ident, levels, **kwargs)
            data = data.sel(pres=levels)

    else:
        if levels is None:
            levels = config.era_plevels
        #
        # READ ASCII
        #
        data, station = read_ascii(ident, filename=filename, **kwargs)  # DataFrame
        message(ident, levels, **kwargs)
        numlev = data.groupby(data.index).nunique().max(axis=1)  # number of levels
        #
        # Interpolation
        #
        data = dataframe(data, 'pres', levels=levels, **kwargs)
        #
        # convert from Table to Array and add Metadata
        #
        new = {}
        for ivar in data.columns.tolist():
            if ivar == 'pres':
                continue
            tmp = data.loc[:, ['pres', ivar]].reset_index().set_index(['date', 'pres']).to_xarray()  # 1D -> 2D
            new[ivar] = tmp[ivar]
            new[ivar]['pres'].attrs.update({'units': 'Pa', 'standard_name': 'air_pressure', 'axis': 'Z'})
            new[ivar]['date'].attrs.update({'axis': 'T'})

            #
            # Access global _metadata attributes
            #
            if ivar in _metadata.keys():
                if 'dpd' in ivar:
                    if 'dewp' not in data.columns:
                        attrs = _metadata[ivar]
                        attrs.update({'esat': 'foeewmo', 'rounded': 1})
                        new[ivar].attrs.update(attrs)

                else:
                    new[ivar].attrs.update(_metadata[ivar])
        #
        # Dataset + Attributes
        #
        data = xr.Dataset(new)
        data.attrs.update({'ident': ident, 'source': 'ECMWF', 'info': 'MARS ODB', 'dataset': 'ERA-INTERIM',
                           'levels': 'plevs [%d -%d] #%d' % (min(levels), max(levels), len(levels)),
                           'processed': 'UNIVIE, IMG', 'libs': config.libinfo})
        #
        # Station Information
        #
        if not station.index.is_unique:
            station = station.reset_index().drop_duplicates(['date', 'lon', 'lat', 'alt']).set_index('date')

        station = station.reindex(np.unique(data.date.values))  # same dates as data
        # station = station.fillna(method='ffill')  # fill Missing information with last known
        station['numlev'] = numlev
        station = station.to_xarray()
        for ivar, idata in station.data_vars.items():
            data[ivar] = idata

        if save:
            data.to_netcdf(config.rasodir + '/%s/MARS_ODB.nc' % ident)
            message(ident, 'Saving: ', config.rasodir + '/%s/MARS_ODB.nc' % ident, **kwargs)

    return data


def read_ascii(ident, filename=None, filename_pattern=None, **kwargs):
    """ Read odb ascii dump from ECMWF ERAINTERIM ARCHIVE

    Parameters
    ----------
    ident : str
        WMO Radiosonde ID
    filename : str
        Filename
    filename_pattern : str
        Filename pattern
    kwargs : dict
        optional keyword arguments

    Returns
    -------
    DataFrame, DataFrame :
        data , station parameters
    """

    import numpy as np
    import pandas as pd

    from ..fun import message
    from .. import config

    if filename_pattern is None:
        filename_pattern = "%06d_t.txt.gz"

    if isinstance(ident, (int, float)):
        ident = "%06d" % (int(ident))

    if filename is None:
        if config.marsdir == '':
            message(ident, 'Config: MARSDIR is unset', **kwargs)
        filename = config.marsdir + '/' + filename_pattern % int(ident)

    colnames = ['date', 'time', 'obstype', 'codetype', 'sondetype', 'ident', 'lat', 'lon', 'alt', 'pres',
                'varno', 'obsvalue', 'biascorr', 'fg_dep', 'an_dep', 'status', 'anflag', 'event1']

    message(ident, 'Reading', filename, **kwargs)

    tmp = pd.read_csv(filename, sep=' ', error_bad_lines=False, header=None, names=colnames, engine='c',
                      dtype={'date': str, 'time': str}, skipinitialspace=True)
    #
    # Convert Date
    #
    tmp['newdate'] = pd.to_datetime(tmp.date + tmp.time, format='%Y%m%d%H%M%S')
    tmp = tmp.set_index('newdate').drop(['date', 'time'], 1)
    tmp.index.name = 'date'
    #
    # Different IDs?
    #
    message(ident, str(tmp.shape), **kwargs)
    if np.size(tmp.ident.unique()) > 1:
        message("Multiple Idents in file: ", tmp.ident.unique(), **kwargs)
        tmp = tmp[tmp.ident == int(ident)]
    #
    # Separate Station Information
    #
    station = tmp[['lon', 'lat', 'alt', 'sondetype', 'codetype', 'obstype']].reset_index().drop_duplicates(
        'date').set_index('date').sort_index()
    station.index.name = 'date'
    tmp = tmp.drop(['lon', 'lat', 'alt', 'sondetype', 'codetype', 'obstype', 'ident', 'event1', 'status', 'anflag'],
                   axis=1)
    #
    # Variable Mappings
    #
    variables = {1: 'geop', 2: 'temp', 3: 'uwind', 4: 'vwind', 7: 'qhumi', 29: 'rhumi', 59: 'dewp'}
    #
    # DataFrame with variables and same pres and dates
    #
    data = pd.DataFrame(columns=['date', 'pres'])
    for i, ivar in variables.items():
        if not tmp.varno.isin([i]).any():
            continue
        #
        # new names
        #
        mappings = {'obsvalue': ivar, 'fg_dep': '%s_fg_dep' % ivar, 'an_dep': '%s_an_dep' % ivar,
                    'biascorr': '%s_biascorr' % ivar}
        qdata = tmp.loc[tmp.varno == i, :].rename(columns=mappings).drop('varno', axis=1)
        qdata.index.name = 'date'
        qdata.drop_duplicates(inplace=True)
        qdata = qdata.reset_index()
        #
        # MERGE (slow)
        #
        data = pd.merge(data, qdata, how='outer', on=['date', 'pres'])
        message(ident, 'Variable:', i, ivar, **kwargs)

    #
    # Sorting
    #
    data = data.sort_values(['date', 'pres']).set_index('date')
    message(ident, 'Dropping duplicates', **kwargs)
    #
    # remove duplicates (?)
    #
    data.drop_duplicates(inplace=True)
    #
    # Apply conversion of t, r to DPD
    #
    if 'dewp' in data.columns:
        # check how much data there is
        if data['dewp'].count().sum() > data['rhumi'].count().sum():
            data['dpd'] = (data['temp'] - data['dewp'])

    if 'dpd' not in data.columns:
        if data.columns.isin(['temp', 'rhumi']).sum() == 2:
            data['dpd'] = (data['temp'] - dewpoint_ecmwf(data['temp'].values, data['rhumi'].values)).round(1)
            logic = data['dpd'].values < 0
            data['dpd'] = np.where(logic, 0, data['dpd'].values)

    return data, station


def foeewm(t, **kwargs):
    """from IFS Documentation Cycle 31,
    Teten's formula for mixed phases water, ice
    after Tiedtke 1993 (cloud and large-scale precipitation scheme)
    Based on Buck 1981 & Alduchov and Eskridge 1996

    Args:
        t (any): air temperature K

    Returns:
        np.ndarray : saturation water vapor in Pa
    """
    import numpy as np
    # T Larger than 273.15 K > only water
    ew = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))  # Liquid
    ei = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))  # Ice
    e = np.where(t > 273.15, ew, ei)
    # combine ice and water for mixed clouds
    e = e + np.where((t >= 250.15) & (t <= 273.15), (ew - ei) * ((t - 250.16) / 23.) ** 2, 0)
    return e


def foeewmo(t, **kwargs):
    """from IFS Documentation Cycle 31,
    Teten's formula for water only
    after Tiedtke 1993 (cloud and large-scale precipitation scheme)
    Uses the Tetens's formula
    Based on Buck 1981 & Alduchov and Eskridge 1996

    Args:
        t (any): air temperature K

    Returns:
        es : saturation water vapor in Pa
    """
    import numpy as np
    return 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))


def sh2rh_ecmwf(q, t, p):
    """ ECMWF IFS CYC31R1 Data Assimilation Documentation (Page 86-88)

    Conversion of q, t and p to r

    Parameters
    ----------
    q       spec. humidity  [kg/kg]
    t       temperature     [K]
    p       pressure        [Pa]

    Returns
    -------
    r       rel. humidity   [1]
    """
    import numpy as np
    e = foeewmo(t) / p
    a = np.where(e < 0.5, e, 0.5)
    a = np.where(np.isfinite(t), a, np.nan)  # TODO maybe remove this line?
    return q / (a * (1 + (461.5250 / 287.0597 - 1) * q))


def rh2sh_ecmwf(r, t, p):
    """ ECMWF IFS CYC31R1 Data Assimilation Documentation (Page 86-88)

    Conversion of r, t and p to q

    Parameters
    ----------
    r       rel. humidity  [1]
    t       temperature    [K]
    p       pressure       [Pa]

    Returns
    -------
    q       spec. humidty [kg/kg]
    """
    import numpy as np
    e = foeewmo(t) / p
    a = np.where(e < 0.5, e, 0.5)
    a = np.where(np.isfinite(t), a, np.nan)
    return r * (a / (1 - r * (461.5250 / 287.0597 - 1) * a))


def dewpoint_ecmwf(t, rh, **kwargs):
    """ ECMWF IFS CYC31R1 Data Assimilation Documentation (Page 86-88)
    Derived Variables from TEMP
    Temperature and dew point are transformed into realtive humidity (RH) for
    TEMP observations, with a further transformation of RH into specific humidity (Q)
    for TEMP observations.

    Uses foeewmo for water only saturation water vapor pressure

    Parameters
    ----------
    t       temperature     [K]
    rh      rel. humidity   [1]

    Returns
    -------
    td      dew point       [K]
    """
    import numpy as np
    e = foeewmo(t) * rh
    lnpart = np.where(e > 0, np.log(e / 611.21), np.nan)
    return (17.502 * 273.16 - 32.19 * lnpart) / (17.502 - lnpart)


_metadata = {'t': {'units': 'K', 'standard_name': 'air_temperature'},
             't_fg_dep': {'units': 'K', 'standard_name': 'first guess departure'},
             't_an_dep': {'units': 'K', 'standard_name': 'analysis departure'},
             't_biascorr': {'units': 'K', 'standard_name': 'bias adjustment'},
             'rh': {'units': '1', 'standard_name': 'relative_humidity'},
             'rh_fg_dep': {'units': '1', 'standard_name': 'relative_humidity_first_guess_departure'},
             'rh_an_dep': {'units': '1', 'standard_name': 'relative_humidity_analysis_departure'},
             'rh_biascorr': {'units': '1', 'standard_name': 'relative_humidity_bias_adjusstment'},
             'q': {'units': 'kg/kg', 'standard_name': 'specific_humidity'},
             'q_fg_dep': {'units': 'kg/kg', 'standard_name': 'specific_humidity_first_guess_departure'},
             'q_an_dep': {'units': 'kg/kg', 'standard_name': 'specific_humidity_analysis_departure'},
             'q_biascorr': {'units': 'kg/kg', 'standard_name': 'specific_humidity_bias_adjustment'},
             'td': {'units': 'K', 'standard_name': 'dew_point'},
             'td_fg_dep': {'units': 'K', 'standard_name': 'dew_point_first_guess_departure'},
             'td_an_dep': {'units': 'K', 'standard_name': 'dew_point_analysis_departure'},
             'td_biascorr': {'units': 'K', 'standard_name': 'dew_point_bias_adjustment'},
             'u': {'units': 'm/s', 'standard_name': 'eastward_wind'},
             'u_fg_dep': {'units': 'm/s', 'standard_name': 'eastward_wind_first_guess_departure'},
             'u_an_dep': {'units': 'm/s', 'standard_name': 'eastward_wind_analysis_departure'},
             'u_biascorr': {'units': 'm/s', 'standard_name': 'eastward_wind_bias_adjustment'},
             'v': {'units': 'm/s', 'standard_name': 'northward_wind'},
             'v_fg_dep': {'units': 'm/s', 'standard_name': 'northward_wind_first_guess_departure'},
             'v_an_dep': {'units': 'm/s', 'standard_name': 'northward_wind_analysis_departure'},
             'v_biascorr': {'units': 'm/s', 'standard_name': 'northward_wind_bias_adjustment'},
             'dpd': {'units': 'K', 'standard_name': 'dew_point_depression'}
             }


def usage():
    print("""
    important conversion tool for ECMWF ODB dump to netcdf and standard pressure levels
    """)


if __name__ == "__main__":
    import sys
    # call [odb file] -o -levels -ident
    pass
