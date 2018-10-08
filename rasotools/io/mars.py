# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import xarray as xr

from ..fun import message


__all__ = ['open_mars_odb']


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
    from .. import config

    if directory is None:
        directory = config.rasodir

    if filename is None and not force_read_ascii:
        filename = directory + '/%s/MARS_ODB.nc' % ident
        message(filename, mname='OMO', **kwargs)

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
    from .. import config
    from ..fun import interp_dataframe

    interp_new = False

    if levels is not None:
        if not np.isin(levels, config.era_plevels).all():
            interp_new = True
            save = False
            message("Not all levels in ERA-I levels, interpolating again", levels, mname='INTP', **kwargs)

    if not interp_new and os.path.isfile(config.rasodir + '/%s/MARS_ODB.nc' % ident) and not force:
        # RECOVER ASCII
        data = xr.open_dataset(config.rasodir + '/%s/MARS_ODB.nc' % ident)   # DataSet
        save = False  # don't save non ERA-I levels complaint
        if levels is not None:
            message(ident, levels, mname='SEL', **kwargs)
            data = data.sel(pres=levels)

    else:
        if levels is None:
            levels = config.era_plevels
        # READ ASCII
        data, station = read_ascii(ident, filename=filename, **kwargs)  # DataFrame
        message(ident, levels, mname='INTP', **kwargs)
        numlev = data.groupby(data.index).nunique().max(axis=1)   # number of levels
        data = interp_dataframe(data, 'pres', levels=levels, **kwargs)

        # Add Metadata
        new = {}
        for ivar in data.columns.tolist():
            if ivar == 'pres':
                continue
            tmp = data.loc[:, ['pres', ivar]].reset_index().set_index(['date', 'pres']).to_xarray()  # 1D -> 2D
            new[ivar] = tmp[ivar]
            new[ivar]['pres'].attrs.update({'units': 'Pa', 'standard_name': 'air_pressure', 'axis': 'Z'})
            new[ivar]['date'].attrs.update({'axis': 'T'})

            if ivar in metadata.keys():
                if 'dpd' in ivar:
                    if 'dewp' not in data.columns:
                        attrs = metadata[ivar]
                        attrs.update({'esat': 'FOEEWMO', 'rounded': 1})
                        new[ivar].attrs.update(attrs)

                else:
                    new[ivar].attrs.update(metadata[ivar])

        data = xr.Dataset(new)
        data.attrs.update({'ident': ident, 'source': 'ECMWF', 'info': 'MARS ODB', 'dataset': 'ERA-INTERIM',
                           'levels': 'ERA-I 32 lower', 'processed': 'UNIVIE, IMG', 'libs': config.libinfo})

        station = station.reindex(np.unique(data.date.values))  # same dates as data
        station = station.fillna(method='ffill')  # fill Missing information with last known
        station['numlev'] = numlev
        station = station.to_xarray()
        for ivar, idata in station.data_vars.items():
            data[ivar] = idata

    if save:
        data.to_netcdf(config.rasodir + '/%s/MARS_ODB.nc' % ident)

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
    from .. import config

    if filename_pattern is None:
        filename_pattern = "%06d_t.txt.gz"

    if isinstance(ident, (int, float)):
        ident = "%06d" % (int(ident))

    if filename is None:
        filename = config.marsdir + '/' + filename_pattern % int(ident)

    colnames = ['date', 'time', 'obstype', 'codetype', 'sondetype', 'ident', 'lat', 'lon', 'alt', 'pres',
                'varno', 'obsvalue', 'biascorr', 'fg_dep', 'an_dep', 'status', 'anflag', 'event1']

    message(ident,'Reading', filename, mname='MRA', **kwargs)

    tmp = pd.read_csv(filename, sep=' ', error_bad_lines=False, header=None, names=colnames, engine='c',
                      dtype={'date': str, 'time': str}, skipinitialspace=True)
    #
    # Convert Date
    #
    tmp['newdate'] = pd.to_datetime(tmp.date + tmp.time, format='%Y%m%d%H%M%S')
    tmp = tmp.set_index('newdate').drop(['date', 'time'], 1)
    tmp.index.name = 'date'

    message(ident, str(tmp.shape), mname='MRA', **kwargs)

    if np.size(tmp.ident.unique()) > 1:
        message("Multiple Idents in file: ", tmp.ident.unique(), mname='MRA', **kwargs)
        tmp = tmp[tmp.ident == int(ident)]

    station = tmp[['lon', 'lat', 'alt', 'sondetype', 'codetype', 'obstype']].drop_duplicates().sort_index()
    station.index.name = 'date'
    tmp = tmp.drop(['lon', 'lat', 'alt', 'sondetype', 'codetype', 'obstype', 'ident', 'event1', 'status', 'anflag'], axis=1)

    variables = {1: 'geop', 2: 'temp', 3: 'uwind', 4: 'vwind', 7: 'qhumi', 29: 'rhumi', 59: 'dewp'}

    data = pd.DataFrame(columns=['date', 'pres'])
    for i, ivar in variables.items():
        if not tmp.varno.isin([i]).any():
            continue
        mappings = {'obsvalue': ivar, 'fg_dep': '%s_fg_dep' % ivar, 'an_dep': '%s_an_dep' % ivar,
                    'biascorr': '%s_biascorr' % ivar}
        qdata = tmp.loc[tmp.varno == i, :].rename(columns=mappings).drop('varno', axis=1)
        qdata.index.name = 'date'
        qdata.drop_duplicates(inplace=True)
        qdata = qdata.reset_index()
        data = pd.merge(data, qdata, how='outer', on=['date', 'pres'])
        message(ident, 'Variable:', i, ivar, mname='MRA', **kwargs)

    data = data.sort_values(['date', 'pres']).set_index('date')
    message(ident, 'Dropping duplicates', mname='MRA', **kwargs)
    data.drop_duplicates(inplace=True)
    # Apply conversion of t, r to DPD
    if 'dewp' in data.columns:
        # check how much data there is
        if data['dewp'].count().sum() > data['rhumi'].count().sum():
            data['dpd'] = (data['temp'] - data['dewp'])

    if 'dpd' not in data.columns:
        if data.columns.isin(['temp', 'rhumi']).sum() == 2:
            data['dpd'] = (data['temp'] - dewpoint_ECMWF(data['temp'].values, data['rhumi'].values)).round(1)
            logic = data['dpd'].values < 0
            data['dpd'] = np.where(logic, 0, data['dpd'].values)

    return data, station


def foeewm(t, **kwargs):
    """from IFS Documentation Cycle 31,
    Teten's formula for mixed phases water, ice
    after Tiedtke 1993 (cloud and large-scale precipitation scheme)
    Based on Buck 1981 & Alduchov and Eskridge 1996

    Args:
        t: air temperature K

    Returns:
        es : saturation water vapor in Pa
    """
    import numpy as np
    # T Larger than 273.15 K > only water
    ew = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))  # Liquid
    ei = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))  # Ice
    e = np.where(t > 273.15, ew, ei)
    # combine ice and water for mixed clouds
    e = e + np.where((t >= 250.15) & (t <= 273.15), (ew - ei) * ((t - 250.16) / 23.) ** 2, 0)
    return e


def FOEEWMO(t, **kwargs):
    """from IFS Documentation Cycle 31,
    Teten's formula for water only
    after Tiedtke 1993 (cloud and large-scale precipitation scheme)
    Uses the Tetens's formula
    Based on Buck 1981 & Alduchov and Eskridge 1996

    Args:
        t: air temperature K

    Returns:
        es : saturation water vapor in Pa
    """
    import numpy as np
    return 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))


def sh2rh_ECMWF(q, t, p):
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
    e = FOEEWMO(t) / p
    a = np.where(e < 0.5, e, 0.5)
    a = np.where(np.isfinite(t), a, np.nan)  # TODO maybe remove this line?
    return q / (a * (1 + (461.5250 / 287.0597 - 1) * q))


def rh2sh_ECMWF(r, t, p):
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
    e = FOEEWMO(t) / p
    a = np.where(e < 0.5, e, 0.5)
    a = np.where(np.isfinite(t), a, np.nan)
    return r * (a / (1 - r * (461.5250 / 287.0597 - 1) * a))


def dewpoint_ECMWF(t, rh, **kwargs):
    """ ECMWF IFS CYC31R1 Data Assimilation Documentation (Page 86-88)
    Derived Variables from TEMP
    Temperature and dew point are transformed into realtive humidity (RH) for
    TEMP observations, with a further transformation of RH into specific humidity (Q)
    for TEMP observations.

    Uses FOEEWMO for water only saturation water vapor pressure

    Parameters
    ----------
    t       temperature     [K]
    rh      rel. humidity   [1]

    Returns
    -------
    td      dew point       [K]
    """
    import numpy as np
    e = FOEEWMO(t) * rh
    lnpart = np.where(e > 0, np.log(e / 611.21), np.nan)
    return (17.502 * 273.16 - 32.19 * lnpart) / (17.502 - lnpart)


metadata = {'temp': {'units': 'K', 'standard_name': 'air_temperature'},
            'temp_fg_dep': {'units': 'K', 'standard_name': 'first guess departure'},
            'temp_an_dep': {'units': 'K', 'standard_name': 'analysis departure'},
            'temp_biascorr': {'units': 'K', 'standard_name': 'bias adjustment'},
            'rhumi': {'units': '1', 'standard_name': 'relative_humidity'},
            'rhumi_fg_dep': {'units': '1', 'standard_name': 'relative_humidity_first_guess_departure'},
            'rhumi_an_dep': {'units': '1', 'standard_name': 'relative_humidity_analysis_departure'},
            'rhumi_biascorr': {'units': '1', 'standard_name': 'relative_humidity_bias_adjusstment'},
            'qhumi': {'units': 'kg/kg', 'standard_name': 'specific_humidity'},
            'qhumi_fg_dep': {'units': 'kg/kg', 'standard_name': 'specific_humidity_first_guess_departure'},
            'qhumi_an_dep': {'units': 'kg/kg', 'standard_name': 'specific_humidity_analysis_departure'},
            'qhumi_biascorr': {'units': 'kg/kg', 'standard_name': 'specific_humidity_bias_adjustment'},
            'dewp': {'units': 'K', 'standard_name': 'dew_point'},
            'dewp_fg_dep': {'units': 'K', 'standard_name': 'dew_point_first_guess_departure'},
            'dewp_an_dep': {'units': 'K', 'standard_name': 'dew_point_analysis_departure'},
            'dewp_biascorr': {'units': 'K', 'standard_name': 'dew_point_bias_adjustment'},
            'uwind': {'units': 'm/s', 'standard_name': 'eastward_wind'},
            'uwind_fg_dep': {'units': 'm/s', 'standard_name': 'eastward_wind_first_guess_departure'},
            'uwind_an_dep': {'units': 'm/s', 'standard_name': 'eastward_wind_analysis_departure'},
            'uwind_biascorr': {'units': 'm/s', 'standard_name': 'eastward_wind_bias_adjustment'},
            'vwind': {'units': 'm/s', 'standard_name': 'northward_wind'},
            'vwind_fg_dep': {'units': 'm/s', 'standard_name': 'northward_wind_first_guess_departure'},
            'vwind_an_dep': {'units': 'm/s', 'standard_name': 'northward_wind_analysis_departure'},
            'vwind_biascorr': {'units': 'm/s', 'standard_name': 'northward_wind_bias_adjustment'},
            'dpd': {'units': 'K', 'standard_name': 'dew_point_depression'}
            }
