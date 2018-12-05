# -*- coding: utf-8 -*-
import datetime
import gzip
import os
import numpy as np
import pandas as pd
import xarray as xr
from ..fun import find_files, message


__all__ = ['open_igra', 'open_igra_metadata']


def open_igra(ident, variables=None, filename=None, directory=None, force_read_ascii=False, levels=None, **kwargs):
    """ Open or Read IGRAv2 Radiosonde data

    Parameters
    ----------
    ident : str
        WMO ID or IGRA ID
    variables : list / str
        select only these variables
        pres, gph, temp, rhumi, dpd, windd, winds, lon, lat, numlev
    filename : str
        Filename of ASCII IGRA file
    directory : str
        Directory for raso_archive
    force_read_ascii : bool
        force reading ASCII data
    levels : list
        pressure levels to use
        Default: era interim pressure levels
    kwargs : dict

    Returns
    -------
    Dataset
        xarray Dataset with variables at date and era-interim pressure levels
    """
    from .. import config

    if directory is None:
        directory = config.rasodir

    if filename is None and not force_read_ascii:
        filename = directory + '/%s/IGRAv2.nc' % ident
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
    """ Read IGRAv2 Data and interpolate to standard pressure levels

    Parameters
    ----------
    ident : str
        Radiosonde ID
    filename : str
        filename to read
    save : bool
        Save Xarray temporary data ?
    levels : list
        pressure levels to interpolate to
    force : bool
        read from ascii ?
    kwargs : dict
        optional keyword arguments

    Returns
    -------
    Dataset
        Dataset, Xarray of sounding
    """
    from .. import config
    from ..fun.interp import dataframe

    interp_new = False

    if levels is not None:
        if not np.isin(levels, config.era_plevels).all():
            interp_new = True
            save = False
            message("Not all levels in ERA-I levels, interpolating again", levels,  **kwargs)

    if not interp_new and os.path.isfile(config.rasodir + '/%s/IGRAv2.nc' % ident) and not force:
        # RECOVER ASCII
        data = xr.open_dataset(config.rasodir + '/%s/IGRAv2.nc' % ident)   # DataSet
        save = False  # don't save non ERA-I levels complaint
        if levels is not None:
            message(ident, levels, **kwargs)
            data = data.sel(pres=levels)

    else:
        if levels is None:
            levels = config.era_plevels
        # READ ASCII
        data, station = read_ascii(ident, filename=filename, **kwargs)  # DataFrame
        message(ident, levels, **kwargs)
        data = dataframe(data, 'pres', levels=levels, **kwargs)

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
        data.attrs.update({'ident': ident, 'source': 'NOAA NCDC', 'dataset': 'IGRAv2',
                           'levels': 'ERA-I 32 lower', 'processed': 'UNIVIE, IMG', 'libs': config.libinfo})

        station = station.reindex(np.unique(data.date.values))  # same dates as data
        station = station.fillna(method='ffill')  # fill Missing information with last known
        station = station.to_xarray()
        for ivar, idata in station.data_vars.items():
            data[ivar] = idata

    if save:
        data.to_netcdf(config.rasodir + '/%s/IGRAv2.nc' % ident)

    return data


def read_ascii(ident, filename=None, filename_pattern=None, **kwargs):
    """Read IGRA version 2 Data from NOAA

    Args
    ----
        ident : str
            WMO ID or IGRA ID
        filename : str
            Filename
        filename_pattern : str
           filename pattern: *%s-data.txt.gz

    Returns
    -------
    DataFrame
        Table of radiosonde soundings with date as index and variables as columns

    Info:
    Format Description of IGRA 2 Sounding Data Files

    ---------------------
    Notes:
    ---------------------

    2. Both types of files are updated once a day in the early morning Eastern
       Time. The latest observations usually become available within two
       calendar days of when they were taken.

    2. Data files are available for two different time spans:

       In subdirectory data-por, data files contain the full period of record.
       In subdirectory data-y2d, files only contain soundings from the current
         (or current and previous) year. For example, as of August 2016,
         the files in the data-y2d subdirectory begin with January 1, 2016.

    3. Each file in the data-por and data-y2d subdirectories contains the
       sounding data for one station.
       The name of the file corresponds to a station's IGRA 2 identifier (e.g.,
       "USM00072201-data.txt.zip"  contains the data for the station with the
       identifier USM00072201).

    3. Each sounding consists of one header record and n data
       records, where n (given in the header record) is the number of levels
       in the sounding.

    ---------------------
    Header Record Format:
    ---------------------

    ---------------------------------
    Variable   Columns  Type
    ---------------------------------
    HEADREC       1-  1  Character
    ID            2- 12  Character
    YEAR         14- 17  Integer
    MONTH        19- 20  Integer
    DAY          22- 23  Integer
    HOUR         25- 26  Integer
    RELTIME      28- 31  Integer
    NUMLEV       33- 36  Integer
    P_SRC        38- 45  Character
    NP_SRC       47- 54  Character
    LAT          56- 62  Integer
    LON          64- 71  Integer
    ---------------------------------

    These variables have the following definitions:

    HEADREC		is the header record indicator (always set to "#").

    ID		is the station identification code. See "igra2-stations.txt"
            for a complete list of stations and their names and locations.

    YEAR 		is the year of the sounding.

    MONTH 		is the month of the sounding.

    DAY 		is the day of the sounding.

    HOUR 		is the nominal or observation hour of the sounding (in UTC on
            the date indicated in the YEAR/MONTH/DAY fields). Possible
            valid hours are 00 through 23, and 99 = missing. Hours are
            given as provided by the data provider, and the relationship
            between this hour and the release time varies by data
            provider, over time, and among stations.

    RELTIME 	is the release time of the sounding in UTC. The format is
            HHMM, where HH is the hour and MM is the minute. Possible
            are 0000 through 2359, 0099 through 2399 when only the release
            hour is available, and 9999 when both hour and minute are
            missing.

    NUMLEV 		is the number of levels in the sounding (i.e., the number of
            data records that follow).

    P_SRC 		is the data source code for pressure levels in the sounding.
            It has 25 possible values:

            bas-data = British Antarctic Survey READER Upper-Air Data
            cdmp-amr = African Monthly Radiosonde Forms
                       digitized by the U.S. Climate Data Modernization
                       Program
            cdmp-awc = "African Wind Component Data" digitized from
                       Monthly Forms by the U.S. Climate Data
                       Modernization Program
            cdmp-mgr = "WMO-Coded Messages" for Malawi, digitized from
                       "Computer-Generated Forms" by the U.S. Climate
                       Data Modernization Program
            cdmp-zdm = Zambian "Daily UA MB Ascent Sheets" digitized by
                       the U.S. Climate Data Modernization Program
            chuan101 = Comprehensive Historical Upper Air Network (v1.01)
            erac-hud = ERA-CLIM Historical Upper Air Data
            iorgc-id = IORGC/JAMSTEC-Digitized data for Indonesia
            mfwa-ptu = West African Temperature-Humidity Soundings
                       digitized by Meteo-France
            ncar-ccd = C-Cards Radiosonde Data Set from NCAR
            ncar-mit = MIT Global Upper Air Data from NCAR
            ncdc6210 = NCDC Marine Upper Air Data (NCDC DSI-6210)
            ncdc6301 = NCDC U.S. Rawindsonde Data (NCDC DSI-6301)
            ncdc6309 = NCDC "NCAR-NMC Upper Air" (NCDC DSI-6309)
            ncdc6310 = NCDC "Global U/A Cards" (NCDC DSI-6310)
            ncdc6314 = Global Telecommunications System messages received
                       and processed at Roshydromet and archived at NCDC
                       (NCDC DSI-6314)
            ncdc6315 = NCDC "People's Republic of China Data" (NCDC DSI-6315)
            ncdc6316 = NCDC "Argentina National Upper Air Data" (NCDC
                       DSI-6316)
            ncdc6319 = NCDC "Korea National Upper Air Data" (NCDC DSI-6319)
            ncdc6322 = Global Telecommunications System messages received
                       at the Australian Bureau of Meteorology and
                       archived at NCDC (NCDC DSI-6322)
            ncdc6323 = NCDC "Australian U/A Thermo/Winds Merged" (NCDC
                       DSI-6323)
            ncdc6324 = NCDC "Brazil National Upper Air Data" (NCDC DSI-6324)
            ncdc6326 = NCDC "Global Upper Air Cards" (NCDC DSI-6326)
            ncdc6355 = Russian Ice Island upper air data  processed by
                       NCAR and archived at NCDC
            ncdc-gts = Global Telecommunications System (GTS) messages
                       received at NCDC from the National Centers for
                       Environmental Prediction
            ncdc-nws =  U.S. National Weather Service upper air data
                        received at NCDC in real-time
            ngdc-har = Historical Arctic radiosonde archive from the
                       National Geophysical Data Center
            usaf-ds3 = U.S. Air Force 14th Weather Squadron Upper Air
                       Data Set ( received in DS3 format)

    NP_SRC 		is the data source code for non-pressure levels in the
            sounding. These include levels whose vertical coordinate
            is only identified by height as well as surface levels without
            either pressure or height.
            NP_SRC has 15 possible values:

            cdmp-adp = "African Daily Pilot Balloon Ascent Sheets" digitized
                       by the U.S. Climate Data Modernization Program
            cdmp-awc = "African Wind Component Data" digitized from
                       "Monthly Forms" by the U.S. Climate Data
                       Modernization Program
            cdmp-us2 = "U.S. Winds Aloft digitized from "Daily Computation
                       Sheets" by the U.S. Climate Data Modernization
                       Program
            cdmp-us3 = "U.S. Winds Aloft" digitized from "Military Daily
                       Computation Sheets" by the U.S. Climate Data
                       Modernization Program
            cdmp-usm = U.S. pilot balloon observations digitized from
                       "Monthly Forms" by the U.S. Climate Data
                       Modernization Program
            chuan101 = Comprehensive Historical Upper Air Network (v1.01)
            erac-hud = ERA-CLIM Historical Upper Air Data
            mfwa-wnd = West African Winds Aloft digitized by Meteo-France
            ncdc6301 = NCDC U.S. Rawindsonde Data (NCDC DSI-6301)
            ncdc6309 = NCDC "NCAR-NMC Upper Air" (NCDC DSI-6309)
            ncdc6314 = Global Telecommunications System messages received
                       and processed at Roshydromet and archived at NCDC
                       (NCDC DSI-6314)
            ncdc-gts = Global Telecommunications System (GTS) messages
                       received at NCDC from the National Centers for
                       Environmental Prediction
            ncdc-nws =  U.S. National Weather Service upper air data
                        received at NCDC in real-time
            ngdc-har = Historical Arctic radiosonde archive from the
                       National Geophysical Data Center
            usaf-ds3 = U.S. Air Force 14th Weather Squadron Upper Air
                       Data Set (received in DS3 format)

    LAT 		is the Latitude at which the sounding was taken. For mobile
            stations, it is the latitude at the time of observation.
            For fixed stations, it is the same as the latitude shown
            in the IGRA station list regardless of the date of the
            sounding since no attempt was made to reconstruct the
            sounding-by-sounding location history of these stations.

    LON 		is the longitude at which the sounding was taken. For mobile
            stations, it is the longitude at the time of observation.
            For fixed stations, it is the same as the longitude shown
            in the IGRA station list regardless of the date of the
            sounding since no attempt was made to reconstruct the
            sounding-by-sounding location history of these stations.

    ---------------------
    Data Record Format:
    ---------------------

    -------------------------------
    Variable        Columns Type
    -------------------------------
    LVLTYP1         1-  1   Integer
    LVLTYP2         2-  2   Integer
    ETIME           4-  8   Integer
    PRESS          10- 15   Integer
    PFLAG          16- 16   Character
    GPH            17- 21   Integer
    ZFLAG          22- 22   Character
    TEMP           23- 27   Integer
    TFLAG          28- 28   Character
    RH             29- 33   Integer
    DPDP           35- 39   Integer
    WDIR           41- 45   Integer
    WSPD           47- 51   Integer
    -------------------------------

    These variables have the following definitions:

    LVLÞTYP1 	is the major level type indicator. It has the following
            three possible values:

            1 = Standard pressure level (for levels at 1000, 925, 850,
                700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30,
                20, 10, 7, 5, 3, 2, and 1 hPa)
            2 = Other pressure level
            3 = Non-pressure level

    LVLÞTYP2 	is the minor level type indicator. It has the following
            three possible values:

            1 = Surface
            2 = Tropopause
            0 = Other

    ETIME		is the elapsed time since launch. The format is MMMSS, where
            MMM represents minutes and SS represents seconds, though
            values are not left-padded with zeros. The following special
            values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.

    PRESS 		is the reported pressure (Pa or mb * 100, e.g.,
            100000 = 1000 hPa or 1000 mb). -9999 = missing.

    PFLAG 		is the pressure processing flag indicating what level of
            climatology-based quality assurance checks were applied. It
            has three possible values:

            blank = Not checked by any climatology checks. If data value
                    not equal to -9999, it passed all other applicable
                    checks.
            A     = Value falls within "tier-1" climatological limits
                    based on all days of the year and all times of day
                    at the station, but not checked by
                    "tier-2" climatology checks due to
                    insufficient data.
            B     = Value passes checks based on both the tier-1
                    climatology and a "tier-2" climatology specific to
                    the time of year and time of day of the data value.

    GPH 		is the reported geopotential height (meters above sea level).
            This value is often not available at variable-pressure levels.
            The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.

    ZFLAG 		is the  geopotential height processing flag indicating what
            level of climatology-based quality assurance checks were
            applied. It has three possible values:

            blank = Not checked by any climatology checks or flag not
                    applicable. If data value not equal to -8888 or -9999,
                    it passed all other applicable checks.
            A     = Value falls within "tier-1" climatological limits
                    based on all days of the year and all times of day
                    at the station, but not checked by
                    "tier-2" climatology checks due to insufficient data.
            B     = Value passes checks based on both the tier-1
                    climatology and a "tier-2" climatology specific to
                    the time of year and time of day of the data value.

    TEMP 		is the reported temperature (degrees C to tenths, e.g.,
            11 = 1.1°C). The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.

    TFLAG 		is the temperature processing flag indicating what
            level of climatology-based quality assurance checks were
            applied. It has three possible values:

            blank = Not checked by any climatology checks or flag not
                    applicable. If data value not equal to -8888 or -9999,
                    it passed all other applicable checks.
            A     = Value falls within "tier-1" climatological limits
                    based on all days of the year and all times of day
                    at the station, but not checked by "tier-2"
                    climatology checks due to insufficient data.
            B     = Value passes checks based on both the tier-1
                    climatology and a "tier-2" climatology specific to
                    the time of year and time of day of the data value.

    RH 		is the reported relative humidity (Percent to tenths, e.g.,
            11 = 1.1%). The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.
    DPDP 		is the reported dewpoint depression (degrees C to tenths, e.g.,
            11 = 1.1°C). The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.

    WDIR 		is the reported wind direction (degrees from north,
            90 = east). The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.

    WSPD 		is the reported wind speed (meters per second to tenths, e.g.,
            11 = 1.1 m/s). The following special values are used:

            -8888 = Value removed by IGRA quality assurance, but valid
                    data remain at the same level.
            -9999 = Value missing prior to quality assurance.
    """
    from .. import config

    if filename_pattern is None:
        filename_pattern = "*%s-data.txt.gz"

    if filename is None:
        filename = find_files(config.igradir, filename_pattern % ident)
        # filename = igradir + '/' + filename_pattern % ident
        if len(filename) > 1:
            message("Found:", ", ".join(filename), name='IRA', **kwargs)
        filename = filename[0]
        message("Using:", filename, name='IRA', **kwargs)

    if not os.path.isfile(filename):
        raise IOError("File not Found! %s" % filename)

    with gzip.open(str(filename), 'rt', encoding='utf8') as infile:
        tmp = infile.read()  # alternative readlines (slower)
        data = tmp.splitlines()  # Memory (faster)

    raw = []
    headers = []
    dates = []
    for i, line in enumerate(data):
        if line[0] == '#':
            # Header
            ident = line[1:12]
            year = line[13:17]
            month = line[18:20]
            day = line[21:23]
            hour = line[24:26]
            reltime = line[27:31]
            numlev = int(line[32:36])
            p_src = line[37:45]
            np_src = line[46:54]
            lat = int(line[55:62])/10000.
            lon = int(line[63:71])/10000.

            if int(hour) == 99:
                time = reltime + '00'
            else:
                time = hour + '0000'

            # wired stuff !?
            if '99' in time:
                time = time.replace('99', '00')

            idate = datetime.datetime.strptime(year + month + day + time, '%Y%m%d%H%M%S')
            # headers.append((idate, numlev, p_src.strip(), np_src.strip(), lat, lon))
            headers.append((idate, numlev, lat, lon))
        else:
            # Data
            lvltyp1 = int(line[0])  # 1-  1   integer
            lvltyp2 = int(line[1])  # 2-  2   integer
            etime = int(line[3:8])  # 4-  8   integer
            press = int(line[9:15])  # 10- 15   integer
            pflag = line[15]  # 16- 16   character
            gph = int(line[16:21])  # 17- 21   integer
            zflag = line[21]  # 22- 22   character
            temp = int(line[22:27]) / 10.  # 23- 27   integer
            tflag = line[27]  # 28- 28   character
            rh = int(line[28:33]) / 10.  # 30- 34   integer
            dpdp = int(line[34:39]) / 10.  # 36- 40   integer
            wdir = int(line[40:45])  # 41- 45   integer
            wspd = int(line[46:51]) / 10.  # 47- 51   integer

            # raw.append((lvltyp1, lvltyp2, etime, press, pflag, gph, zflag, temp, tflag, rh, dpdp, wdir, wspd))
            raw.append((press, gph, temp, rh, dpdp, wdir, wspd))
            dates.append(idate)

    # columns=['ltyp1', 'ltyp2', 'etime', 'pres', 'pflag', 'gph', 'zflag', 'temp', 'tflag', 'rhumi', 'dpd', 'windd', 'winds']
    out = pd.DataFrame(data=raw, index=dates, columns=['pres', 'gph', 'temp', 'rhumi', 'dpd', 'windd', 'winds'])
    out = out.replace([-999.9, -9999, -8888, -888.8], np.nan)
    out['temp'] += 273.15
    out['rhumi'] /= 100.
    out.index.name = 'date'
    headers = pd.DataFrame(data=headers, columns=['date', 'numlev', 'lat', 'lon']).set_index('date')
    return out, headers


metadata = {'temp': {'units': 'K', 'standard_name': 'air_temperature'},
            'rhumi': {'units': '1', 'standard_name': 'relative_humidity'},
            'dpd': {'units': 'K', 'standard_name': 'dew_point_depression'},
            'windd': {'units': 'degree', 'standard_name': 'wind_to_direction'},
            'winds': {'units': 'm/s', 'standard_name': 'wind_speed'}}


def open_igra_metadata(ident, filename=None):
    import pandas as pd
    # TODO: io routine for IGRA metadata
    # use default filename
    from .. import config

    file = config.get_data('igra-metadata.txt')
    contents = pd.read_csv(file)
    # search for ident in file and make a timeseries -> xarray
    return contents


__doc__ = """
Integrated Global Radiosonde Archive (IGRA) V2beta Readme File

Imke Durre (imke.durre@noaa.gov) - last updated September 2014


TABLE OF CONTENTS

I.    OVERVIEW
II.   WHAT'S NEW IN IGRA 2
III.  DOWNLOAD QUICK START
IV.   CONTENTS  OF FTP DIRECTORY
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

I. OVERVIEW

This file provides guidance for how to navigate the FTP directory for
IGRA v2beta, the beta release of version 2 of the Integrated Global
Radiosonde Archive. It provides a brief overview of what is new in IGRA 2,
step-by-step instructions for downloading desired Data and metadata,
and an explanation of the contents of the directory and all of its subdirectories.
The formats of the various types of files available are described in
separate documents.

In the context of this dataset, the designation "beta" means that all
scientific development has been completed, and the dataset is now in the
documentation and review phase that is a prerequisite for it to officially
replace IGRA 1 as NCDC's baseline upper-air dataset.

Send any feedback to Imke Durre at imke.durre@noaa.gov.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

II. WHAT'S NEW IN IGRA 2

Following is a summary of what is new and different in IGRA 2 compared to
IGRA 1.

  - More Data: IGRA 2 contains nearly twice as many stations and 30% more
    soundings than IGRA 1.

  - Longer Records: The earliest year with Data in IGRA 2 is 1905, and
    there are several hundred stations with Data before 1946. In IGRA 1,
    only one station's record extends back to before 1946, and its record
    begins in 1938.

  - Ships and Ice Islands: IGRA now contains Data from 112 floating stations,
    including 17 fixed weather ships and buoys, 72 mobile ships, and 23
    Russian ice islands.

  - Additional Variables:
    1. Reported relative humidity and time elapsed since launch are now
       provided in the sounding Data files whenever they are available. This
       allows for the inclusion of humidity observations prior to 1969,
       the first year with dewpoint depression in IGRA 1.
    2. The derived-parameter files now include both reported and calculated
       relative humidity. In soundings in which humidity is reported only
       as relative humidity, all moisture-related derived parameters are based
       on the reported relative humidity. In all other soundings, they are
       based on dewpoint depression/calculated relative humidity.
    3. In addition to monthly means of geopotential height, temperature,
       an wind, monthly means of vapor pressure are now also available.
    For details on these variables and associated changes in Data format,
    see the respective format descriptions.

  - Additional Data Sources:
    1. IGRA 2 is constructed from a total of 33 Data
       sources, including 10 of the 11 Data sources used in IGRA 1.
    2. To improve spatial coverage, Data received via the Global
       Telecommunications System (GTS) by the U.S. Air Force
       14th Weather Squadron replace the less complete NCDC/NCEP-based
       1973-1999 GTS Data which was the largest contributor of xData to
       IGRA 1. This change particularly improves the spatial coverage
       over China in the 1970s and 1980s.
    3. Daily updates now come not only from the GTS, but, for U.S. stations,
       also directly from the U.S. National Weather Service (NWS), resulting
       in higher-precision, higher-vertical resolution Data for U.S.
       stations in near real-time.
    4. Global coverage prior to the 1970s was enhanced primarily by the
       "C-Cards" and "MIT" read_datasets from the National Centers for
       Atmospheric Research as well as Version 1.01 of the Comprehensive
       Historical Upper Air Network from the Institute for Atmospheric
       and Climate Science at ETH Zurich in Switzerland.
    5. Additional Data sources include pilot balloon observations for
       the United States that were digitized under the Climate Data
       Modernization Program (CDMP), Radiosonde and pilot balloon
       observations for several countries in Africa from CDMP and
       Meteo-France, ship and ice island observations from NCDC's archive,
       Antarctic Radiosonde observations provided by the British Antarctic
       Survey, the Historical Arctic Radiosonde Archive from the National
       Geophysical Data Center, and 1990s Indonesian Radiosonde xData
       provided by the Japan Agency for Marine-Earth Science and Technology.

  - Eleven-character Station IDs: To accommodate stations other than those
    with world Meteorological Organization (WMO) station numbers, IGRA now
    uses 11-character station identifiers that consist of a two-character
    country code, a one-character station network code, and an
    eight-character station identifier.
    The station IDs for WMO stations, which account for approximately 80% of
    the IGRA 2 stations, contain a network code of "M" followed by "000"
    followed by the five-digit WMO identification number. For example, the
    IGRA 2 station identifier for Key West (WMO# 72201) is USM00072201.
    The remaining stations are identified by ship call signs (network
    code "V"), Weather Bureau, Army, Navy (WBAN) numbers ("W"),
    International Civil Aviation Organization call signs ("I"), and
    specially constructed identifiers ("X").
    For more details, see the format description of the station list.

  - Changed station list format: The order of fields in the station list
    has been changed for consistency with some of NCDC's other read_datasets. In
    addition, the identification of stations as GCOS Upper Air Network (GUAN)
    and Lanzante/Klein/Seidel (LKS) stations has been removed. Relevant LKS
    stations are captured within the RATPAC product, and the latest list of
    GUAN stations is best obtained directly from the WMO.

  - Additional Information in Sounding Headers:
    1. Header records in sounding Data files now include two xData source codes,
       one for pressure levels and one for non-pressure levels.
    2. In order to be able to indicate the position of mobile stations at
       each observation time, fields for the latitude and longitude have
       been added to the sounding headers in Data files. For fixed stations,
       including moored ships, the coordinates entered into these fields are
       always the same as those shown in the IGRA station list since the
       actual position is generally not known on a sounding-by-sounding
       basis at those stations. Coordinates are not included in the sounding
       headers of the derived-parameter files since sounding-derived
       parameters are provided only for fixed stations.
    For more details, see the format description of the Data files.

  - Soundings Without Observation Hour: Unlike IGRA 1, IGRA 2 contains
    soundings from some Data sources in which the time of day at which
    an observation was made is indicated only by the release time, i.e.,
    the time at which the balloon was launched, and the
    nominal/observation hour is missing (= 99). Since conventions for
    determining the observation hour from the release time vary over
    time and among agencies, no attempt has been made to infer the
    observation hour from the release time in IGRA 2.

  - Modified Level Type Indicators: The meaning of the first digit of
    the level type indicator in sounding records has changed as follows:

    Blank is no longer used.
    1 continues to indicate a standard pressure level.
    2 indicates a non-standard pressure level regardless of whether it
      contains thermodynamic Data or only wind xData.
    3 indicates a non-pressure level, which always only contains wind
      observations in IGRA 2.

  - Non-Pressure Surface Levels: Unlike in IGRA 1, IGRA 2 contains surface
    levels that do not contain a pressure value. Such levels only appear in
    soundings that consist entirely of Data levels whose vertical coordinate is
    identified only by height.

  - Methodological Changes:
    1. The process of choosing which Data sources contribute to each station
       record as well as the process of merging multiple Data sources into
       one station record were redesigned to increase automation, accommodate
       a greater variety of Data sources and station identifiers, and preserve
       a larger number of pilot balloon observations.
    2. In addition, some minor improvements were made to the quality assurance
       procedures, including, most notably, the addition of basic checks on
       elapsed time and relative humidity as well as improved selection of
       a single surface level within soundings in which multiple levels are
       identified as surface.
    3. The compositing procedure was redesigned. Stations are now composited
       when they are within 5 km of each other and their records do not contain
       soundings at concurrent observation times.
    All of these modifications will be described in greater detail in a
    future article.

  - Additional Station History Information:
    1. The IGRA metadata file, which contains documented information about
       the instrumentation and observing practices at many stations, has been
       augmented with additional records extracted from the Gaffen (1996)
       collection that formed the basis for the original IGRA metadata.
       The additional records are for nearly 700 IGRA 2 stations for
       which no Data was available in IGRA 1.
    2. To provide information on instrumentation used in recent years for which
       documented station history information is not available in the IGRA
       IGRA metadata file, the WMO Radiosonde/sounding system and measuring
       equipment codes contained in Global Telecommunications System messages
       are also supplied in separate files for the years 2000-2013. Note that
       these codes have not been checked for accuracy and are provided
       as received.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

III. DOWNLOAD QUICK START

http://www1.ncdc.noaa.gov/pub/Data/igra/v2beta/ .

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""
