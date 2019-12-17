# -*- coding: utf-8 -*-


__all__ = ['temperature', 'specific_humidity', 'relative_humidity', 'water_vapor_pressure',
           'dew_point_depression']


def temperature(data, dim='plev', flags_only=False, report=True, **kwargs):
    """ Quality control of Temperatures with RTTOV Profile limits

    Args:
        data (DataArray): temperatures [K]
        dim (str): pressure
        flags_only (bool): return only flags
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
        DataArray : flags [True: rejected, False: accepted]
    """
    import numpy as np
    from xarray import DataArray
    from .. import config
    from ..fun import message, load_rttov

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray, ", type(data))

    if config.rttov_profile_limits is None:
        load_rttov()

    if 'units' in data.attrs:
        if data.attrs['units'] != 'K':
            raise RuntimeWarning("Temperature are not in Kelvin")

    # Valid Range: [183.15, 333.15] K
    rt = config.rttov_profile_limits
    pressure = data[dim].values
    tmin = rt.Temperature_min.values
    tmax = rt.Temperature_max.values
    pin = None
    if 'units' in data[dim].attrs:
        if data[dim].attrs['units'] == 'hPa':
            pin = rt.Pressure.values  # hPa to hPa
    if pin is None:
        pin = rt.Pressure.values * 100.  # hPa to Pa

    tmins = np.interp(np.log(pressure), np.log(pin), tmin, left=tmin.min(), right=tmin.max())
    tmins = np.broadcast_to(tmins, data.values.shape)
    tmaxs = np.interp(np.log(pressure), np.log(pin), tmax, left=tmax.min(), right=tmax.max())
    tmaxs = np.broadcast_to(tmaxs, data.values.shape)
    with np.errstate(invalid='ignore'):
        logic = (data.values < tmins) | (data.values > tmaxs)

    tcor = np.sum(logic)  # number of corrected values
    message("Temperatures %d [%d]" % (int(tcor), int(np.sum(np.isfinite(data.values)))), **kwargs)
    if 'QC' in data.attrs.keys():
        tcor += data.attrs['QC']

    if data.name is None:
        data.name = 't'
    # 0 no_qc 1 good_data 2 outside_range 3 dpd30
    flags = DataArray(data=np.zeros_like(data.values), coords=data.coords, dims=data.dims,
                      name=data.name + '_qc',
                      attrs={'QC': tcor, 'flag_values': (0, 1, 2, 3),
                             'flag_meanings': "no_qc good_data outside_range dpd30",
                             'valid_range': (0, 3)})
    flags.values[~logic] = 1  # good data
    flags.values[logic] = 2  # outside range

    if report:
        rep = qcreport(flags, data=data, min=tmins, max=tmaxs)

    if flags_only:
        if report:
            return flags, rep
        return flags

    data = data.copy()
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = tcor
    data.attrs['ancilliary_variables'] = data.name + '_qc'
    if report:
        return data, flags, rep
    return data, flags


def specific_humidity(data, dim='plev', flags_only=False, report=True, **kwargs):
    """ Quality control of spec. humidity with RTTOv Profile limits

    Args:
        data (DataArray): specifc humidity [kg/kg]
        dim (str): pressure
        flags_only (bool): return only flags
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
        DataArray : flags [True: rejected, False: accepted]
    """
    import numpy as np
    from xarray import DataArray
    from .humidity import vap2sh
    from .. import config
    from ..fun import message, load_rttov

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray, ", type(data))

    if config.rttov_profile_limits is None:
        load_rttov()

    if 'units' in data.attrs:
        if data.attrs['units'] != 'kg/kg':
            raise RuntimeWarning("specific humidity is not in kg/kg")

    # Valid Range: [0, 1]  kg/kg
    rt = config.rttov_profile_limits
    pressure = data[dim].values
    qmin = vap2sh(rt.Water_vapour_min.values, rt.Pressure.values)
    qmax = vap2sh(rt.Water_vapour_max.values, rt.Pressure.values)
    pin = None
    if 'units' in data[dim].attrs:
        if data[dim].attrs['units'] == 'hPa':
            pin = rt.Pressure.values  # hPa to hPa
    if pin is None:
        pin = rt.Pressure.values * 100.  # hPa to Pa

    qmins = np.interp(np.log(pressure), np.log(pin), qmin, left=qmin.min(), right=qmin.max())
    qmins = np.broadcast_to(qmins, data.values.shape)
    qmaxs = np.interp(np.log(pressure), np.log(pin), qmax, left=qmax.min(), right=qmax.max())
    qmaxs = np.broadcast_to(qmaxs, data.values.shape)
    with np.errstate(invalid='ignore'):
        logic = (data.values < qmins) | (data.values > qmaxs)

    qcor = np.sum(logic)  # number of corrected values
    message("Spec. Humidity %d [%d]" % (int(qcor), int(np.sum(np.isfinite(data.values)))), **kwargs)
    if 'QC' in data.attrs.keys():
        qcor += data.attrs['QC']

    if data.name is None:
        data.name = 'q'

    # 0 no_qc 1 good_data 2 outside_range 3 dpd30
    flags = DataArray(data=np.zeros_like(data.values), coords=data.coords, dims=data.dims,
                      name=data.name + '_qc',
                      attrs={'QC': qcor, 'flag_values': (0, 1, 2, 3),
                             'flag_meanings': "no_qc good_data outside_range dpd30",
                             'valid_range': (0, 3)})
    flags.values[~logic] = 1  # good data
    flags.values[logic] = 2  # outside range

    if report:
        rep = qcreport(flags, data=data, min=qmins, max=qmaxs)

    if flags_only:
        if report:
            return flags, rep
        return flags

    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = qcor
    data.attrs['ancilliary_variables'] = data.name + '_qc'
    if report:
        return data, flags, rep
    return data, flags


def relative_humidity(data, flags_only=False, report=True, **kwargs):
    """Quality control of rel. humidity with RTTOV Profile limits

    Args:
        data (DataArray): rel. humidity [1]
        flags_only (bool): return only flags
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
        DataArray : flags [True: rejected, False: accepted]
    """
    import numpy as np
    from xarray import DataArray
    from ..fun import message

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray, ", type(data))

    if 'units' in data.attrs:
        if data.attrs['units'] != '1':
            raise RuntimeWarning("rel. humidity has units 1 (ratio)")

    r_absmin = 0
    r_absmax = 1.03  # 3 % plus
    # Valid Range: [ 0; 1] ratio
    with np.errstate(invalid='ignore'):
        logic = (data.values < r_absmin) | (data.values > r_absmax)

    rcor = np.sum(logic)  # number of corrected values
    message("Rel. Humidity %d [%d]" % (int(rcor), int(np.sum(np.isfinite(data.values)))), **kwargs)
    if 'QC' in data.attrs.keys():
        rcor += data.attrs['QC']

    if data.name is None:
        data.name = 'rh'

    # 0 no_qc 1 good_data 2 outside_range 3 dpd30
    flags = DataArray(data=np.zeros_like(data.values), coords=data.coords, dims=data.dims,
                      name=data.name + '_qc',
                      attrs={'QC': rcor, 'flag_values': (0, 1, 2, 3),
                             'flag_meanings': "no_qc good_data outside_range dpd30",
                             'valid_range': (0, 3)})
    flags.values[~logic] = 1  # good data
    flags.values[logic] = 2  # outside range
    if report:
        rep = qcreport(flags, data=data, absmin=r_absmin, absmax=r_absmax)

    if flags_only:
        if report:
            return flags, rep
        return flags

    data = data.copy()
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = rcor
    data.attrs['ancilliary_variables'] = data.name + '_qc'
    if report:
        return data, flags, rep
    return data, flags


def dew_point_depression(data, flags_only=False, report=True, **kwargs):
    """Quality control of dewpoint depression with RTTOV Profile limits

    Args:
        data (DataArray): dewpoint depression [K]
        flags_only (bool): return only flags
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
        DataArray : flags [True: rejected, False: accepted]
    """
    import numpy as np
    from xarray import DataArray
    from ..fun import message

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray, ", type(data))

    if 'units' in data.attrs:
        if data.attrs['units'] != 'K':
            raise RuntimeWarning("Dewpoint depression is not in Kelvin")

    dpd_absmin = 0
    dpd_absmax = 80
    # Valid Range: [0, 80] K
    with np.errstate(invalid='ignore'):
        logic = (data.values < dpd_absmin) | (data.values > dpd_absmax)

    dpdcor = np.sum(logic)  # number of corrected values
    message("Dew point depression %d [%d]" % (int(dpdcor), int(np.sum(np.isfinite(data.values)))), **kwargs)
    if 'QC' in data.attrs.keys():
        dpdcor += data.attrs['QC']

    if data.name is None:
        data.name = 'dpd'

    # 0 no_qc 1 good_data 2 outside_range 3 dpd30
    flags = DataArray(data=np.zeros_like(data.values), coords=data.coords, dims=data.dims,
                      name=data.name + '_qc',
                      attrs={'QC': dpdcor, 'flag_values': (0, 1, 2, 3),
                             'flag_meanings': "no_qc good_data outside_range dpd30",
                             'valid_range': (0, 3)})
    flags.values[~logic] = 1  # good data
    flags.values[logic] = 2  # outside range
    if report:
        rep = qcreport(flags, data=data, absmin=dpd_absmin, absmax=dpd_absmax)

    if flags_only:
        if report:
            return flags, rep
        return flags

    data = data.copy()
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = dpdcor
    data.attrs['ancilliary_variables'] = data.name + '_qc'
    if report:
        return data, flags, rep
    return data, flags


def water_vapor_pressure(data, dim='plev', flags_only=False, report=True, **kwargs):
    """ Quality control of water vapor pressure with RTTOV Profile limits

    Args:
        data (DataArray): water vapor [Pa]
        dim (str): pressure
        flags_only (bool): return only flags
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
        DataArray : flags [True: rejected, False: accepted]
    """
    import numpy as np
    from xarray import DataArray
    from .humidity import ppmv2pa
    from .. import config
    from ..fun import message, load_rttov

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray, ", type(data))

    if config.rttov_profile_limits is None:
        load_rttov()

    if 'units' in data.attrs:
        if data.attrs['units'] != 'Pa':
            raise RuntimeWarning("Water vapor are not in Pa")

    # Valid Range: [183.15, 333.15] K
    rt = config.rttov_profile_limits
    pressure = data[dim].values
    pfactor = 100  # Pa
    u1 = 'Pa'
    if 'units' in data[dim].attrs:
        u1 = data.attrs['units']
        if u1 == 'hPa':
            pfactor = 1

    pin = rt.Pressure.values * pfactor
    #
    # Check Water vapor Unit [hPa or Pa] ?
    #
    dfactor = 1  # Pa
    u2 = 'Pa'
    if 'units' in data.attrs:
        u2 = data.attrs['units']
        if u2 == 'hPa':
            dfactor = 0.01
    message("VP [{}]: {} , P [{}]: {}".format(u2, dfactor, u1, pfactor), **kwargs)
    #
    # Pressure is in hPa -> pfactor
    # Water vapor can be hPa or Pa -> dfactor
    #
    vpmin = ppmv2pa(rt.Water_vapour_min.values, rt.Pressure.values * pfactor) * dfactor
    vpmax = ppmv2pa(rt.Water_vapour_max.values, rt.Pressure.values * pfactor) * dfactor
    #
    #
    #
    vpmins = np.interp(np.log(pressure), np.log(pin), vpmin, left=vpmin.min(), right=vpmin.max())
    vpmins = np.broadcast_to(vpmins, data.values.shape)
    vpmaxs = np.interp(np.log(pressure), np.log(pin), vpmax, left=vpmax.min(), right=vpmax.max())
    vpmaxs = np.broadcast_to(vpmaxs, data.values.shape)
    with np.errstate(invalid='ignore'):
        logic = (data.values < vpmins) | (data.values > vpmaxs)

    vpcor = np.sum(logic)  # number of corrected values
    message("Water vapor pressure %d [%d]" % (int(vpcor), int(np.sum(np.isfinite(data.values)))), **kwargs)
    if 'QC' in data.attrs.keys():
        vpcor += data.attrs['QC']

    if data.name is None:
        data.name = 'vp'
    # 0 no_qc 1 good_data 2 outside_range 3 dpd30
    flags = DataArray(data=np.zeros_like(data.values), coords=data.coords, dims=data.dims,
                      name=data.name + '_qc',
                      attrs={'QC': vpcor, 'flag_values': (0, 1, 2, 3),
                             'flag_meanings': "no_qc good_data outside_range dpd30",
                             'valid_range': (0, 3)})
    flags.values[~logic] = 1  # good data
    flags.values[logic] = 2  # outside range
    if report:
        rep = qcreport(flags, data=data, min=vpmins, max=vpmaxs)

    if flags_only:
        if report:
            return flags, rep
        return flags

    data = data.copy()
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = vpcor
    data.attrs['ancilliary_variables'] = data.name + '_qc'
    if report:
        return data, flags, rep
    return data, flags


def qcreport(flags, iflag=2, data=None, min=None, max=None, absmin=None, absmax=None):
    from numpy import where, char, array, datetime64
    # search data (logic), produce a list of flagged values
    events = []
    order = list(flags.dims)
    indices = where(flags == iflag)
    if indices[0].size > 0:
        events = array("[QC] " + flags.name + " | (")
        for i, idim in enumerate(order):
            if isinstance(flags[idim].values[0], datetime64):
                txt = idim + "='{}'"
            else:
                txt = idim + "={}"
            txt += ',' if i < len(order) - 1 else ')'
            events = char.add(events, array(list(map(txt.format, flags[idim].values[indices[i]]))))

        if data is not None:
            if min is not None:
                events = char.add(events, array(list(map(" | {:.4f} < ".format, min[indices]))))
            elif absmin is not None:
                events = char.add(events, array(" | {} <".format(absmin)))
            else:
                events = char.add(events, array(" | "))
            if data is not None:
                events = char.add(events, array(list(map("{:.4f}".format, data.values[indices]))))
            if max is not None:
                events = char.add(events, array(list(map(" < {:.4f}".format, max[indices]))))
            if absmax is not None:
                events = char.add(events, array("> {}".format(absmax)))
        events = list(events)
    return events

#
# def report_object(flags, data, min=None, max=None, absmin=None, absmax=None):
#     from numpy import where
#     from xarray import Dataset
#     events = Dataset()
#     indices = where(flags > 1)
#     if indices[0].size > 0:
#         for i, idim in enumerate(flags.dims):
#             events[idim] = ('event', flags[idim].values[indices[i]])
#         events = events.set_coords(flags.dims)
#         events[data.name] = ('event', data.values[indices])  # this might be an xarray object
#         if min is not None:
#             events['min'] = ('event', min[indices])
#         if max is not None:
#             events['max'] = ('event', max[indices])
#         if absmin is not None:
#             events['absmin'] = ('event', [absmin] * events.event.size)
#         if absmax is not None:
#             events['absmax'] = ('event', [absmax] * events.event.size)
#
#     return events
#
#
# def report_object_to_string(da):
#     from numpy import datetime64
#     coords = list(da.coords)
#     cformat = "[QC] | sel("
#     for i, idim in enumerate(coords):
#         if isinstance(da[idim].values[0], datetime64):
#             cformat += "%s='{%s}'" % (idim, idim)
#         else:
#             cformat += "%s={%s}" % (idim, idim)
#         if i < (len(coords) - 1):
#             cformat += ','
#     cformat += ") | "
#     varis = list(da.data_vars)
#     for i in varis:
#         if 'min' in i:
#             imin = i
#         elif 'max' in i:
#             imax = i
#         else:
#             ivar = i
#     if 'abs' in imin:
#         cformat += "{%s} < {%s:.2e} < {%s}" % (imin, ivar, imax)
#     else:
#         cformat += "{%s:.2e} < {%s:.2e} < {%s:.2e}" % (imin, ivar, imax)
# isonde.data.exp001_dpd_era5_qc_reports.to_dataframe().apply(lambda x: "QC sel(hour={hour}, time='{time}', plev={plev}) | {absmin} < {dpd:.2e} < {absmax}".format(**dict(x)), axis=1)
#     events = da.to_dataframe().apply(lambda x: cformat.format(**dict(x)), axis=1)
#     return events.to_list()
