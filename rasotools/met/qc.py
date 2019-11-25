# -*- coding: utf-8 -*-


__all__ = ['temperature', 'specific_humidity', 'relative_humidity', 'water_vapor_pressure', 'dew_point_depression']


def temperature(data, dim='pres', return_flags=False, **kwargs):
    """ Quality control of Temperatures with RTTOV Profile limits

    Args:
        data (DataArray): temperatures [K]
        dim (str): pressure
        return_flags (bool): return Mask
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
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
    # for i, idim in enumerate(dataset.dims):
    #     if idim == dim:
    #         continue
    #     tmins = np.expand_dims(tmins, axis=i)

    tmins = np.broadcast_to(tmins, data.values.shape)
    tmaxs = np.interp(np.log(pressure), np.log(pin), tmax, left=tmax.min(), right=tmax.max())
    # for i, idim in enumerate(dataset.dims):
    #     if idim == dim:
    #         continue
    #     tmaxs = np.expand_dims(tmaxs, axis=i)
    tmaxs = np.broadcast_to(tmaxs, data.values.shape)
    with np.errstate(invalid='ignore'):
        logic = (data.values < tmins) | (data.values > tmaxs)

    tcor = np.sum(logic)  # number of corrected values
    message("Temperatures %d [%d]" % (tcor, np.sum(np.isfinite(data.values))), **kwargs)
    if return_flags:
        return DataArray(data=logic, coords=data.coords, dims=data.dims,
                         name=data.name+'_qcflag',
                         attrs={'QC': "RTTOV Profile limits (%d)" % tcor})

    data = data.copy()
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = "RTTOV Profile limits (%d)" % tcor
    return data


def specific_humidity(data, dim='pres', return_logic=False, **kwargs):
    """ Quality control of spec. humidity with RTTOv Profile limits

    Args:
        data (DataArray): specifc humidity [kg/kg]
        dim (str): pressure
        return_logic (bool): return Mask
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
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
    message("Spec. Humidity %d [%d]" % (qcor, np.sum(np.isfinite(data.values))), **kwargs)
    if return_logic:
        return logic

    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = "RTTOV Profile limits (%d)" % qcor
    return data


def relative_humidity(data, return_logic=False, **kwargs):
    """Quality control of rel. humidity with RTTOV Profile limits

    Args:
        data (DataArray): rel. humidity [1]
        return_logic (bool): return Mask
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
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
    r_absmax = 1
    # Valid Range: [ 0; 1] ratio
    with np.errstate(invalid='ignore'):
        logic = (data.values < r_absmin) | (data.values > r_absmax)

    rcor = np.sum(logic)  # number of corrected values
    message("Rel. Humidity %d [%d]" % (rcor, np.sum(np.isfinite(data.values))), **kwargs)
    if return_logic:
        return logic
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = "RTTOV Profile limits (%d)" % rcor
    return data


def dew_point_depression(data, return_logic=False, **kwargs):
    """Quality control of dewpoint depression with RTTOV Profile limits

    Args:
        data (DataArray): dewpoint depression [K]
        return_logic (bool): return Mask
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
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
    dpd_absmax = 60
    # Valid Range: [0, 60] K
    with np.errstate(invalid='ignore'):
        logic = (data.values < dpd_absmin) | (data.values > dpd_absmax)

    dpdcor = np.sum(logic)  # number of corrected values
    message("Dew point depression %d [%d]" % (dpdcor, np.sum(np.isfinite(data.values))), **kwargs)
    if return_logic:
        return logic

    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = "RTTOV Profile limits (%d)" % dpdcor
    return data


def water_vapor_pressure(data, dim='pres', return_logic=False, **kwargs):
    """ Quality control of water vapor pressure with RTTOV Profile limits

    Args:
        data (DataArray): water vapor [Pa]
        dim (str): pressure
        **kwargs:

    Returns:
        DataArray : quality controlled dataset
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
        if data.attrs['units'] != 'K':
            raise RuntimeWarning("Water vapor are not in Pa")

    # Valid Range: [183.15, 333.15] K
    rt = config.rttov_profile_limits
    pressure = data[dim].values
    vpmin = ppmv2pa(rt.Water_vapour_min.values, rt.Pressure.values)
    vpmax = ppmv2pa(rt.Water_vapour_max.values, rt.Pressure.values)
    pin = None
    if 'units' in data[dim].attrs:
        if data[dim].attrs['units'] == 'hPa':
            pin = rt.Pressure.values  # hPa to hPa
    if pin is None:
        pin = rt.Pressure.values * 100.  # hPa to Pa

    vpmins = np.interp(np.log(pressure), np.log(pin), vpmin, left=vpmin.min(), right=vpmin.max())
    vpmins = np.broadcast_to(vpmins, data.values.shape)
    vpmaxs = np.interp(np.log(pressure), np.log(pin), vpmax, left=vpmax.min(), right=vpmax.max())
    vpmaxs = np.broadcast_to(vpmaxs, data.values.shape)
    with np.errstate(invalid='ignore'):
        logic = (data.values < vpmins) | (data.values > vpmaxs)

    vpcor = np.sum(logic)  # number of corrected values
    message("Water vapor pressure %d [%d]" % (vpcor, np.sum(np.isfinite(data.values))), **kwargs)
    if return_logic:
        return logic
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = "RTTOV Profile limits (%d)" % vpcor
    return data

