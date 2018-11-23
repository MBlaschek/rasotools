# -*- coding: utf-8 -*-
import numpy as np
from ..fun import message

__all__ = ['temperature', 'specific_humidity', 'relative_humidity', 'water_vapor_pressure', 'dew_point_depression',
           'dew_point', 'profile_limits']


def temperature(data, pressure, axis=1, return_logic=False, **kwargs):
    rt = profile_limits(tohpa=True, simple_names=True)
    rt['p'] *= 100.  # hPa to Pa
    # Valid Range: [183.15, 333.15] K
    if not isinstance(data, np.ndarray):
        raise ValueError("requires a numpy array")

    if not np.all((np.sort(pressure) == pressure)):
        message("Pressure levels not sorted, applying to data", **kwargs)

    index = [np.newaxis if (iname != pressure) else slice(None, None) for iname, idim in data.dims.items()]
    tmins = np.interp(np.log(pin), np.log(rt.p.values), rt.tmin.values, left=rt.tmin.min(), right=rt.tmin.max())
    tmaxs = np.interp(np.log(pin), np.log(rt.p.values), rt.tmax.values, left=rt.tmax.min(), right=rt.tmax.max())
    with np.errstate(invalid='ignore'):
        logic = (data.values < tmins[index]) | (data.values > tmaxs[index])
    tcor = np.sum(logic)  # number of corrected values
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = tcor
    message(funcid + "Temperatures %d [%d]" % (tcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data


def specific_humidity(data, return_logic=False, verbose=0):
    import numpy as np

    from xData import DataArray
    from ..met.humidity import vap2sh
    from ..tools import message

    funcid = "[QC] Q "
    q_absmin = 0
    q_absmax = 1
    ###################
    rt = profile_limits(tohpa=True, simple_names=True)
    rt['p'] *= 100.  # hPa to Pa
    # Valid Range: [0, 1]  kg/kg
    if not isinstance(data, DataArray):
        raise ValueError(funcid + "requires a DataArray class object")

    pressure = data.get_dimension_by_axis('Z')
    data = data.copy()
    pin = data.dims[pressure].values
    index = [np.newaxis if (iname != pressure) else slice(None, None) for iname, idim in data.dims.items()]
    rt['qmin'] = vap2sh(rt.vpmin.values, rt.p.values)  # both missing hPa factor
    rt['qmax'] = vap2sh(rt.vpmax.values, rt.p.values)  # both missing hPa factor

    qmins = np.interp(np.log(pin), np.log(rt.p.values), rt.tmin.values, left=rt.qmin.min(), right=rt.qmin.max())
    qmaxs = np.interp(np.log(pin), np.log(rt.p.values), rt.tmax.values, left=rt.qmax.min(), right=rt.qmax.max())
    with np.errstate(invalid='ignore'):
        logic = (data.values < qmins[index]) | (data.values > qmaxs[index])
    # logic = (data.values < q_absmin) | (data.values > q_absmax)  # Alterate formulation
    qcor = np.sum(logic)  # number of corrected values
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = qcor
    message(funcid + "spec. Humidity %d [%d]" % (qcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data


def relative_humidity(data, return_logic=False, verbose=0, **kwargs):
    import numpy as np

    from xData import DataArray
    from ..tools import message

    funcid = "[QC] R "
    r_absmin = kwargs.pop('absmin', 0)
    r_absmax = kwargs.pop('absmax', 1)
    # Valid Range: [ 0; 1] ratio
    if not isinstance(data, DataArray):
        raise ValueError(funcid + "requires a DataArray class object")

    data = data.copy()
    with np.errstate(invalid='ignore'):
        logic = (data.values < r_absmin) | (data.values > r_absmax)

    rcor = np.sum(logic)  # number of corrected values
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = rcor
    message(funcid + "rel. Humidity %d [%d]" % (rcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data


def dew_point_depression(data, return_logic=False, verbose=0, **kwargs):
    import numpy as np

    from xData import DataArray
    from ..tools import message

    funcid = "[QC] DPD "
    dpd_absmin = 0
    dpd_absmax = 60
    # Valid Range: [0, 60] K
    if not isinstance(data, DataArray):
        raise ValueError(funcid + "requires a DataArray class object")

    data = data.copy()
    with np.errstate(invalid='ignore'):
        logic = (data.values < dpd_absmin) | (data.values > dpd_absmax)

    dpdcor = np.sum(logic)  # number of corrected values
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = dpdcor
    message(funcid + "Dew point depression %d [%d]" % (dpdcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data


def dew_point(data, return_logic=False, temperatures=None, verbose=0, **kwargs):
    import numpy as np

    from xData import DataArray
    from ..tools import message

    funcid = "[QC] Td "
    dpd_absmax = 60
    ###################
    rt = profile_limits(tohpa=True, simple_names=True)
    rt['p'] *= 100.  # hPa to Pa
    # Valid Range: [183 - 333] K & < T
    if not isinstance(data, DataArray):
        raise ValueError(funcid + "requires a DataArray class object")

    data = data.copy()
    with np.errstate(invalid='ignore'):
        logic = (data.values < (rt.tmin.min() - dpd_absmax)) | (data.values > rt.tmax.max())

    tdcor = np.sum(logic)
    if temperatures is not None:
        if not isinstance(temperatures, DataArray):
            raise ValueError(funcid + "requires a DataArray class object")
        if temperatures.values.shape == data.values.shape:
            logic = logic | (data > temperatures.values)
            tdtcor = np.sum(logic)
            data.attrs['QC_T'] = tdtcor
            message(funcid + "Dew point (Temperatures) %d [%d]" % (tdtcor, np.sum(np.isfinite(data.values))),
                    verbose)

    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = tdcor
    message(funcid + "Dew point %d [%d]" % (tdcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data


def water_vapor_pressure(data, return_logic=False, verbose=0, **kwargs):
    import numpy as np

    from xData import DataArray
    from ..tools import message

    funcid = "[QC] Vp "
    rt = profile_limits(tohpa=True, simple_names=True)
    rt['p'] *= 100.  # hPa to Pa
    # Valid Range: [0, 20000] Pa
    if not isinstance(data, DataArray):
        raise ValueError(funcid + "requires a DataArray class object")

    pressure = data.get_dimension_by_axis('Z')
    data = data.copy()
    pin = data.dims[pressure].values
    index = [np.newaxis if (iname != pressure) else slice(None, None) for iname, idim in data.dims.items()]
    vpmins = np.interp(np.log(pin), np.log(rt.p.values), rt.vpmin.values, left=rt.vpmin.min(), right=rt.vpmin.max())
    vpmaxs = np.interp(np.log(pin), np.log(rt.p.values), rt.vpmax.values, left=rt.vpmax.min(), right=rt.vpmax.max())
    with np.errstate(invalid='ignore'):
        logic = (data.values < vpmins[index]) | (data.values > vpmaxs[index])
    vpcor = np.sum(logic)  # number of corrected values
    data.values = np.where(logic, np.nan, data.values)  # replace
    data.attrs['QC'] = vpcor
    message(funcid + "Water vapor pressure %d [%d]" % (vpcor, np.sum(np.isfinite(data.values))), verbose)
    if return_logic:
        return logic
    return data

#
# Support function
#


def profile_limits(tohpa=False, simple_names=False):
    """ Get RTTOV v11 Profile Limits
    Pressure (hPa)
    Temperature min (K)
    Temperature max (K)
    Water vapour min (ppmv) -> topa > hPa
    Water vapour max (ppmv)
    Ozone min (ppmv)
    Ozone max (ppmv)
    CO2 min (ppmv)
    CO2 max (ppmv)

    Returns
    -------
    rttov table
    """
    from ..met.humidity import ppmv2pa
    from .. import config

    if config.rttov_profile_limits is None:
        raise RuntimeError('No RTTOV Profile limits present. Check Module Data')

    rttov_profile_limits = config.rttov_profile_limits.copy()
    if tohpa:
        for ivar in rttov_profile_limits.columns[rttov_profile_limits.columns.str.contains('ppmv')].tolist():
            rttov_profile_limits.loc[:, ivar] = ppmv2pa(rttov_profile_limits[ivar],
                                                        rttov_profile_limits[u'Pressure (hPa)'])  # ppmv to pa

        rttov_profile_limits.columns = rttov_profile_limits.columns.str.replace('ppmv', 'hPa')  # rename

    if simple_names:
        rttov_profile_limits.columns = ['p', 'tmin', 'tmax', 'vpmin', 'vpmax', 'omin', 'omax', 'co2min', 'co2max']

    return rttov_profile_limits
