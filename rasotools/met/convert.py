# -*- coding: utf-8 -*-

__all__ = ['to_rh', 'to_vp', 'to_dpd', 'to_sh', 'to_dewpoint']


def to_rh(temp, dpd=None, spec_humi=None, press=None, method='HylandWexler', precision=6, **kwargs):
    """
    Convert dewpoint departure or specific humidity to relative humidity

    1. DPD to RH (T):
        + DPD to VP, T to VPsat
        + VP / VPsat

    2. Q to RH (Q, T, P):
        + Q to VP, T to VPsat
        + VP / VPsat

    Args:
        temp (DataArray): temperature
        dpd (DataArray): dewpoint depression
        spec_humi (DataArray): specific humidity
        press (DataArray,str): air pressure or dim name
        method (str): Saturation water vapor pressure formulation
        precision (int): decimal precision

    Returns:
        DataArray : relative humidity [1]
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap
    from ..fun import message, leveldown

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray", type(temp))

    if dpd is None and spec_humi is None:
        raise RuntimeError("Requires either dpd or q for conversion")

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError("Requires a DataArray", type(dpd))

    if spec_humi is not None:
        if not isinstance(spec_humi, DataArray):
            raise ValueError("Requires a DataArray", type(spec_humi))

        if press is None:
            raise RuntimeError("Conversion Q>RH requires a pressure variable as well")

    rvar = temp.copy()
    if press is not None:
        if isinstance(press, str):
            if press in rvar.dims:
                press = rvar[press].values
                press = _conform(press, rvar.values.shape)
        elif isinstance(press, DataArray):
            press = press.values
        else:
            pass

    if dpd is not None:
        # DPD to RH
        vpdata = svp(temp.values - dpd.values, method=method, p=press, **kwargs)
        rvar.values = vpdata / svp(temp.values, method=method, p=press, **kwargs)
        origin = 't,dpd' if press is None else 't,dpd,p'
    else:
        vpdata = sh2vap(spec_humi.values, press)
        rvar.values = vpdata / svp(temp.values, method=method, p=press, **kwargs)
        origin = 't,q,p'

    r_att = {'units': '1', 'standard_name': 'relativ_humidity', 'long_name': 'relative humidity',
             'esat': method, 'origin': origin}

    if press is not None:
        r_att['enhancement_factor'] = "yes"

    if ((rvar.values < 0) | (rvar.values > 1)).any():
        message("Warning relative humidiy outside of normal range", **leveldown(**kwargs))

    r_att['precision'] = precision
    rvar.attrs.update(r_att)
    rvar.values = around(rvar.values, decimals=precision)
    rvar.name = 'rh'
    return rvar


def to_vp(temp, dpd=None, td=False, rel_humi=None, spec_humi=None, press=None, method='HylandWexler', precision=9,
          **kwargs):
    """ Convert dewpoint departure, rel. humidity or specific humidity to water vapor pressure

    1. DPD to VP (T)
    2. Td to VP
    3. RH to VP (T)
    4. Q to VP (T, P)

    Args:
        temp (DataArray): temperature (input)
        rel_humi (DataArray):   rel. humidity (input)
        dpd:   Name of DPD (Input)
        td (bool): if temp is dewpoint or temperature
        spec_humi:  Name of Q (Input)
        press:   Name of pressure (Input)
        method:  Saturation water vapor pressure formulation

    Returns
    -------
        DataArray : water vapor pressure [Pa]
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray", type(temp))

    if dpd is None and spec_humi is None and rel_humi is None and not td:
        raise RuntimeError("Requires either dpd, td, q or r for conversion")

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError("DPD Requires a DataArray", type(dpd))

    if spec_humi is not None and not isinstance(spec_humi, DataArray):
        raise ValueError("Q Requires a DataArray", type(spec_humi))

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError("R Requires a DataArray", type(rel_humi))

    if spec_humi is not None:
        if press is None:
            raise RuntimeError("Conversion Q>VP requires a pressure variable as well")

    vpvar = temp.copy()
    for iatt in list(vpvar.attrs.keys()):
        del vpvar.attrs[iatt]

    if press is not None:
        if isinstance(press, str):
            if press in vpvar.dims:
                press = vpvar[press].values
                press = _conform(press, vpvar.values.shape)
        elif isinstance(press, DataArray):
            press = press.values
        else:
            pass

    if dpd is not None:
        # DPD to VP
        vpvar.values = svp(temp.values - dpd.values, method=method, p=press)
        origin = 't,dpd' if press is None else 't,dpd,p'
    elif td:
        # Td to VP
        vpvar.values = svp(temp.values, method=method, p=press)
        origin = 'td' if press is None else 'td,p'

    elif rel_humi is not None:
        # RH to VP
        vpvar.values = svp(temp.values, method=method, p=press) * rel_humi.values
        origin = 't,rh' if press is None else 't,rh,p'

    else:
        # Q to VP
        vpvar.values = sh2vap(spec_humi.values, press)
        origin = 'q,p'

    r_att = {'origin': origin, 'esat': method, 'standard_name': 'water_vapor_pressure',
             'long_name': 'water vapor pressure', 'units': 'Pa'}

    if press is not None:
        r_att['enhancement_factor'] = "yes"

    r_att['precision'] = precision
    vpvar.attrs.update(r_att)
    vpvar.values = around(vpvar.values, decimals=precision)
    vpvar.name = 'vp'
    return vpvar


def to_dpd(temp, rel_humi=None, vp=None, spec_humi=None, press=None, svp_method='HylandWexler',
           dewp_method='dewpoint_Boegel', precision=2, **kwargs):
    """ Convert relative humidity, specific humidity or water vapor pressure to dewpoint departure

    1. RH to DPD (T):
        + RH to VP (T)
        + VP to DPD (T)

    2. VP to DPD (T):
        + VP to DPD (T)

    3. Q to DPD (T, P):
        + Q to VP (P)
        + VP to DPD (T)

    Args:
        temp: Name of temperature (input)
        rel_humi: Name of rel. humidity (Input)
        vp: Name of DPD (Input)
        spec_humi: Name of Q (Input)
        press: Name of pressure (Input)
        svp_method: Saturation water vapor pressure formulation
        dewp_method:
        precision (int)

    Returns:
        DataArray: dewpoint depression [K]
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap, dewpoint
    from ..fun import message, leveldown

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray", type(temp))

    if rel_humi is None and spec_humi is None and vp is None:
        raise RuntimeError("Requires either r, q or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError("R Requires a DataArray", type(rel_humi))

    if spec_humi is not None and not isinstance(spec_humi, DataArray):
        raise ValueError("Q Requires a DataArray", type(spec_humi))

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError("VP Requires a DataArray", type(vp))

    if spec_humi is not None:
        if press is None:
            raise RuntimeError("Conversion Q>DPD requires a pressure variable as well")

    dpdvar = temp.copy()
    for iatt in list(dpdvar.attrs.keys()):
        del dpdvar.attrs[iatt]

    if press is not None:
        if isinstance(press, str):
            if press in dpdvar.dims:
                press = dpdvar[press].values
                press = _conform(press, dpdvar.values.shape)
        elif isinstance(press, DataArray):
            press = press.values
        else:
            pass

    kwargs['tol'] = kwargs.get('tol', 0.1)  # Dewpoint minimization accuracy
    if rel_humi is not None:
        # RH to VP to DPD
        # if 'ECMWF' in method:
        #     dpdvar.values = temp.values - dewpoint_ecmwf(temp.values, rel_humi.values)
        # else:
        vpdata = rel_humi.values * svp(temp.values, method=svp_method, p=press, **kwargs)
        dpdvar.values = temp.values - dewpoint(vpdata, method=dewp_method, **kwargs)
        origin = 't,rh' if press is None else 't,rh,p'
    elif vp is not None:
        # VP to DPD
        # if method.lower() == 'ecmwf':
        #     dpdvar.values = temp.values - vp2td(vp.values)
        # else:
        dpdvar.values = temp.values - dewpoint(vp.values, method=dewp_method, **kwargs)
        origin = 't,vp'
    else:
        # Q to DPD
        # if method.lower() == 'ecmwf':
        #     dpdvar.values = temp.values - rh2td(temp.values, q2rh(spec_humi.values, temp.values, press.values))
        # else:
        vpdata = sh2vap(spec_humi.values, press.values)
        dpdvar.values = temp.values - dewpoint(vpdata, method=dewp_method, **kwargs)
        origin = 't,q,p'

    r_att = {'origin': origin, 'svp': svp_method, 'dewp': dewp_method, 'standard_name': 'dew_point_depression',
             'long_name': 'dew point depression', 'units': 'K'}
    if press is not None:
        r_att['enhancement_factor'] = "yes"

    if (dpdvar.values < 0).any():
        message("dew point depression outside range", **leveldown(**kwargs))

    r_att['precision'] = precision
    dpdvar.values = around(dpdvar.values, decimals=precision)
    dpdvar.name = 'dpd'
    dpdvar.attrs.update(r_att)
    return dpdvar


def to_sh(vp=None, temp=None, rel_humi=None, dpd=None, press=None, method='HylandWexler', precision=6, **kwargs):
    """
    vp -(p)  sh
    rh (t,p) -> vp (p)-> sh
    dpd (t,p) -> vp (p)-> sh

    Args:
        vp: water vapor pressure
        temp: temperature
        rel_humi: rel. humidity
        dpd: dewpoint departure
        press: air pressure
        method:
        precision:

    Returns:
        spec_humi: specific humidity
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp
    from .humidity import vap2sh

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray", type(temp))

    if rel_humi is None and dpd is None and vp is None:
        raise RuntimeError("Requires either r, dpd or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError("R Requires a DataArray", type(rel_humi))

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError("DPD Requires a DataArray", type(dpd))

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError("VP Requires a DataArray", type(vp))

    if press is None:
        raise RuntimeError("Conversion ?>Q requires a pressure variable as well")

    if vp is not None:
        qvar = vp.copy()
    else:
        qvar = temp.copy()

    if isinstance(press, str):
        if press in qvar.dims:
            press = qvar[press].values
            press = _conform(press, qvar.values.shape)
    elif isinstance(press, DataArray):
        press = press.values
    else:
        pass

    kwargs['tol'] = kwargs.get('tol', 0.1)  # Dewpoint minimization accuracy
    if rel_humi is not None:
        # VP to Q
        qvar.values = vap2sh(vp.values, press)
        origin = 'vp,p'
    elif vp is not None:
        # RH to Q
        vpdata = rel_humi.values * svp(temp.values, method=method, p=press, **kwargs)
        qvar.values = vap2sh(vpdata, press)
        origin = 't,rh,p'
    else:
        # DPD to Q
        vpdata = svp(temp.values - dpd.values, method=method, p=press, **kwargs)
        qvar.values = vap2sh(vpdata, press)
        origin = 't,dpd,p'

    r_att = {'origin': origin, 'esat': method, 'standard_name': 'specific_humidity',
             'long_name': 'specific humidity', 'units': 'kg/kg'}
    if press is not None:
        r_att['enhancement_factor'] = "yes"

    r_att['precision'] = precision
    qvar.attrs.update(r_att)
    qvar.values = around(qvar.values, decimals=precision)
    qvar.name = 'sh'
    return qvar


def to_dewpoint(vp=None, temp=None, rel_humi=None, spec_humi=None, press=None, svp_method='HylandWexler',
                dewp_method='dewpoint_Boegel', precision=2, **kwargs):
    """ calculate dewpoint
    VP -> Td
    T, RH (p= -> vp -> Td
    Q,P -> vp -> Td

    Returns:
        dewp: dewpoint
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap, dewpoint
    from ..fun import message, leveldown

    if rel_humi is None and temp is None and vp is None and spec_humi is None:
        raise RuntimeError("Requires either rel. humidity, temp or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError("RH Requires a DataArray", type(rel_humi))

    if temp is not None and not isinstance(temp, DataArray):
        raise ValueError("TEMP Requires a DataArray", type(temp))

    if rel_humi is not None and temp is None:
        raise ValueError("requires TEMP and RH")

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError("VP Requires a DataArray", type(vp))

    if spec_humi is not None:
        if press is None:
            raise RuntimeError("Conversion Q>Td requires a pressure variable as well")

        dewp = spec_humi.copy()

    elif temp is not None:
        dewp = temp.copy()

    else:
        dewp = vp.copy()

    if press is not None:
        if isinstance(press, str):
            if press in dewp.dims:
                press = dewp[press].values
                press = _conform(press, dewp.values.shape)
        elif isinstance(press, DataArray):
            press = press.values
        else:
            pass

    if vp is not None:
        dewp.values = dewpoint(vp.values, method=dewp_method, **kwargs)
        origin = 'vp'
    elif temp is not None:
        vp = rel_humi.values * svp(temp.values, method=svp_method, p=press, **kwargs)
        dewp.values = dewpoint(vp, method=dewp_method, **kwargs)

        if ((temp.values - dewp.values) < 0).any():
            message("Dew point supersaturated", **leveldown(**kwargs))

        origin = 't,rh' if press is None else 't,rh,p'
    else:
        dewp.values = sh2vap(spec_humi.values, press)
        origin = 'q,p'

    r_att = {'origin': origin, 'svp': svp_method, 'dewp': dewp_method, 'standard_name': 'dew_point',
             'long_name': 'dew point', 'units': 'K'}
    if press is None:
        r_att['enhancement_factor'] = "yes"

    r_att['precision'] = precision
    dewp.values = around(dewp.values, decimals=precision)
    dewp.name = 'td'
    dewp.attrs.update(r_att)
    return dewp


def saturation_water_vapor(temp, press=None, method='HylandWexler', precision=9, **kwargs):
    """ Calculate saturation water vapor

    Args:
        temp: temperatur
        method: method
        **kwargs:

    Returns:
        svp: satuartion water vapor pressure [Pa]
    """
    from numpy import around
    from xarray import DataArray
    from .esat import svp

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray", type(temp))

    evar = temp.copy()
    if press is not None:
        if isinstance(press, str):
            if press in evar.dims:
                press = evar[press].values
                press = _conform(press, evar.values.shape)
        elif isinstance(press, DataArray):
            press = press.values
        else:
            pass

    evar.values = svp(temp.values, method=method, p=press, **kwargs)
    origin = 't' if press is None else 't,p'

    r_att = {'svp': method, 'standard_name': 'saturation_water_vapor_pressure',
             'long_name': 'saturation water vapor pressure', 'units': 'Pa',
             'origin': origin}
    if press is not None:
        r_att['enhancement_factor'] = "yes"

    r_att['precision'] = precision
    evar.attrs.update(r_att)
    evar.values = around(evar.values, decimals=precision)
    evar.name = 'td'
    return evar


def total_precipitable_water(data, dim='plev', levels=None, min_levels=8, fill_nan=True, **kwargs):
    """ Calculate total preciptable water by vertical integration of
    specific humidity profiles

     W = np.trapz(q, x=p) / 9.81  # kg/m2 == mm

    Args:
        data        (DataArray): Specific Humidity DataArray Class
        levels      (list): List of required pressure levels
        min_levels  (int):  minimum required levels for valid integration (exclusive with levels)
        fill_nan    (bool): convert missing numbers to 0

    Returns:
        DataArray : integrated TPW (vertical coordinate removed)

    Notes:
        Both Methods work fine
            W = np.trapz( q, x=p ) / 9.81
            W = np.sum( q * dp ) / 9.81   requires dp calculation dpres (NCL)
        The integral is with rho_water (assumed 1000) and neglected for conversion of m to mm
    """
    from xarray import DataArray
    from ..fun.xarray import xarray_function_wrapper
    from .tpw import tpw

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray", type(data))

    if dim not in data.dims:
        raise ValueError("dim not found", dim)

    if 'standard_name' in data.attrs:
        if 'specific' not in data.attrs['standard_name']:
            raise RuntimeError("requires specific humidity, found:", data.attrs['standard_name'])
    else:
        RuntimeWarning("requires specfifc humidty, no standard_name present")

    data = data.copy()
    if levels is not None:
        data = data.sel(**{dim: levels})  # subset with only these pressure levels

    axis = data.get_axis_num(dim)

    if levels is not None:
        min_levels = len(levels)  # make sure we have exactly these levels

    counts = data.count(dim)

    data = xarray_function_wrapper(data, wfunc=tpw, dim=dim, axis=axis, min_levels=min_levels, fill_nan=fill_nan,
                                   pin=data[dim].values)
    # Update Name and Units
    data.name = 'tpw'
    r_att = {'standard_name': 'total_precipitable_water', 'min_levels': min_levels,
             'long_name': 'total precipitable water', 'units': 'mm',
             'cell_method': 'integral: specific_humidity/9.81'}

    data.attrs.update(r_att)
    data = data.to_dataset()
    data['counts'] = counts
    return data


def vertical_interpolation(data, dim='plev', levels=None, **kwargs):
    """ Apply a vertical log-pressure interpolation, no extrapolation

    todo: add table interpolation

    Args:
        data (DataArray): input data
        dim (str): vertical (pressure, lev  ) coordinate
        levels (list, ndarray): new vertical levels
        **kwargs:

    Returns:
        DataArray : interpolated Array
    """
    from xarray import DataArray
    import numpy as np
    from ..fun.interp import profile
    from .. import config

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray", type(data))

    if dim not in data.dims:
        raise ValueError("Requires a valid dimension", dim, "of", data.dims)

    if levels is None:
        levels = config.std_plevels

    data = data.copy()
    axis = data.get_axis_num(dim)
    pin = data[dim].values
    values = np.apply_along_axis(profile, axis, data.values, pin, levels)
    data = data.reindex({dim: levels})  # can fail with duplicated values
    data.values = values
    cmethod = "%s: intp(%d > %d)" % (dim, len(pin), len(levels))
    if 'cell_method' in data.attrs:
        data.attrs['cell_method'] = cmethod + data.attrs['cell_method']
    else:
        data.attrs['cell_method'] = cmethod
    return data


def adjust_dpd30(data, num_years=10, dim='time', subset=slice(None, '1994'), value=30, bins=None, thres=1,
                 return_mask=False,
                 **kwargs):
    """ Specifc Function to remove a certain value from the Histogram (DPD)

    Parameters
    ----------
    Data        DataArray       input database
    var         str             variable
    nyears      int             evaluation period: anomaly within 10 years
    ylimit      int             only before that year
    ival        float
    bins        list

    Returns
    -------
    copy of DataArray
    """
    import numpy as np
    from xarray import DataArray
    from ..fun import message
    from .manual import remove_spurious_values
    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray", type(data))

    if dim not in data.dims:
        raise ValueError("DataArray class has no datetime dimension")

    if bins is None:
        bins = np.arange(0, 60)
        message("Using default bins [0, 60]", **kwargs)

    axis = data.dims.index(dim)
    dates = data[dim].sel({dim: subset}).values.copy()
    message("Using Subset %s" % str(subset), **kwargs)
    values = data.sel({dim: subset}).values.copy()

    if np.sum(np.isfinite(values)) == 0:
        return data

    if False:
        # How much data (30 +- thres) is there anyway?
        i30 = data.copy()
        i30.values = (i30 > (value - thres)) & (i30 < (value + thres))
        # i30 = i30.resample(freq='A', agg=np.sum)
        years, groups = date_rolling(dates, axis, data.values.ndims, num_years, freq='A')
        i30 = groupby_apply(groups, i30, np.sum, groups={'years': years}, axis=axis)

    values, count, mask = remove_spurious_values(dates, values, axis=axis, num_years=num_years, value=value, bins=bins,
                                                 thres=thres, **kwargs)
    data.loc[{dim: subset}] = values  # Assign new values
    data.attrs['DPD%d' % int(value)] = count

    if return_mask:
        dmask = data.copy()
        dmask.values = False  # Everything false
        dmask.values = mask
        dmask.name += '_mask'
        dmask.attrs['units'] = '1'
        dmask.attrs['standard_name'] += '_mask'
        return data, dmask
    return data


def _conform(data, shape):
    """ Make numpy array conform to a certain shape

    Args:
        data:
        shape:

    Returns:

    """
    import numpy as np
    if not isinstance(data, np.ndarray):
        raise ValueError('Requires a numpy array')

    if not isinstance(shape, (tuple, list)):
        raise ValueError('Requires a tuple or list')

    data = data.copy()
    n = data.shape

    assert np.any([i in shape for i in n]), "Shapes do not allign?!"

    for i, j in enumerate(shape):
        if j not in n:
            data = np.expand_dims(data, axis=i)
    return data
