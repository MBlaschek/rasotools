# -*- coding: utf-8 -*-

__all__ = ['to_rh']


def to_rh(temp, dpd=None, spec_humi=None, press=None, method='HylandWexler', **kwargs):
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

    Returns:
        DataArray : relative humidity [1]
    """
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap

    if not isinstance(temp, DataArray):
        raise ValueError("Requires a DataArray class Object")

    if dpd is None and spec_humi is None:
        raise RuntimeError("Requires either dpd or q for conversion")

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError("Requires a DataArray class Object")

    if spec_humi is not None:
        if not isinstance(spec_humi, DataArray):
            raise ValueError("Requires a DataArray class Object")

        if press is None:
            raise RuntimeError("Conversion requires a pressure variable as well")

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
        vpdata = svp(temp.values - dpd.values, method=method, p=press)
        rvar.values = vpdata / svp(temp.values, method=method, p=press)
        origin = 'DPD'
    else:
        vpdata = sh2vap(spec_humi.values, press)
        rvar.values = vpdata / svp(temp.values, method=method, p=press)
        origin = 'Q'

    r_att = {'units': '1', 'standard_name': 'relativ_humidity', 'esat': method, 'origin': origin}
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'

    rvar.attrs.update(r_att)
    return rvar


def to_vp(temp, dpd=None, rel_humi=None, spec_humi=None, press=None, method='HylandWexler', **kwargs):
    """ Convert dewpoint departure, rel. humidity or specific humidity to water vapor pressure

    1. DPD to VP (T)
    2. RH to VP (T)
    3. Q to VP (T, P)

    Args:
        temp: Name of temperature
        rel_humi:   Name of rel. humidity (Input)
        dpd:   Name of DPD (Input)
        spec_humi:  Name of Q (Input)
        press:   Name of pressure (Input)
        method:  Saturation water vapor pressure formulation

    Returns
    -------
        DataArray : water vapor pressure [Pa]
    """
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap
    funcid = "[VP] "
    if not isinstance(temp, DataArray):
        raise ValueError(funcid + "Requires a DataArray class")

    if dpd is None and spec_humi is None and rel_humi is None:
        raise RuntimeError(funcid + "Requires either dpd, q or r for conversion")

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError(funcid + "DPD Requires a DataArray class")

    if spec_humi is not None and not isinstance(spec_humi, DataArray):
        raise ValueError(funcid + "Q Requires a DataArray class")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError(funcid + "R Requires a DataArray class")

    if spec_humi is not None:
        if press is None and spec_humi.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>VP requires a pressure variable as well")

    vpvar = temp.copy()
    attrs = {'units': 'Pa', 'standard_name': 'water_vapor_pressure'}
    for iatt in list(vpvar.attrs.keys()):
        del vpvar.attrs[iatt]

    if kwargs.get('p', None) is not None:
        if kwargs['p'] == 'press':
            if press is None:
                press = vpvar.dims[vpvar.get_dimension_by_axis('Z')].values  # get pressure values
                press = _conform(press, vpvar.values.shape)
            elif isinstance(press, DataArray):
                kwargs['p'] = press.values
            else:
                kwargs['p'] = press

    if dpd is not None:
        # DPD to VP
        vpvar.values = svp(temp - dpd, method=method, **kwargs)
        origin = 'DPD'

    elif rel_humi is not None:
        # RH to VP
        vpvar.values = svp(temp, method=method, **kwargs) * rel_humi
        origin = 'R'

    else:
        if press is None:
            press = spec_humi.dims[spec_humi.get_dimension_by_axis('Z')].values  # get pressure values
            press = _conform(press, spec_humi.values.shape)

        elif isinstance(press, DataArray):
            press = press.values  # Assume Pa ?

        else:
            pass
        # Q to VP
        vpvar.values = sh2vap(spec_humi, press)
        origin = 'Q'

    attrs.update({'origin': origin, 'esat': method})
    attrs.update(kwargs)
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'
    for ikey, ival in attrs.items():
        vpvar.attrs[ikey] = ival

    return vpvar


def to_dpd(temp, rel_humi=None, vp=None, spec_humi=None, press=None, method='HylandWexler', **kwargs):
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
        method: Saturation water vapor pressure formulation

    Returns:
        DataArray: dewpoint depression [K]
    """
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap, dewpoint
    funcid = "[DPD] "

    if not isinstance(temp, DataArray):
        raise ValueError(funcid + "Requires a DataArray class")

    if rel_humi is None and spec_humi is None and vp is None:
        raise RuntimeError(funcid + "Requires either r, q or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError(funcid + "R Requires a DataArray class")

    if spec_humi is not None and not isinstance(spec_humi, DataArray):
        raise ValueError(funcid + "Q Requires a DataArray class")

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError(funcid + "VP Requires a DataArray class")

    if spec_humi is not None:
        if press is None and spec_humi.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>DPD requires a pressure variable as well")

    dpdvar = temp.copy()
    attrs = {'units': 'K', 'standard_name': 'dewpoint_depression'}
    for iatt in list(dpdvar.attrs.keys()):
        del dpdvar.attrs[iatt]

    if kwargs.get('p', None) is not None:
        if kwargs['p'] == 'press':
            if press is None:
                press = dpdvar.dims[dpdvar.get_dimension_by_axis('Z')].values  # get pressure values
                press = _conform(press, dpdvar.values.shape)
            elif isinstance(press, DataArray):
                kwargs['p'] = press.values
            else:
                kwargs['p'] = press

    kwargs['tol'] = kwargs.get('tol', 0.1)  # Dewpoint minimization accuracy
    if rel_humi is not None:
        # RH to VP to DPD
        # if 'ECMWF' in method:
        #     dpdvar.values = temp.values - dewpoint_ecmwf(temp.values, rel_humi.values)
        # else:
        vpdata = rel_humi.values * svp(temp.values, method=method, **kwargs)
        dpdvar.values = temp.values - dewpoint(vpdata, method=method, **kwargs)
        origin = 'R'
    elif vp is not None:
        # VP to DPD
        # if method.lower() == 'ecmwf':
        #     dpdvar.values = temp.values - vp2td(vp.values)
        # else:
        dpdvar.values = temp.values - dewpoint(vp.values, method=method, **kwargs)
        origin = 'VP'
    else:
        # Q to DPD
        # if method.lower() == 'ecmwf':
        #     dpdvar.values = temp.values - rh2td(temp.values, q2rh(spec_humi.values, temp.values, press.values))
        # else:
        vpdata = sh2vap(spec_humi.values, press.values)
        dpdvar.values = temp.values - dewpoint(vpdata, method=method, **kwargs)
        origin = 'Q'

    attrs.update({'origin': origin, 'esat': method})
    attrs.update(kwargs)
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'
    for ikey, ival in attrs.items():
        dpdvar.attrs[ikey] = ival

    return dpdvar


def to_sh(vp=None, temp=None, rel_humi=None, dpd=None, press=None, method='HylandWexler', **kwargs):
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

    Returns:
        spec_humi: specific humidity
    """
    from xarray import DataArray
    from .esat import svp
    from .humidity import vap2sh
    funcid = "[SH] "

    if not isinstance(temp, DataArray):
        raise ValueError(funcid + "Requires a DataArray class")

    if rel_humi is None and dpd is None and vp is None:
        raise RuntimeError(funcid + "Requires either r, dpd or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError(funcid + "R Requires a DataArray class")

    if dpd is not None and not isinstance(dpd, DataArray):
        raise ValueError(funcid + "DPD Requires a DataArray class")

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError(funcid + "VP Requires a DataArray class")

    if vp is not None:
        if press is None and vp.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>DPD requires a pressure variable as well")

        qvar = vp.copy()
    else:
        if press is None and temp.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>DPD requires a pressure variable as well")

        qvar = temp.copy()

    attrs = {'units': 'kg/kg', 'standard_name': 'specific_humidity'}
    for iatt in list(qvar.attrs.keys()):
        del qvar.attrs[iatt]

    if press is None:
        press = qvar.dims[qvar.get_dimension_by_axis('Z')].values  # get pressure values
        press = _conform(press, qvar.values.shape)
    elif isinstance(press, DataArray):
        press = press.values

    if kwargs.get('p', None) is not None:
        if kwargs['p'] == 'press':
            kwargs['p'] = press

    kwargs['tol'] = kwargs.get('tol', 0.1)  # Dewpoint minimization accuracy
    if rel_humi is not None:
        # VP to Q
        qvar.values = vap2sh(vp.values, press)
        origin = 'VP'
    elif vp is not None:
        # RH to Q
        vpdata = rel_humi.values * svp(temp.values, method=method, **kwargs)
        qvar.values = vap2sh(vpdata, press)
        origin = 'R'
    else:
        # DPD to Q
        vpdata = svp(temp.values - dpd.values, method=method, **kwargs)
        qvar.values = vap2sh(vpdata, press)
        origin = 'DPD'

    attrs.update({'origin': origin, 'esat': method})
    attrs.update(kwargs)
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'
    for ikey, ival in attrs.items():
        qvar.attrs[ikey] = ival

    return qvar


def to_dewpoint(vp=None, temp=None, rel_humi=None, spec_humi=None, press=None, method='HylandWexler', **kwargs):
    """ calculate dewpoint
    VP -> Td
    T, RH (p= -> vp -> Td
    Q,P -> vp -> Td

    Returns:
        dewp: dewpoint
    """
    from xarray import DataArray
    from .esat import svp
    from .humidity import sh2vap, dewpoint
    funcid = "[Td]"
    if not isinstance(vp, DataArray):
        raise ValueError(funcid + "Requires a DataArray class")

    if rel_humi is None and temp is None and vp is None and spec_humi is None:
        raise RuntimeError(funcid + "Requires either rel. humidity, temp or vp for conversion")

    if rel_humi is not None and not isinstance(rel_humi, DataArray):
        raise ValueError(funcid + "RH Requires a DataArray class")

    if temp is not None and not isinstance(temp, DataArray):
        raise ValueError(funcid + "TEMP Requires a DataArray class")

    if rel_humi is not None and temp is None:
        raise ValueError(funcid + "requires TEMP and RH")

    if vp is not None and not isinstance(vp, DataArray):
        raise ValueError(funcid + "VP Requires a DataArray class")

    if spec_humi is not None:
        if press is None and spec_humi.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>Td requires a pressure variable as well")

        dewp = spec_humi.copy()

    elif temp is not None:
        if press is None and temp.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>Td requires a pressure variable as well")

        dewp = temp.copy()

    else:
        if press is None and vp.get_dimension_by_axis('Z') is None:
            raise RuntimeError(funcid + "Conversion Q>DPD requires a pressure variable as well")

        dewp = vp.copy()

    attrs = {'units': 'K', 'standard_name': 'dewpoint'}
    for iatt in list(dewp.attrs.keys()):
        del dewp.attrs[iatt]

    if press is None:
        press = dewp.dims[dewp.get_dimension_by_axis('Z')].values  # get pressure values
        press = _conform(press, dewp.values.shape)
    elif isinstance(press, DataArray):
        press = press.values

    if kwargs.get('p', None) is not None:
        if kwargs['p'] == 'press':
            kwargs['p'] = press

    if vp is not None:
        dewp.values = dewpoint(vp.values, method=method, **kwargs)
        origin = 'VP'
    elif temp is not None:
        vp = rel_humi.values * svp(temp.values, method=method, **kwargs)
        dewp.values = dewpoint(vp, method=method, **kwargs)
        origin = 'R'
    else:
        dewp.values = sh2vap(spec_humi.values, press)
        origin = 'S'

    attrs.update({'origin': origin, 'esat': method})
    attrs.update(kwargs)
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'

    for ikey, ival in attrs.items():
        dewp.attrs[ikey] = ival

    return dewp


def saturation_water_vapor(temp, method='HylandWexler', **kwargs):
    """ Calculate saturation water vapor

    Args:
        temp: temperatur
        method: method
        **kwargs:

    Returns:
        svp: satuartion water vapor pressure [Pa]
    """
    from xarray import DataArray
    from .esat import svp
    funcid = "[ES] "
    if not isinstance(temp, DataArray):
        raise ValueError(funcid + "Requires a DataArray class Object")

    evar = temp.copy()
    for iatt in list(evar.attrs.keys()):
        del evar.attrs[iatt]

    if kwargs.get('p', None) is not None:
        if kwargs['p'] == 'press':
            press = evar.dims[evar.get_dimension_by_axis('Z')].values  # get pressure values
            kwargs['p'] = _conform(press, evar.values.shape)

    evar.values = svp(temp.values, method=method, **kwargs)
    attrs = {'units': 'Pa', 'standard_name': 'saturation_water_vapor_pressure', 'esat': method}
    attrs.update(kwargs)
    if 'p' in kwargs.keys():
        kwargs['p'] = 'enhancement_factor'
    for ikey, ival in attrs.items():
        evar.attrs[ikey] = ival

    # esat.mask(esat < 0)
    return evar


def total_precipitable_water(data, levels=None, min_levels=8, fill_nan=True, method='HylandWexler', temp=None,
                             return_counts=False, **kwargs):
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
    import numpy as np
    from xarray import DataArray
    from .esat import svp
    from .humidity import vap2sh
    from .tpw import tpw

    if not isinstance(data, DataArray):
        raise ValueError("Requires a DataArray class object")

    if 'standard_name' in data.attrs.keys():
        if temp is None:
            if 'specific' not in data.attrs['standard_name']:
                raise RuntimeWarning("Standard Name does not say: specific; This Function requires specific humidity")
        else:
            if 'dew_point' not in data.attrs['standard_name']:
                raise RuntimeError("dewpoint not found")
    data = data.copy()
    ilev = data.get_dimension_by_axis('Z')
    if levels is not None:
        data = data.subset(dims={ilev: levels})  # subset with only these pressure levels

    ilev = data.get_dimension_by_axis('Z')
    axis = data.order.index(ilev)

    # compatible pressure array
    pin = np.zeros(data.values.shape)
    nshape = [1] * len(data.order)
    nshape[axis] = data.dims[ilev].values.size
    plevs = data.dims[ilev].values.reshape(tuple(nshape))
    pin[::] = plevs  # fill in

    if levels is not None:
        min_levels = len(levels)  # make sure we have exactly these levels

    values = data.values.copy()

    if temp is not None:
        # values = dpd2sh(values, temp.values, pin, method=method)
        td = temp.values - values  # dew point
        e = svp(td, method=method, **kwargs)  # water vapor
        values = vap2sh(e, pin)  # specific humidity

    values = tpw(values, pin, axis=axis, min_levels=min_levels, fill_nan=fill_nan)

    # put into new array
    order = list(data.order)  # copy
    order.remove(ilev)
    dims = data.get_dimension_values()
    dims.pop(ilev)
    data.update_values_dims_remove(values, order, dims)

    # Update Name and Units
    data.name = 'tpw'
    data.attrs['units'] = 'mm'
    data.attrs['standard_name'] = 'total_precipitable_water'
    data.attrs['operator'] = 'q to TPW'
    if levels is None:
        data.attrs['tpw_minlevs'] = min_levels

    if return_counts:
        icounts = np.sum(np.isfinite(values), axis=axis) / float(plevs.size)  # relative
        counts = data.copy()
        counts.name = 'tpw_counts'
        counts.values = icounts
        counts.attrs.units = '1'
        counts.attrs.standard_name = 'level_count'
        return data, counts

    return data


def vertical_interpolation(data, dim, levels=None, **kwargs):
    """ Apply a vertical log-pressure interpolation, no extrapolation

    Args:
        data (DataArray): input data
        dim (str): vertical (pressure) coordinate
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
        raise ValueError("Requires a DataArray class object")

    if dim not in data.dims:
        raise ValueError("Requires a valid dimension", dim, "of", data.dims)

    if levels is None:
        levels = config.std_plevels

    data = data.copy()
    axis = data.dims.index(dim)
    pin = data[dim].values
    values = np.apply_along_axis(profile, axis, data.values, pin, levels)
    data = data.reindex({dim: levels})
    data.values = values
    cmethod = "%s: intp(%d > %d)" % (dim, len(pin), len(levels))
    if 'cell_method' in data.attrs:
        data.attrs['cell_method'] = cmethod + data.attrs['cell_method']
    else:
        data.attrs['cell_method'] = cmethod
    return data


def adjust_dpd30(data, num_years=10, datedim='date', subset=slice(None, '1994'), value=30, bins=None, thres=1,
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
        raise ValueError("Requires a DataArray class object")

    if datedim not in data.dims:
        raise ValueError("DataArray class has no datetime dimension")

    if bins is None:
        bins = np.arange(0, 60)
        message("Using default bins [0, 60]", **kwargs)

    axis = data.dims.index(datedim)
    dates = data[datedim].values.copy()
    itx = [slice(None)] * len(data.dims)
    itx[axis] = subset
    message("Using Subset %s" % str(subset), **kwargs)
    values = data.loc[itx].values

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
    data.loc[itx] = values  # Assign new values
    data.attrs['DPD%d' % int(value)] = count

    if return_mask:
        dmask = data.copy()
        dmask.loc[::] = False  # Everything false
        dmask.loc = mask
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
