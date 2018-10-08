# -*- coding: utf-8 -*-
import pandas as pd


__all__ = ['read_radiosondelist']


def read_radiosondelist(filename=None, minimal=True, with_igra=False):
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


def wmo2igra(ident):
    from .. import get_data
    ident = str(ident)  # make sure
    igra2wmo = pd.read_json(get_data('igra2wmo.json'), dtype=False)  # sa strings
    if igra2wmo.wmo.str.contains(ident).any():
        return igra2wmo[igra2wmo.wmo.str.contains(ident)].index.tolist()[0]
    return None


def igra2wmo(ident):
    from .. import get_data
    ident = str(ident)  # make sure
    igra2wmo = pd.read_json(get_data('igra2wmo.json'), dtype=False)  # sa strings
    if igra2wmo.index.str.contains(ident).any():
        return igra2wmo[igra2wmo.index.str.contains(ident)].wmo.tolist()[0]
    return None
