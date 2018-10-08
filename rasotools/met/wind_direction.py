# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

__all__ = ['polar_to_uv', 'uv_to_polar']


def polar_to_uv(data, drop=False):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Requires a DataFrame")

    if data.columns.isin(['ws', 'wd']).sum() != 2:
        raise ValueError("Missing ws or wd")

    data['u'] = data['ws'] * np.cos(data['wd'])
    data['v'] = data['ws'] * np.sin(data['wd'])

    if drop:
        data.drop(['ws', 'wd'], 1, inplace=True)

    return data


def uv_to_polar(data, drop=False):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Requires a DataFrame")

    if data.columns.isin(['u', 'v']).sum() != 2:
        raise ValueError("Missing u or v")

    data['ws'] = np.sqrt(data['u'] ** 2 + data['v'] ** 2)
    data['wd'] = np.arctan2(data['v'], data['u'])

    if drop:
        data.drop(['u', 'v'], 1, inplace=True)

    return data
