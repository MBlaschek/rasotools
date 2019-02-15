# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr

__all__ = ['Network']


class Network(object):
    def __init__(self, name=None, idents=None, center_lon=None, center_lat=None, distance=None):
        self.name = name if name is not None else "Standard"
        self.sondes = list()
        self.lon = None
        self.lat = None
        self.distance = None
        self.weights = None
        self.data = None
        self._rslist = None

        if idents is not None:
            if not isinstance(idents, list):
                idents = [idents]
            self.sondes = idents

        if center_lon is not None and center_lat is not None and distance is not None:
            self.distance = distance
            self.lon = center_lon
            self.lat = center_lat
            self.sondes = self.get_idents(center_lon, center_lat, distance)
        self.nsondes = len(self.sondes)

    def __repr__(self):
        summary = u"Network {name} ({nsondes}) Lon: {lon} [deg] Lat: {lat} [deg] Dist: {distance} [km]".format(
            **self.__dict__)
        summary += u"\nData: {data} Weights: {weights}".format(
            **{'data': self.data is not None, 'weights': self.weights is not None})

        if self.data is not None:
            summary += "\n Data:\n"
            summary += repr(self.data).replace('\n', '\n - ')

        return summary
    #
    # def __setitem__(self, key, value):
    #     if not isinstance(value, (list, tuple)):
    #         raise ValueError("Set Radiosonde:  [ID] = (lon, lat)")
    #     if key not in self.sondes:
    #         self.sondes.append(key)
    #         self.lon.append(value[0])
    #         self.lat.append(value[1])
    #     else:
    #         idx = self.sondes.index(key)
    #         print(self.lon[idx], self.lat[idx], ">", value)
    #         self.lon[idx] = value[0]
    #         self.lat[idx] = value[1]
    #
    # def __delitem__(self, key):
    #     if key in self.sondes:
    #         idx = self.sondes.index(key)
    #         self.sondes.pop(idx)
    #         self.lon.pop(idx)
    #         self.lat.pop(idx)
    #
    # def __getitem__(self, key):
    #     if key in self.sondes:
    #         idx = self.sondes.index(key)
    #         return self.sondes[idx], self.lon[idx], self.lat[idx]

    def get_idents(self, lon, lat, distance, filename=None):
        from .fun import distance as f_dist
        from .io import read_radiosondelist

        if self._rslist is None:
            sondes = read_radiosondelist(filename=filename)
        else:
            sondes = self._rslist

        sondes['dist'] = [f_dist(lon, lat, i[1].lon, i[1].lat) for i in sondes.iterrows()]
        self._rslist = sondes
        return list(sondes[sondes.dist <= distance].index)

    def calculate_weights(self, lon=None, lat=None, distance=None):
        if hasattr(self, lon) and hasattr(self, lat):
            lon = self.lon
            lat = self.lat
            if hasattr(self, distance):
                distance = self.distance

        if lon is None or lat is None or distance is None:
            raise RuntimeError()

        self.weights = 0

    def open_data(self, filename_pattern=None, directory=None, drop_variables=None):
        from .fun import find_files
        files = find_files(directory, filename_pattern, recursive=True)
        # filter with sonde ids ?
        index = pd.Index(self.sondes, name='sondes')
        self.data = xr.open_mfdataset(files, concat_dim=index)
        print(self.data)

    def table(self, idents=None, filename=None):
        from .io import read_radiosondelist
        if idents is None:
            idents = self.sondes

        if self._rslist is None:
            self._rslist = read_radiosondelist(filename=filename)

        return self._rslist.loc[idents]
