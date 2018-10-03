# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr

__all__ = ['Network']


class Network(object):
    def __init__(self, name=None, idents=None, lon=None, lat=None, distance=None):
        self.name = name if name is not None else "Standard"
        self.sondes = []
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

        if lon is not None and lat is not None and distance is not None:
            self.distance = distance
            self.lon = lon
            self.lat = lat
            self.sondes = self.get_idents(lon, lat, distance)
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

    def load_data(self, data_vars='all'):
        # launch open_radiosonde for each ID
        # concat to Dataset
        # change data_vars from all to list or minimal
        # open_mfdataset (multiple files)
        pass

    def table(self, idents=None, filename=None):
        from .io import read_radiosondelist
        if idents is None:
            idents = self.sondes

        if self._rslist is None:
            self._rslist = read_radiosondelist(filename=filename)

        return self._rslist.loc[idents]
