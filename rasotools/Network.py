# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

__all__ = ['Network', 'network_from_stationlist', 'network_from_datafiles']


def network_from_stationlist(name, stationlist, ident='wmo', lon='lon', lat='lat'):
    new = Network(name)
    new.idents = list(stationlist[ident])
    new.lon = stationlist[lon].values
    new.lat = stationlist[lat].values
    new.stations = stationlist
    return new


def network_from_datafiles(name, pattern, directory=None, **kwargs):
    from .fun.station import build_stationlist_from_netcdf_archive
    #
    # find files
    # read meta information from files
    #
    stationlist = build_stationlist_from_netcdf_archive(pattern, directory=directory, **kwargs)
    new = network_from_stationlist(name, stationlist)

    return new


class Network(object):
    """ Network of Radiosonde stations
    name


    """

    def __init__(self, name, idents=None):
        if not isinstance(name, str):
            raise ValueError("Name needs to be s string")

        self.name = name
        if idents is not None:
            if not isinstance(idents, list):
                idents = [idents]
            self.idents = idents
        else:
            self.idents = list()

        self.lon = None
        self.lat = None
        self.distance = None
        self.weights = None
        self.data = None
        self.stations = None

    def __repr__(self):
        return "%s  (%d)" % (self.name, len(self.idents))

    def __iter__(self):
        # iter(self.__dict__)   # does the same
        return (x for x in self.idents)

    def __getitem__(self, item):
        if item in self.idents:
            i = self.idents.index(item)
            return self.stations.iloc[i]
        return None

    # def __setitem__(self, key, value):
    #     if isinstance(value, (tuple, list)):
    #         if
    #         self.idents.append()
    #     setattr(self, key, value)

    # def __delitem__(self, key):
    #     if key in self.__dict__.keys():
    #         delattr(self, key)

    def _calculate_dist_matrix(self):
        from .fun.cal import distance
        if self.lon is not None and self.lat is not None:
            print("Calculating Distance Matrix ...")
            result = [distance(self.lon, self.lat, ilon, ilat) for ilon, ilat in zip(self.lon, self.lat)]
            self.distance = pd.DataFrame(result, index=self.idents, columns=self.idents)
        else:
            raise RuntimeError("Network is missing lon, lat information")

    def neighboring_stations(self, ident, distance):
        if self.distance is None:
            self._calculate_dist_matrix()

        if ident not in self.idents:
            raise ValueError("Ident not found", ident)

        result = self.distance.loc[ident].sort_values()
        return pd.Series(result[result <= distance], name='dist_%s_km' % ident)

    def add(self, ident, **kwargs):
        pass

    def set_station(self, ident, lon, lat):
        pass

    def del_station(self, ident):
        pass

    def load_data(self, files=None, pattern=None, directory=None, variables=None, invert_selection=False, **kwargs):
        from .fun import find_files
        from . import config

        def iadd_id(x):
            return _add_id(x, variables, invert=invert_selection)

        if directory is None:
            directory = config.rasodir

        if files is None:
            files = find_files(directory, pattern)
            print(directory, pattern)

        self.data = xr.open_mfdataset(files, concat_dim='sonde', preprocess=iadd_id, **kwargs)

    def apply_function(self):
        pass

    def plot_map(self, filename=None, **kwargs):
        from .plot import map
        ax = map.points(self.lon, self.lat, title=self.name, **kwargs)
        if filename is not None:
            plt.savefig(filename, **kwargs)
        return ax

    def plot_neighboring_stations(self, ident, distance, filename=None, **kwargs):
        import matplotlib.patches as mpatches
        from .plot import map
        neighbors = self.neighboring_stations(ident, distance * 2)
        neighbors.name = 'distance'
        idx = np.in1d(self.idents, neighbors.index.values)
        stations = pd.DataFrame({'lon': self.lon[idx], 'lat': self.lat[idx]}, index=np.array(self.idents)[idx])
        stations['distance'] = neighbors
        stations = stations.sort_values('distance')
        print(stations)
        stations['color'] = 'b'
        stations.loc[ident, 'color'] = 'r'
        stations.loc[stations.distance > distance, 'color'] = 'w'
        # todo add title with distance
        ax = map.points(stations.lon, stations.lat, labels=stations.index.values, color=stations.color, **kwargs)
        lon_radius = distance / (111.32 * np.cos(stations.lat[0] * np.pi / 180.))
        lat_radius = distance / 110.574
        print(lon_radius, lat_radius)
        ax.add_patch(
            mpatches.Ellipse(xy=[stations.lon[0], stations.lat[0]], width=lon_radius*2., height=lat_radius*2.,
                            color='red', alpha=0.2,
                            zorder=2, transform=ax.projection))

        if filename is not None:
            plt.savefig(filename, **kwargs)

        return ax


def _add_id(ds, ivars, invert=False):
    if 'ident' in ds.attrs:
        ds.coords['sonde'] = ds.attrs['ident']
    elif 'station_id' in ds.attrs:
        ds.coords['sonde'] = ds.attrs['station_id']
    else:
        ds.coords['sonde'] = ds.encoding['source'].split('/')[-2]

    if ivars is not None:
        if invert:
            ds = ds[[i for i in list(ds.data_vars) if i not in ivars]]
        else:
            ds = ds[[i for i in list(ds.data_vars) if i in ivars]]
    return ds
