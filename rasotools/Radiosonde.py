# -*- coding: utf-8 -*-
import os

import numpy as np
import xarray as xr

from .fun import message, update_kw, dict_add

__all__ = ["Radiosonde", "open_radiosonde"]


def dim_summary(obj):
    """ Auxilliary class function for printing
    """
    if hasattr(obj, 'data_vars'):
        return "%d vars [%s]" % (len(obj.data_vars), ", ".join(["%s(%s)" % (i, j) for i, j in obj.dims.items()]))

    if hasattr(obj, 'shape'):
        i = np.shape(obj)
        if i != ():
            return i

    if hasattr(obj, 'size'):
        i = np.size(obj)
        if i == 1:
            return obj
        else:
            return i

    if isinstance(obj, (list, tuple, dict)):
        i = len(obj)
        if i == 1:
            return obj
        else:
            return i

    return obj


def formatting(obj):
    return u'<%s (%s)>' % (type(obj).__name__, dim_summary(obj))


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return u"\n".join("%-10s : %s" % (i, formatting(getattr(self, i))) for i in self.__dict__.keys())

    def __iter__(self):
        # iter(self.__dict__)   # does the same
        return (x for x in self.__dict__.keys())

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        if key in self.__dict__.keys():
            delattr(self, key)


class Radiosonde(object):
    def __init__(self, ident, data=None, **kwargs):
        self.ident = ident
        if data is not None:
            self.data = Bunch(**data)
        else:
            self.data = Bunch()  # empty
        self.attrs = Bunch(**kwargs)
        self.directory = None

    def __repr__(self):
        summary = u"Radiosonde (%s)\n" % self.ident
        if len(list(self.data)) > 0:
            summary += "Data: \n"
            summary += repr(self.data)
        if len(list(self.attrs)) > 0:
            summary += "\nGlobal Attributes: \n"
            summary += repr(self.attrs)
        if self.directory is not None:
            summary += "\nDirectory: " + self.directory
        return summary

    def __getitem__(self, item):
        if item in self.data:
            return getattr(self.data, item)
        return None

    def __setitem__(self, key, value):
        if key in self.data:
            print("Warning overwriting ...", key)
        setattr(self.data, key, value)

    def __delitem__(self, key):
        if key in self.data:
            delattr(self.data, key)

    def add(self, name, filename=None, directory=None, xwargs={}, **kwargs):
        """ Add data to radiosonde class object [container]

        Args:
            name (str): used as filename and/or as data name
            filename (str): filename to read from (netcdf)
            directory (str): directory of radiosonde store, default config rasodir
            xwargs (dict): xarray open_dataset keywords
        """
        from . import config

        if self.directory is not None:
            directory = self.directory

        if directory is None:
            if 'rasodir' in config and os.path.isdir(config.rasodir):
                directory = config.rasodir + '/' + str(self.ident) + '/'
            else:
                directory = './' + str(self.ident) + '/'
            message(directory, **kwargs)

        if filename is not None:
            if '.nc' not in filename:
                print("Warning can read only NetCDF / Xarray open_dataset formats")

            if '*' not in filename:
                if not os.path.isfile(filename):
                    if directory is not None:
                        if not os.path.isfile(directory + '/' + filename):
                            filename = directory + '/' + str(self.ident) + '/' + filename
                        else:
                            filename = directory + '/' + filename

        if filename is None:
            if os.path.isfile(directory + '/' + name + '.nc'):
                filename = directory + '/' + name + '.nc'
            else:
                message("Not found:", name, directory, **kwargs)
                return

        if '*' in filename:
            try:
                message("Reading ...", filename, **kwargs)
                self.data[name] = xr.open_mfdataset(filename, **xwargs)
            except OSError:
                message("Reading ...", directory + '/' + filename, **kwargs)
                self.data[name] = xr.open_mfdataset(directory + '/' + filename, **xwargs)

        else:
            message("Reading ...", filename, **kwargs)
            self.data[name] = xr.open_dataset(filename, **xwargs)

        # merge dictionaries and append common
        self.attrs.__dict__ = dict_add(vars(self.attrs), dict(self.data[name].attrs))
        if 'ident' in self.attrs:
            if self.ident != self.attrs['ident']:
                print("Warning different idents: ", self.ident, self.attrs['ident'])
            if self.ident == "":
                self.ident = self.attrs['ident']

    def clear(self, exclude=None, **kwargs):
        if exclude is None:
            exclude = []
        else:
            if not isinstance(exclude, list):
                exclude = [exclude]

        for idata in list(self.data):
            if idata not in exclude:
                if hasattr(self.data[idata], 'close'):
                    self.data[idata].close()  # for xarray netcdf open files
                    message(idata, 'closed', **kwargs)
                del self.data[idata]
                message(idata, **kwargs)

        for iatt in list(self.attrs):
            if iatt not in exclude:
                del self.attrs[iatt]

    def rename(self, old_name, new_name, **kwargs):
        if old_name in self.data:
            self.data.__dict__[new_name] = self.data.__dict__.pop(old_name)
            message(old_name, " > ", new_name, **kwargs)

    def get_info(self, add_attrs=False, **kwargs):
        from .io.lists import read_radiosondelist
        sondes = read_radiosondelist(**update_kw('minimal', False, **kwargs))
        if self.ident in sondes.index:
            infos = sondes.xs(self.ident).to_dict()
            message(Bunch(**infos), **kwargs)
            if add_attrs:
                self.attrs.__dict__.update(infos)
                message("Attributes added", **update_kw('verbose', 1, **kwargs))
        else:
            message("No information found: %s" % self.ident, **kwargs)

    def list_store(self, directory=None, varinfo=False, ncinfo=False):
        import time
        from . import config
        from .io.info import view
        from .fun import print_fixed

        if self.directory is not None:
            directory = self.directory

        if directory is None:
            directory = config.rasodir
            directory += "/%s/" % self.ident

        if os.path.isdir(directory):
            summe = 0
            print("Available Files (Datasets): ")
            print(directory)
            print('_' * 80)
            for ifile in os.listdir(directory):
                if os.path.isdir(directory + ifile):
                    continue

                current = os.path.getsize(directory + ifile) / 1024 / 1024
                itime = time.ctime(os.path.getctime(directory + ifile))
                print("%-20s : %4d MB  : %s" % (ifile, current, itime))
                if '.nc' in ifile:
                    if varinfo:
                        with xr.open_dataset(directory + ifile) as f:
                            print("Variables:")
                            print(print_fixed(list(f.data_vars), ',', 80, offset=10))
                    elif ncinfo:
                        view(directory + ifile)
                    else:
                        pass
                print('_' * 80)
                summe += current

            print("\n%20s : %4d MB" % ('Total', summe))
        else:
            print("Store not found!")

    def to_netcdf(self, name, filename=None, directory=None, force=False, add_global=True, **kwargs):
        """Write each data variable to NetCDF 4

        Args:
            name (str): data name
            filename (str): filename
            directory (str): directory
            force (bool): force new file
        Result:
            bool : Status
        """
        from . import config

        if not isinstance(name, list):
            name = [name]

        if directory is None:
            if self.directory is not None:
                directory = self.directory

            elif 'rasodir' in config:
                if os.path.isdir(config.rasodir):
                    directory = config.rasodir + '/' + str(self.ident) + '/'
            else:
                directory = './' + str(self.ident) + '/'

        message("to dir:", directory, **kwargs)
        attrs = vars(self.attrs)

        for iname in name:
            if iname in self.data and isinstance(self.data[iname], (xr.DataArray, xr.Dataset)):

                if filename is None:
                    ifilename = directory + '/' + iname + '.nc'
                else:
                    ifilename = filename

                # Create directory if necessary
                if os.path.isdir(os.path.dirname(ifilename)):
                    idir = os.path.dirname(ifilename)
                    message("makedir: ", idir, **kwargs)
                    os.makedirs(idir, exist_ok=True)  # no Errors

                message("Writing", ifilename, **kwargs)

                iobj = getattr(self.data, iname)
                if len(iobj.attrs) or add_global:
                    iobj.attrs.update(attrs)  # add global attributes

                if force:
                    iobj.to_netcdf(ifilename, mode='w', **kwargs)
                    force = False

                elif os.path.isfile(ifilename):
                    iobj.to_netcdf(ifilename, mode='a', **kwargs)

                else:
                    iobj.to_netcdf(ifilename, mode='w', **kwargs)

            else:
                print("[DATA]", iname, "no Xarray object? ", type(iobj))


def open_radiosonde(name, ident=None, filename=None, directory=None, **kwargs):
    """ Create a Radiosonde object from opening a dataset

    Args:
        name (str): used as filename and/or as data name
        ident (str): radiosonde wmo or igra id
        filename (str): filename to read from (netcdf)
        directory (str): directory of radiosonde store, default config rasodir

    Returns:
        Radiosonde : Radiosonde class object
    """
    if ident is None:
        ident = "Unknown"
    out = Radiosonde(ident)
    out.add(name, filename=filename, directory=directory, **kwargs)
    return out
