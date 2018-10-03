# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import xarray as xr
from .fun import message

__all__ = ["Radiosonde"]


def dim_summary(obj):
    if hasattr(obj, 'shape'):
        i = np.shape(obj)
    elif hasattr(obj, 'len'):
        i = len(obj)
    elif hasattr(obj, 'size'):
        i = np.size(obj)
    elif hasattr(obj, 'data_vars'):
        # Xarray Dataset
        i = "%d vars [%s]" % (len(obj.data_vars), ", ".join(["%s(%s)" % (i,j) for i,j in obj.dims.items()]))
    else:
        i = obj
    return i


def formatting(obj):
    return u'<%s (%s)>' % (type(obj).__name__, dim_summary(obj))


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return u"\n".join("%-10s : %s" % (i, formatting(getattr(self, i))) for i in self.__dict__.keys())

    def __iter__(self):
        return (x for x in self.__dict__.keys())


class Radiosonde(object):
    def __init__(self, ident, data=None, **kwargs):
        self.ident = ident
        self.data = Bunch(**data)
        self.attrs = Bunch(**kwargs)

    def __repr__(self):
        summary = u"Radiosonde (%s)\n" % self.ident
        if len(list(self.data)) > 0:
            summary += "Data: \n"
            summary += repr(self.data)
        if len(list(self.attrs)) > 0:
            summary += "\nGlobal Attributes: \n"
            summary += repr(self.attrs)
        return summary

    def __getitem__(self, item):
        return getattr(self.data, item)

    def __setitem__(self, key, value):
        setattr(self.data, key, value)

    def __delitem__(self, key):
        if key in self.data:
            delattr(self.data, key)

    def add(self, name, variable=None, filename=None, directory=None, **kwargs):
        from . import config

        if not isinstance(name, list):
            name = [name]

        if 'rasodir' in config:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir

        if variable is not None:
            if not isinstance(variable, list):
                variable = [variable]

        for iname in name:
            if filename is None:
                if self.ident is not None:
                    ifilename = directory + '/' + str(self.ident) + '/' + iname + '.nc'
                else:
                    ifilename = directory + '/' + iname + '.nc'

            else:
                ifilename = filename

            message(ifilename, mname='ADD', level=1, **kwargs)
            ds = xr.open_dataset(ifilename, **kwargs)

            if variable is not None:
                if isinstance(ds, xr.Dataset):
                    ivars = list_in_list(variable, list(ds.data_vars))
                    ds = ds[ivars]

                    if len(ds.data_vars) == 0:
                        ds = None

                    elif len(ds.data_vars) == 1:
                        ds = ds[ds.data_vars[0]]

                    else:
                        pass

                elif isinstance(ds, xr.DataArray):
                    ivars = list_in_list(variable, [ds.name])
                    if len(ivars) == 0:
                        ds = None

                else:
                    ds = None

            else:
                ds = xr.open_dataset(ifilename, **kwargs)

            if ds is not None:
                message(iname, mname='ADD', level=1, **kwargs)
                setattr(self.data, iname, ds)

        if self.ident == 'Unknown':
            print("Please change the IDENT")

    def inquire(self):
        pass

    def to_netcdf(self, name, filename=None, directory=None):
        from . import config

        if not isinstance(name, list):
            name = [name]

        if 'directory' in self.attrs:
            directory = getattr(self.attrs, 'directory', '.')

        if 'rasodir' in config:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir

        attrs = vars(self.attrs)

        for iname in name:
            if iname in self.data:
                if filename is None:
                    ifilename = directory + '/' + iname + '.nc'
                else:
                    ifilename = filename
                iobj = getattr(self.data, iname)
                iobj.attrs.update(attrs)  # add attributes
                if hasattr(iobj, 'to_netcdf'):
                    iobj.to_netcdf(ifilename)
                else:
                    print("Object", iname, "has no to_netcdf", type(iobj))

    def dump(self, name=None, filename=None, directory=None):
        from . import config

        if 'rasodir' in config:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir

        if 'directory' in self.attrs:
            directory = getattr(self.attrs, 'directory', '.')

        if directory is None:
            directory = '.'

        if filename is None:
            filename = directory + '/' + name + '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def open_radiosonde(name, ident=None, variable=None, filename=None, directory=None, **kwargs):
    out = Radiosonde(ident if ident is not None else "Unknown")
    out.add(name, variable=variable, filename=filename, directory=directory, **kwargs)
    return out


def load_radiosonde(filename, directory=None):

    if directory is None:
        filename = directory + '/' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def list_in_list(jlist, ilist):
    out = []
    for ivar in jlist:
        if '*' in ivar:
            new = [jvar for jvar in ilist if ivar.replace('*','') in jvar]
            out.extend(new)
        elif ivar in ilist:
            out.append(ivar)
        else:
            pass
    return out
