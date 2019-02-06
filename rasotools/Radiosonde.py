# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import xarray as xr
from .fun import message, dict2str, kwu

__all__ = ["Radiosonde", "open_radiosonde", "load_radiosonde"]


def dim_summary(obj):
    """ Auxilliary class function for printing
    """
    if hasattr(obj, 'shape'):
        i = np.shape(obj)
    elif hasattr(obj, 'len'):
        i = len(obj)
    elif hasattr(obj, 'size'):
        i = np.size(obj)
    elif hasattr(obj, 'data_vars'):
        i = "%d vars [%s]" % (len(obj.data_vars), ", ".join(["%s(%s)" % (i, j) for i, j in obj.dims.items()]))
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
        self.aux = Bunch()

    def __repr__(self):
        summary = u"Radiosonde (%s)\n" % self.ident
        if len(list(self.data)) > 0:
            summary += "Data: \n"
            summary += repr(self.data)
        if len(list(self.attrs)) > 0:
            summary += "\nGlobal Attributes: \n"
            summary += repr(self.attrs)
        if len(list(self.aux)) > 0:
            summary += "\nAuxiliary Information: \n"
            summary += repr(self.aux)
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

    def add(self, name, variable=None, filename=None, directory=None, xwargs={}, **kwargs):
        """ Add data to radiosonde class object [container]

        Args:
            name (str): used as filename and/or as data name
            variable (str): variable names to look for
            filename (str): filename to read from (netcdf)
            directory (str): directory of radiosonde store, default config rasodir
            xwargs (dict): keyword arguments to xarray open_dataset
        """
        from . import config

        if not isinstance(name, list):
            name = [name]

        if directory is None:
            if 'rasodir' in config and os.path.isdir(config.rasodir):
                directory = config.rasodir
            else:
                directory = '.'

        if filename is not None:
            if '.nc' not in filename:
                print("Warning can read only NetCDF / Xarray open_dataset formats")

            if not os.path.isfile(filename):
                if directory is not None:
                    if not os.path.isfile(directory + '/' + filename):
                        filename = directory + '/' + str(self.ident) + '/' + filename
                    else:
                        filename = directory + '/' + filename

        if variable is not None:
            if not isinstance(variable, list):
                variable = [variable]

        for iname in name:
            if filename is None:
                if os.path.isfile(directory + '/' + str(self.ident) + '/' + iname + '.nc'):
                    ifilename = directory + '/' + str(self.ident) + '/' + iname + '.nc'
                elif os.path.isfile(directory + '/' + iname + '.nc'):
                    ifilename = directory + '/' + iname + '.nc'
                else:
                    message("Not found:", iname, directory)
                    continue

            else:
                ifilename = filename

            message("Reading ...", ifilename, **kwargs)
            ds = xr.open_dataset(ifilename, **xwargs)

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
                ds = xr.open_dataset(ifilename, **xwargs)

            if ds is not None:
                message(iname, **kwu('level', 1, **kwargs))
                setattr(self.data, iname, ds)

        if self.ident == 'Unknown':
            print("Please change the IDENT")

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

    def rename(self, old_name, new_name, **kwargs):
        if old_name in self.data:
            self.data.__dict__[new_name] = self.data.__dict__.pop(old_name)
            message(old_name, " > ", new_name, **kwargs)

    def count(self, name, dim=None, per='M', **kwargs):
        """ Count available data per time period (M)

        Args:
            name (str): dataset name
            dim (str): datetime dimension
            per (str): period (M, A,...)
            **kwargs:

        Returns:
            xr.Dataset : counts
        """
        from .met.time import count_per
        if name in self.data:
            tmp = {}
            for ivar, idata in self.data[name].data_vars.items():
                tmp[ivar] = count_per(idata, dim=dim, per=per)
            return xr.Dataset(tmp)
        message('Unknown Input', name, 'Choose: ', list(self.data), **kwargs)
        return None

    def count_times(self, name, dim=None, **kwargs):
        """ Count sounding times

        Args:
            name (str): dataset name
            dim (str): datetime dimension
            **kwargs:

        Returns:
            xr.Dataset : counts per sounding time
        """
        from .met.time import count_per_times
        if name in self.data:
            tmp = {}
            for ivar, idata in self.data[name].data_vars.items():
                if dim in idata.dims:
                    tmp[ivar] = count_per_times(idata, dim=dim)
            return xr.Dataset(tmp)
        message('Unknown Input', name, 'Choose: ', list(self.data), **kwargs)
        return None

    def list_store(self, directory=None, varinfo=False, ncinfo=False):
        import time
        from . import config
        from .io.info import view
        from .fun import print_fixed

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

    def to_netcdf(self, name, filename=None, directory=None, force=False, xargs={}, **kwargs):
        """Write each data variable to NetCDF 4

        Args:
            name (str): data name
            filename (str): filename
            directory (str): directory
            force (bool): force new file
            xargs (dict): keywords as dict to XArray function to_netcdf

        """
        from . import config

        if not isinstance(name, list):
            name = [name]

        if directory is None:
            if 'rasodir' in config:
                if os.path.isdir(config.rasodir):
                    directory = config.rasodir
            else:
                directory = '.'
        directory += '/' + str(self.ident) + '/'
        message("Output directory:", directory, **kwargs)
        attrs = vars(self.attrs)

        for iname in name:
            if iname in self.data:
                if filename is None:
                    ifilename = directory + '/' + iname + '.nc'
                else:
                    ifilename = filename

                # Create directory if necessary
                if os.path.isdir(os.path.dirname(ifilename)):
                    message("Creating directory: ", ifilename, **kwargs)
                    os.makedirs(os.path.dirname(ifilename), exist_ok=True)  # no Errors

                if force:
                    xargs['mode'] = 'w'
                    force = False

                elif os.path.isfile(ifilename):
                    xargs['mode'] = 'a'

                else:
                    xargs['mode'] = 'w'

                message("Writing", ifilename, dict2str(xargs), **kwargs)
                iobj = getattr(self.data, iname)
                iobj.attrs.update(attrs)  # add attributes

                # Xarray Object or not?
                if hasattr(iobj, 'to_netcdf'):
                    iobj.to_netcdf(ifilename, **xargs)
                else:
                    print("Object", iname, "has no to_netcdf", type(iobj))
        # fin

    def dump(self, name=None, filename=None, directory=None, **kwargs):
        """ Pickle dump the whole radiosonde class object

        Args:
            name (str): name of file or ident
            filename (str): filename
            directory (str): directory of radiosonde store, default config rasodir

        """
        from . import config

        if name is None:
            name = self.ident

        if 'rasodir' in config:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir

        if 'directory' in self.attrs:
            directory = getattr(self.attrs, 'directory', '.')

        if directory is None:
            directory = '.'

        message("Using", directory, **kwargs)

        if filename is None:
            filename = directory + '/' + name + '.pkl'

        message("Writing", filename, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)  # Pickle everything


def open_radiosonde(name, ident=None, variable=None, filename=None, directory=None, xwargs={}, **kwargs):
    """ Create a Radiosonde object from opening a dataset

    Args:
        name (str): used as filename and/or as data name
        ident (str): radiosonde wmo or igra id
        variable (list, str): variable names to look for
        filename (str): filename to read from (netcdf)
        directory (str): directory of radiosonde store, default config rasodir
        xwargs (dict): keyword arguments to xarray open_dataset

    Returns:
        Radiosonde : Radiosonde class object
    """
    if ident is None:
        ident = "Unknown"
        print("No Radiosonde Identifier specified !!!")

    out = Radiosonde(ident)
    out.add(name, variable=variable, filename=filename, directory=directory, xwargs=xwargs, **kwargs)
    return out


def load_radiosonde(name=None, filename=None, directory=None, **kwargs):
    """ Read a radiosonde class dump file

    Args:
        name (str): name of dump in rasodir
        filename (str): filename of dump file
        directory (str): directory of file (name)

    Returns:
        Radiosonde : Radiosonde class object
    """
    if name is None and filename is None:
        raise ValueError("Requires either name or filename")

    if filename is None:
        if directory is None:
            filename = directory + '/' + name + '.pkl'

    message("Reading", filename, **kwargs)
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def list_in_list(jlist, ilist):
    """ compare lists and use wildcards

    Args:
        jlist (list): list of search patterns
        ilist (list): list of available choices

    Returns:
        list : common elements
    """
    out = []
    for ivar in jlist:
        if '*' in ivar:
            new = [jvar for jvar in ilist if ivar.replace('*', '') in jvar]
            out.extend(new)
        elif ivar in ilist:
            out.append(ivar)
        else:
            pass
    return out
