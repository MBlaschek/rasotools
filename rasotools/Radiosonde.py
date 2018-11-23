# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import xarray as xr
from .fun import message

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
        return getattr(self.data, item, default=None)

    def __setitem__(self, key, value):
        if key in self.data:
            print("Warning overwriting ...", key)
        setattr(self.data, key, value)

    def __delitem__(self, key):
        if key in self.data:
            delattr(self.data, key)

    def add(self, name, variable=None, filename=None, directory=None, xwargs={}, **kwargs):
        """ Add data to radiosonde class object [container]

        Parameters
        ----------
        name : str
            used as filename and/or as data name
        variable : list / str
            variable names to look for
        filename : str
            filename to read from (netcdf)
        directory :
            directory of radiosonde store, default config rasodir
        xwargs : dict
            keyword arguments to xarray open_dataset
        kwargs : dict
            optional keyword arguments

        """
        from . import config

        if not isinstance(name, list):
            name = [name]

        if 'rasodir' in config and directory is None:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir

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

            message("Reading ...",ifilename, mname='ADD', **kwargs)
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
                ds = xr.open_dataset(ifilename, **kwargs)

            if ds is not None:
                message(iname, mname='ADD', level=1, **kwargs)
                setattr(self.data, iname, ds)

        if self.ident == 'Unknown':
            print("Please change the IDENT")

    def clear(self, exclude=None, **kwargs):
        for idata in self.data:
            if idata not in exclude:
                del self.data[idata]
                message(idata, mname='CLEAR', **kwargs)

    def rename(self, old_name, new_name, **kwargs):
        if old_name in self.data:
            self.data.__dict__[new_name] = self.data.__dict__.pop(old_name)
            message(old_name, " > ", new_name, mname='RENAME', **kwargs)

    def inquire(self, dataset, dim=None, per='M', get_counts=True, get_times=True):
        from .met.time import count_per_times, count_per

        if dataset in self.data:
            idata = self.data[dataset]
            if get_counts or get_times:
                tmp = {}
                for ivar in idata.data_vars:
                    tmp[ivar] = count_per(idata[ivar], dim=dim, per=per)
                counts = xr.Dataset(tmp)
                if not get_times:
                    return counts

            if get_times:
                tmp = {}
                for ivar in idata.data_vars:
                    if dim in idata[ivar].dims:
                        tmp[ivar] = count_per_times(idata[ivar], dim=dim)
                times = xr.Dataset(tmp)
                if not get_counts:
                    return times

            return counts, times

        return None

    def list_store(self, directory=None, varinfo=False, ncinfo=False):
        import time
        from . import config
        from .io.info import view

        if directory is None:
            directory = config.rasodir

        directory += "/%s/" % self.ident
        if os.path.isdir(directory):
            summe = 0
            print("Available Files (Datasets): ")
            print(directory)
            for ifile in os.listdir(directory):
                if os.path.isdir(directory + ifile):
                    continue

                current = os.path.getsize(directory + ifile) / 1024 / 1024
                itime = time.ctime(os.path.getctime(directory + ifile))
                print("%20s : %4d MB  : %s" % (ifile, current, itime))
                if '.nc' in ifile:
                    if varinfo:
                        print('_' * 80)
                        with xr.open_dataset(directory + ifile) as f:
                            print("Variables: ", ", ".join(list(f.data_vars)))
                        print('_' * 80)
                    elif ncinfo:
                        print('_' * 80)
                        view(directory + ifile)
                        print('_' * 80)
                    else:
                        pass

                summe += current

            print("\n%20s : %4d MB" % ('Sum', summe))
        else:
            print("Store not found!")

    def to_netcdf(self, name, filename=None, directory=None, force=False, xargs={}, **kwargs):
        """ Write each data variable to NetCDF 4

        Parameters
        ----------
        name : str
            data name
        filename : str
            filename
        directory : str
            directory to write to, default: rasodir in config
        force : bool
            force new file
        xargs : dict
            keywords as dict to Xarray function
        Returns
        -------

        """
        from . import config

        if not isinstance(name, list):
            name = [name]

        if 'rasodir' in config:
            if os.path.isdir(config.rasodir):
                directory = config.rasodir + '/' + str(self.ident) + '/'
                message("Using", directory, mname='NC', **kwargs)

        attrs = vars(self.attrs)

        for iname in name:
            if iname in self.data:
                if filename is None:
                    ifilename = directory + '/' + iname + '.nc'
                else:
                    ifilename = filename

                if force:
                    xargs['mode'] = 'w'
                    force = False
                else:
                    xargs['mode'] = 'a'

                message("Writing", ifilename, mname='NC', level=1, **kwargs)
                iobj = getattr(self.data, iname)
                iobj.attrs.update(attrs)  # add attributes
                if hasattr(iobj, 'to_netcdf'):
                    iobj.to_netcdf(ifilename, **xargs)
                else:
                    print("Object", iname, "has no to_netcdf", type(iobj))

    def dump(self, name=None, filename=None, directory=None, **kwargs):
        """ Pickle dump the whole radiosonde class object

        Parameters
        ----------
        name : str
            name of file or ident
        filename : str
            filename
        directory : str
            directory of radiosonde store, default config rasodir
        kwargs : dict
            optional keyword arguments

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

        message("Using", directory, mname='PKL', **kwargs)

        if filename is None:
            filename = directory + '/' + name + '.pkl'

        message("Writing", filename, mname='PKL', **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)   # Pickle everything


def open_radiosonde(name, ident=None, variable=None, filename=None, directory=None, xwargs={}, **kwargs):
    """ Create a Radiosonde object from opening a dataset

    Parameters
    ----------
    name : str
        used as filename and/or as data name
    ident : str
        radiosonde wmo or igra id
    variable : list / str
        variable names to look for
    filename : str
        filename to read from (netcdf)
    directory :
        directory of radiosonde store, default config rasodir
    xwargs : dict
        keyword arguments to xarray open_dataset
    kwargs : dict
        optional keyword arguments

    Returns
    -------
    Radiosonde
    """
    if ident is None:
        ident = "Unknown"
        print("No Radiosonde Identifier specified !!!")

    out = Radiosonde(ident)
    out.add(name, variable=variable, filename=filename, directory=directory, xwargs=xwargs, **kwargs)
    return out


def load_radiosonde(name=None, filename=None, directory=None, **kwargs):
    """ Read a radiosonde class dump file

    Parameters
    ----------
    name : str
        name of dump in rasodir
    filename : str
        filename of dump file
    directory : str
        directory of file (name)
    kwargs : dict

    Returns
    -------
    Radiosonde
    """
    if name is None and filename is None:
        raise ValueError("Requires either name or filename")

    if filename is None:
        if directory is None:
            filename = directory + '/' + name + '.pkl'

    message("Reading", filename, mname='PKL', **kwargs)
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def list_in_list(jlist, ilist):
    """ compare lists and use wildcards

    Parameters
    ----------
    jlist : list
        list of search patterns
    ilist : list
        list of available choices

    Returns
    -------
    list
        common elements
    """
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
