"""
RASOTOOLS applies raso to xarray
"""
__version__ = '0.2'
__author__ = 'MB'
__status__ = 'dev'
__date__ = 'Do Nov 22 16:45:24 CET 2018'
__institute__ = 'Univie, IMGW'
__github__ = 'git@github.com:MBlaschek/rasotools.git'
__doc__ = """
Radiosonde Tools Collection v%s
Maintained by %s at %s
Github: %s [%s]
Updated: %s
""" % (__version__, __author__, __institute__, __github__, __status__, __date__)
import os as _os
from .Radiosonde import Bunch
from .Radiosonde import *
from .Network import *
from . import io
from . import grid
from . import fun
from . import bp
from . import plot


def _getlibs():
    "Get library version information for printing"
    import numpy
    import pandas
    import xarray
    return __version__, numpy.__version__, pandas.__version__, xarray.__version__


config = Bunch()
config.homedir = _os.getenv("HOME")
config.wkdir = _os.getcwd()
config.igradir = ''
config.marsdir = ''
config.outdir = _os.getcwd() + '/results'
config.rasodir = _os.getcwd() + '/raso_archive'
config.std_plevels = [1000., 2000., 3000., 5000., 7000., 10000., 15000., 20000., 25000., 30000., 40000., 50000., 70000.,
                      85000., 92500., 100000.]
config.era_plevels = [1000., 2000., 3000., 5000., 7000., 10000., 12500., 15000., 17500., 20000., 22500., 25000., 30000.,
                      35000., 40000., 45000., 50000., 55000., 60000., 65000., 70000., 75000., 77500., 80000., 82500.,
                      85000., 87500., 90000., 92500., 95000., 97500., 100000.]

config.rttov_profile_limits = None
config.month_to_season = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                          10: 'SON', 11: 'SON', 12: 'DJF'}
config.libinfo = "RT(%s) NP(%s) PD(%s) XR(%s)" % _getlibs()


def get_data(path=None):
    """
    get a filename form the Module AX directory of the module

    Parameters
    ----------
    path : str
        filename

    Returns
    -------
    str
         path to the file
    """
    if path is None:
        print("Choose one: ")
        print("\n".join(_os.listdir(_os.path.join(_os.path.abspath(_os.path.dirname(__file__)), 'ax'))))
    else:
        return _os.path.join(_os.path.abspath(_os.path.dirname(__file__)), 'ax', path)


def load_config(filename='rasoconfig.json'):
    """ load config from JSON file

    Parameters
    ----------
    filename : str
        Filename of config file

    Example File
    ------------
    {
    "marsdir": "",
    "igradir": "",
    "rasodir": "/home/mblaschek/workspace/raso_archive"
    }
    """
    import os
    import json

    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            variables = json.loads(file.read())

        for ikey, ival in variables.items():
            setattr(config, ikey, ival)

        print("Configuration loaded: ", filename)


def dump_config(filename='rasoconfig.json'):
    """ Write config to file

    Parameters
    ----------
    filename : str
        Filename to save config
    """
    import json
    variables = vars(config)
    with open(filename, 'w') as file:
        file.write(json.dumps(variables, indent=0))

    print("Configuration written: ", filename)


def load_rttov():
    """ Load RTTOV profile limits for quality checks

    """
    import os
    import pandas as pd
    filename = get_data('rttov_54L_limits.csv')
    if os.path.isfile(filename):
        setattr(config, 'rttov_profile_limits', pd.read_csv(filename))
