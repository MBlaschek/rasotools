from .lists import *
from .igra import *
from .mars import *
from . import info


def example_data():
    """ Example Radiosonde Dataset for tests

    Returns:
        Dataset
    """
    from .. import get_data
    from xarray import open_dataset
    return open_dataset(get_data('AUM00011035_XARRAY.nc'))

