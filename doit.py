import numpy as np
import pandas as pd
import xarray as xr
import rasotools as rt

rt.config.wkdir = rt.config.homedir + '/workspace/run'
rt.config.outdir = rt.config.homedir + '/workspace/run/results'
rt.config.rasodir = rt.config.homedir + '/workspace/raso_archive'

b = xr.DataArray(range(10), coords={'time': pd.date_range('1-1-1991',periods=10)}, dims=['time'])
a = rt.Radiosonde('011035',data={'temp': b})
try:
    a['MARS'] = xr.open_dataset(rt.config.rasodir + '/011035/MARS_combined.nc')
except:
    print('NO MARS DATA Found')
