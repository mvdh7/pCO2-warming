from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import xarray as xr
import numpy as np


# Import OceanSODA
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)

# %% Calculate surface field of dlnpCO2/dT
results_06 = pyco2.sys(
    par1=soda.talk.mean("time").data,
    par2=soda.dic.mean("time").data,
    par1_type=1,
    par2_type=2,
    temperature=soda.temperature.mean("time").data,
    salinity=soda.salinity.mean("time").data,
    opt_k_carbonic=6,
    opt_total_borate=1,
)
results_10 = pyco2.sys(
    par1=soda.talk.mean("time").data,
    par2=soda.dic.mean("time").data,
    par1_type=1,
    par2_type=2,
    temperature=soda.temperature.mean("time").data,
    salinity=soda.salinity.mean("time").data,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
soda["dlnpCO2_dT_06"] = (("lat", "lon"), results_06["dlnpCO2_dT"] * 1e3)
soda["dlnpCO2_dT_10"] = (("lat", "lon"), results_10["dlnpCO2_dT"] * 1e3)
fig, ax = plt.subplots(dpi=300)
soda.dlnpCO2_dT_10.plot(vmin=35, vmax=45, ax=ax)
fig, ax = plt.subplots(dpi=300)
soda.dlnpCO2_dT_06.plot(vmin=35, vmax=45, ax=ax)
