from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

# Import OceanSODA-ETZH
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)
results = pyco2.sys(
    par1=soda.talk.mean("time").data,
    par2=soda.dic.mean("time").data,
    par1_type=1,
    par2_type=2,
    temperature=soda.temperature.mean("time").data,
    temperature_out=soda.temperature.mean("time").data + 1,
    salinity=soda.salinity.mean("time").data,
    opt_k_carbonic=10,
)

soda["pCO2"] = (("lat", "lon"), results["pCO2"])
soda["pCO2_w_pyco2"] = (("lat", "lon"), results["pCO2_out"])
soda["pCO2_w_pyco2_diff"] = soda.pCO2_w_pyco2 - soda.pCO2

# %%
soda["pCO2_w_linear"] = np.exp(np.log(soda.pCO2) + 0.0423)
soda["pCO2_w_linear_diff"] = soda.pCO2_w_linear - soda.pCO2
soda["pCO2_w_linear_error"] = soda.pCO2_w_linear - soda.pCO2_w_pyco2
soda.pCO2_w_linear_error.plot()
