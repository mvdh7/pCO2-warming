from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from scipy.stats import linregress
from matplotlib import dates as mdates, pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

variable_var = "alkalinity"

# Import OceanSODA-ETZH
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)
soda["year"] = soda.time.to_pandas().dt.year
soda_years = np.unique(soda.year.data)
soda_datenum = mdates.date2num(soda.time)

alkalinity = soda.talk.mean(("lat", "lon")).data
if not variable_var == "alkalinity":
    alkalinity = [np.nanmean(alkalinity)]
dic = soda.dic.mean(("lat", "lon")).data
if not variable_var == "dic":
    dic = [np.nanmean(dic)]
temperature = soda.temperature.mean(("lat", "lon")).data
if not variable_var == "temperature":
    temperature = [np.nanmean(temperature)]
salinity = soda.salinity.mean(("lat", "lon")).data
if not variable_var == "salinity":
    salinity = [np.nanmean(salinity)]

# Calculate with PyCO2SYS - global mean
results = pyco2.sys(
    par1=dic,
    par2=alkalinity,
    par1_type=2,
    par2_type=1,
    temperature=temperature,
    salinity=salinity,
    opt_k_carbonic=10,
    opt_buffers_mode=0,
)
soda["dlnpCO2_dT_time"] = (("time"), results["dlnpCO2_dT"])

# %% Calculate with PyCO2SYS
soda_total_range = np.full_like(soda.dic.isel(time=0).data, np.nan)
soda_seasonal_range = np.full_like(soda.dic.isel(time=0).data, np.nan)
soda_seasonal_range_std = np.full_like(soda.dic.isel(time=0).data, np.nan)
soda_trend = np.full_like(soda.dic.isel(time=0).data, np.nan)
for a in range(0, 180, 1):
    print(a, "/", 180)
    for o in range(0, 360, 1):
        sodap = soda.isel(lat=a, lon=o)
        if not sodap.dic.isnull().all():
            alkalinity = sodap.talk.data
            if not variable_var == "alkalinity":
                alkalinity = [np.nanmean(alkalinity)]
            dic = sodap.dic.data
            if not variable_var == "dic":
                dic = [np.nanmean(dic)]
            temperature = sodap.temperature.data
            if not variable_var == "temperature":
                temperature = [np.nanmean(temperature)]
            salinity = sodap.salinity.data
            if not variable_var == "salinity":
                salinity = [np.nanmean(salinity)]
            soda_total_range[a, o] = 1
            eta = pyco2.sys(
                par1=dic,
                par2=alkalinity,
                par1_type=2,
                par2_type=1,
                temperature=temperature,
                salinity=salinity,
                opt_k_carbonic=10,
                opt_buffers_mode=0,
            )["dlnpCO2_dT"]
            soda_total_range[a, o] = np.max(eta) - np.min(eta)
            # Get seasonal range
            seasonal_range = np.full(sodap.year.data.shape, np.nan)
            for y in soda_years:
                L = sodap.year.data == y
                y_sodap = eta[L]
                if np.all(np.isnan(y_sodap)):
                    print(o)
                seasonal_range[L] = np.nanmax(y_sodap) - np.nanmin(y_sodap)
            soda_seasonal_range[a, o] = np.mean(seasonal_range)
            soda_seasonal_range_std[a, o] = np.std(seasonal_range)
            L = ~np.isnan(eta)
            soda_trend[a, o] = linregress(soda_datenum[L], eta[L]).slope
print("Loop complete!")
soda["total_range"] = (("lat", "lon"), soda_total_range)
soda["seasonal_range"] = (("lat", "lon"), soda_seasonal_range)
soda["seasonal_range_std"] = (("lat", "lon"), soda_seasonal_range_std)
soda["trend"] = (("lat", "lon"), soda_trend)

soda.to_zarr("quickload/f12_soda_{}.zarr".format(variable_var))

# #%%
# sodap = soda.sel(lat=55, lon=-30, method='nearest')
# results = pyco2.sys(
#     par1=sodap.dic.data,
#     par2=sodap.talk.data,
#     par1_type=2,
#     par2_type=1,
#     temperature=sodap.temperature.data,
#     salinity=sodap.salinity.data,
#     opt_k_carbonic=10,
#     opt_buffers_mode=0,
# )
# sodap["dlnpCO2_dT"] = (("time"), results["dlnpCO2_dT"])
# sodap.dlnpCO2_dT.plot()

# %% Map seasonal range
fig, ax = plt.subplots(
    dpi=300, subplot_kw={"projection": ccrs.Robinson(central_longitude=205)}
)
fm = (soda["seasonal_range"] * 100).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=0,
    vmax=0.4,  # max is 1.5, 99% is 0.3 (for seasonal_range)
)
plt.colorbar(
    fm,
    location="bottom",
    label="Seasonal range in $η$ / 10$^{-2}$ °C$^{–1}$",
    pad=0.05,
    aspect=20,
    fraction=0.03,
    extend="max",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor=0.1 * np.array([1, 1, 1]),
)
fig.tight_layout()
fig.savefig("figures/f12_seasonal_range_{}.png".format(variable_var))

# %% Map trend
fig, ax = plt.subplots(
    dpi=300, subplot_kw={"projection": ccrs.Robinson(central_longitude=205)}
)
fm = (soda["trend"] * 100 * 1e3 * 365.25).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=-3,
    vmax=0,
    cmap="Reds_r",
)
plt.colorbar(
    fm,
    location="bottom",
    label="Trend in $η$ / 10$^{-5}$ °C$^{–1}$ yr$^{-1}$",
    pad=0.05,
    aspect=20,
    fraction=0.03,
    extend="both",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor=0.1 * np.array([1, 1, 1]),
)
fig.tight_layout()
fig.savefig("figures/f12_trend_{}.png".format(variable_var))

# %% Map seasonal range and trend
fig, axs = plt.subplots(
    dpi=300,
    ncols=2,
    figsize=(17.4 / 2.54, 7 / 2.54),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
)
ax = axs[0]
fm = (soda["seasonal_range"] * 1e3).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=0,
    vmax=4,  # max is 1.5, 99% is 0.3 (for seasonal_range)
)
plt.colorbar(
    fm,
    location="bottom",
    label="Seasonal range in $η$ / kK$^{–1}$",
    pad=0.05,
    aspect=20,
    fraction=0.05,
    extend="max",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor=0.1 * np.array([1, 1, 1]),
)
ax.text(0, 1, "(c)", transform=ax.transAxes)
ax = axs[1]
fm = (soda["trend"] * 1e3 * 365.25).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=-0.03,
    vmax=0,
    cmap="Reds_r",
)
plt.colorbar(
    fm,
    location="bottom",
    label="Trend in $η$ / kK$^{–1}$ yr$^{-1}$",
    pad=0.05,
    aspect=20,
    fraction=0.05,
    extend="both",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor=0.1 * np.array([1, 1, 1]),
)
ax.text(0, 1, "(d)", transform=ax.transAxes)
fig.tight_layout()
fig.savefig("figures/f12_seasons_and_trend_{}.png".format(variable_var))
