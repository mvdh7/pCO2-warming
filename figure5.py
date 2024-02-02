from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt, dates as mdates
from cartopy import crs as ccrs, feature as cfeature
import xarray as xr
import numpy as np
from scipy.stats import linregress
import pwtools

use_quickload = True
opt_k_carbonic = 10
opt_total_borate = 1

# Import OceanSODA
soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")
gvars = ["temperature", "salinity", "talk", "dic"]
grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
]

# Calculate surface field of dlnpCO2/dT
results = pyco2.sys(
    par1=soda_monthly.talk.data,
    par2=soda_monthly.dic.data,
    par1_type=1,
    par2_type=2,
    temperature=soda_monthly.temperature.data,
    salinity=soda_monthly.salinity.data,
    opt_k_carbonic=opt_k_carbonic,
    opt_total_borate=opt_total_borate,
    opt_buffers_mode=0,
    grads_of=["pCO2", *grads],
    grads_wrt=["temperature", *grads],
)
soda_monthly["dlnpCO2_dT"] = (
    ("month", "lat", "lon"),
    results["dlnpCO2_dT"] * 100,
)

# Calculate ups_h across the globe from the T93 fit
soda_monthly["upsh"] = (
    pwtools.get_eta_h(pwtools.bh_best, soda_monthly.temperature)
) * 100

# %% Import OceanSODA-ETZH, full dataset (this takes about )
if not use_quickload:
    soda = xr.open_dataset(
        "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
        + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
    )
    soda["year"] = soda.time.to_pandas().dt.year
    soda_years = np.unique(soda.year.data)
    soda_datenum = mdates.date2num(soda.time)

    # Calculate with PyCO2SYS - global mean
    results = pyco2.sys(
        par1=soda.dic.mean(["lat", "lon"]).data,
        par2=soda.talk.mean(["lat", "lon"]).data,
        par1_type=2,
        par2_type=1,
        temperature=soda.temperature.mean(["lat", "lon"]).data,
        salinity=soda.salinity.mean(["lat", "lon"]).data,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        opt_buffers_mode=0,
    )
    soda["dlnpCO2_dT_time"] = (("time"), results["dlnpCO2_dT"])

    # Compute seasonal range
    soda_total_range = np.full_like(soda.dic.isel(time=0).data, np.nan)
    soda_seasonal_range = np.full_like(soda.dic.isel(time=0).data, np.nan)
    soda_seasonal_range_std = np.full_like(soda.dic.isel(time=0).data, np.nan)
    soda_trend = np.full_like(soda.dic.isel(time=0).data, np.nan)
    for a in range(0, soda.lat.size):
        print(a + 1, "/", soda.lat.size)
        for o in range(0, soda.lon.size):
            sodap = soda.isel(lat=a, lon=o)
            if not sodap.dic.isnull().all():
                eta = pyco2.sys(
                    par1=sodap.dic.data,
                    par2=sodap.talk.data,
                    par1_type=2,
                    par2_type=1,
                    temperature=sodap.temperature.data,
                    salinity=sodap.salinity.data,
                    opt_k_carbonic=opt_k_carbonic,
                    opt_total_borate=opt_total_borate,
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
    soda[
        [
            "year",
            "dlnpCO2_dT_time",
            "total_range",
            "seasonal_range",
            "seasonal_range_std",
            "trend",
        ]
    ].to_zarr("quickload/soda_figure5.zarr")
else:
    soda = xr.open_dataset("quickload/soda_figure5.zarr", engine="zarr")

# %% Visualise - map
# Get axis limits
pt = soda_monthly.dlnpCO2_dT.mean("month").to_numpy().ravel()
pt = pt[~np.isnan(pt)]
fl = soda_monthly.upsh.mean("month").to_numpy().ravel()
fl = fl[~np.isnan(fl)]
xlims = (
    min(np.quantile(fl, 0.005), np.quantile(pt, 0.005)),
    max(np.quantile(fl, 0.995), np.quantile(pt, 0.995)),
)
xlims = (3.9, 4.7)
fig, axs = plt.subplots(
    dpi=300,
    subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
    figsize=[17.4 / 2.54, 14 / 2.54],
    ncols=2,
    nrows=2,
)
for i, fvar in enumerate(["dlnpCO2_dT", "upsh"]):
    ax = axs[0, i]
    fm = (
        soda_monthly[fvar]
        .mean("month")
        .plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            vmin=xlims[0],
            vmax=xlims[1],
        )
    )
    ax.contour(
        "lon",
        "lat",
        fvar,
        [4.23],
        data=soda_monthly.mean("month"),
        colors="w",
        transform=ccrs.PlateCarree(),
        alpha=0.5,
    )
    if fvar.startswith("dlnpCO2"):
        letter = "(a)"
        cblabel = (
            r"$υ_\mathrm{" + pwtools.okc_codes[opt_k_carbonic] + "}$ / % °C$^{–1}$"
        )
    elif fvar.startswith("upsh"):
        letter = "(b)"
        cblabel = "$υ_h$ / % °C$^{-1}$"
    plt.colorbar(
        fm,
        location="bottom",
        label=cblabel,
        pad=0.05,
        aspect=20,
        fraction=0.05,
        extend="both",
        ticks=(4, 4.2, 4.4, 4.6),
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor="xkcd:dark",
    )
    ax.text(0, 1, letter, transform=ax.transAxes)

# Seasonal range and trend
ax = axs[1, 0]
fm = (soda["seasonal_range"] * 1e2).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=0,
    vmax=0.4,
)
plt.colorbar(
    fm,
    location="bottom",
    label=r"Seasonal range in $υ_\mathrm{Lu00}$ / % °C$^{–1}$",
    pad=0.05,
    aspect=20,
    fraction=0.05,
    extend="max",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor="xkcd:dark",
)
ax.text(0, 1, "(c)", transform=ax.transAxes)
ax = axs[1, 1]
fm = (soda["trend"] * 1e2 * 365.25).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    vmin=-0.003,
    vmax=0,
    cmap="Reds_r",
)
plt.colorbar(
    fm,
    location="bottom",
    label=r"Trend in $υ_\mathrm{Lu00}$ / % °C$^{–1}$ yr$^{-1}$",
    pad=0.05,
    aspect=20,
    fraction=0.05,
    extend="both",
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor="xkcd:dark",
)
ax.text(0, 1, "(d)", transform=ax.transAxes)

fig.tight_layout()
fig.savefig("figures_final/figure5.png")
