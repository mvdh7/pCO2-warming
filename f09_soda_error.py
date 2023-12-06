from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

opt_k_carbonic = 10

all_polyfits = np.genfromtxt("quickload/f06_all_polyfits.txt")

# Import OceanSODA-ETZH
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)

# Set temperature change in °C - positive value means convert to warmer temperature
dT = -1

# Calculate "true" change in pCO2 with PyCO2SYS
results = pyco2.sys(
    par1=soda.talk.mean("time").data,
    par2=soda.dic.mean("time").data,
    par1_type=1,
    par2_type=2,
    temperature=soda.temperature.mean("time").data,
    temperature_out=soda.temperature.mean("time").data + dT,
    salinity=soda.salinity.mean("time").data,
    opt_k_carbonic=opt_k_carbonic,
)
soda["pCO2"] = (("lat", "lon"), results["pCO2"])
soda["pCO2_w_pyco2"] = (("lat", "lon"), results["pCO2_out"])
soda["pCO2_w_pyco2_diff"] = soda.pCO2_w_pyco2 - soda.pCO2

# %% T93 linear equation
soda["pCO2_w_linear"] = np.exp(np.log(soda.pCO2) + 0.0423 * dT)
soda["pCO2_w_linear_diff"] = soda.pCO2_w_linear - soda.pCO2
soda["pCO2_w_linear_error"] = (soda.pCO2_w_linear - soda.pCO2_w_pyco2) / dT

# T93 quadratic equation
T = soda.temperature.mean("time")
soda["pCO2_w_quad"] = soda.pCO2 * np.exp(
    0.0433 * dT - 4.35e-5 * ((T + dT) ** 2 - T**2)
)
soda["pCO2_w_quad_diff"] = soda.pCO2_w_quad - soda.pCO2
soda["pCO2_w_quad_error"] = (soda.pCO2_w_quad - soda.pCO2_w_pyco2) / dT

# Linear equation but with T-dependence of eta following f06
ltfunc = np.polyval(all_polyfits[opt_k_carbonic - 1], T) / 100
ltfunc_w = np.polyval(all_polyfits[opt_k_carbonic - 1], T + dT) / 100

soda["pCO2_w_ltfunc"] = np.exp(np.log(soda.pCO2) + (ltfunc + ltfunc_w) * dT / 2)
soda["pCO2_w_ltfunc_diff"] = soda.pCO2_w_ltfunc - soda.pCO2
soda["pCO2_w_ltfunc_error"] = (soda.pCO2_w_ltfunc - soda.pCO2_w_pyco2) / dT

fig, axs = plt.subplots(
    dpi=300,
    nrows=3,
    figsize=(8.7 / 2.54, 16.5 / 2.54),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
)
letters = ["a", "b", "c"]
for i, v in enumerate(
    ["pCO2_w_linear_error", "pCO2_w_quad_error", "pCO2_w_ltfunc_error"]
):
    ax = axs[i]
    fm = soda[v].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        vmin=-1.5,
        vmax=1.5,
        cmap="RdBu_r",
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor=0.1 * np.array([1, 1, 1]),
    )
    ax.text(0, 1, "(" + letters[i] + ")", transform=ax.transAxes)
plt.colorbar(
    fm,
    location="bottom",
    label="$T$-conversion error / µatm °C$^{-1}$",
    pad=0.08,
    aspect=20,
    fraction=0.04,
    extend="both",
)
fig.tight_layout()
fig.savefig("figures/f09/f09_maps_{:02.0f}.png".format(opt_k_carbonic))

# %% Histograms
fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 8 / 2.54))
bins = np.arange(-1.5, 1.5, 0.05)
ax.hist(
    soda.pCO2_w_linear_error.data.ravel(),
    bins=bins,
    facecolor="xkcd:navy",
    label="T93 linear",
)
ax.hist(
    soda.pCO2_w_quad_error.data.ravel(),
    bins=bins,
    facecolor="xkcd:azure",
    alpha=0.7,
    label="T93 quadratic",
)
ax.hist(
    soda.pCO2_w_ltfunc_error.data.ravel(),
    bins=bins,
    facecolor="xkcd:tangerine",
    alpha=0.7,
    label="$T$-varying linear",
)
ax.set_xlim((-1.5, 1.5))
ax.axvline(0, c="k", lw=0.8)
ax.legend()
ax.set_xlabel("$T$-conversion error / µatm °C$^{-1}$")
ax.set_ylabel("Frequency / $10^3$")
ax.set_ylim([0, 25000])
ax.set_yticks([0, 5000, 10000, 15000, 20000, 25000])
ax.set_yticklabels([0, 5, 10, 15, 20, 25])
fig.tight_layout()
fig.savefig("figures/f09/f09_histogram.png")
