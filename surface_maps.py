import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")

fig, axs = plt.subplots(
    dpi=300,
    nrows=2,
    ncols=2,
    figsize=(18 / 2.54, 14 / 2.54),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
)
letters = ["a", "b", "c", "d"]
fvars = ["dic", "talk", "temperature", "salinity"]
vlims = [[1900, 2150], [2150, 2450], [-2, 30], [31.5, 37.5]]
cmaps = ["viridis", "viridis", "RdBu_r", "viridis"]
labels = [
    "DIC / µmol kg$^{-1}$",
    "Alkalinity / µmol kg$^{-1}$",
    "Temperature / °C",
    "Practical salinity",
]
for i, ax in enumerate(axs.ravel()):
    ax.set_facecolor("xkcd:silver")
    fm = (
        soda_monthly[fvars[i]]
        .mean("month")
        .plot(
            ax=ax,
            vmin=vlims[i][0],
            vmax=vlims[i][1],
            cmap=cmaps[i],
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
    )
    ax.text(0, 1, "(" + letters[i] + ")", transform=ax.transAxes)
    plt.colorbar(
        fm,
        location="bottom",
        label=labels[i],
        pad=0.05,
        aspect=20,
        fraction=0.04,
        extend="both",
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor=0.1 * np.array([1, 1, 1]),
    )
fig.tight_layout()
fig.savefig("figures/surface_maps.png")
