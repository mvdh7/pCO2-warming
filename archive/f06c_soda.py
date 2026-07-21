from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import xarray as xr
import numpy as np
from scipy.stats import linregress

# from scipy.optimize import least_squares
import pwtools

# opt_k_carbonic = 10
opt_total_borate = 1

# Import OceanSODA
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)
gvars = ["temperature", "salinity", "talk", "dic"]
grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
]

for opt_k_carbonic in [10]:  # range(1, 19):
    print(opt_k_carbonic)
    # Calculate surface field of dlnpCO2/dT
    results = pyco2.sys(
        par1=soda.talk.mean("time").data,
        par2=soda.dic.mean("time").data,
        par1_type=1,
        par2_type=2,
        temperature=soda.temperature.mean("time").data,
        salinity=soda.salinity.mean("time").data,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        grads_of=["pCO2", *grads],
        grads_wrt=["temperature", *grads],
    )
    soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        results["dlnpCO2_dT"] * 1e3,
    )

    # Calculate ups_h across the globe from the T93 fit
    soda["upsh_{:02.0f}".format(opt_k_carbonic)] = (
        pwtools.get_eta_h(28963, soda.temperature)
    ).mean("time") * 1000

# # %% Quick load
# soda = xr.open_dataset("quickload/f06_soda.zarr", engine="zarr")

# %% Visualise - map
all_polyfits = []

for opt_k_carbonic in [10]:  # range(1, 19):
    # Get axis limits
    pt = soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    pt = pt[~np.isnan(pt)]
    fl = soda["upsh_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    fl = fl[~np.isnan(fl)]
    xlims = (
        min(np.quantile(fl, 0.005), np.quantile(pt, 0.005)),
        max(np.quantile(fl, 0.995), np.quantile(pt, 0.995)),
    )
    xlims = (39, 47)
    fig, axs = plt.subplots(
        dpi=300,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
        figsize=[17.4 / 2.54, 7 / 2.54],
        ncols=2,
    )
    for i, fvar in enumerate(
        [
            "dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic),
            "upsh_{:02.0f}".format(opt_k_carbonic),
        ]
    ):
        ax = axs[i]
        fm = soda[fvar].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            vmin=xlims[0],
            vmax=xlims[1],
        )
        ax.contour(
            "lon",
            "lat",
            fvar,
            [42.3],
            data=soda,
            colors="w",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
        )
        if fvar.startswith("dlnpCO2"):
            letter = "(a)"
            cblabel = (
                r"$υ_\mathrm{" + pwtools.okc_codes[opt_k_carbonic] + "}$ / k°C$^{–1}$"
            )
        elif fvar.startswith("upsh"):
            letter = "(b)"
            cblabel = "$υ_h$ from Ta93 fit / k°C$^{-1}$"
        plt.colorbar(
            fm,
            location="bottom",
            label=cblabel,
            pad=0.05,
            aspect=20,
            fraction=0.05,
            extend="both",
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "land", "50m"),
            facecolor=0.1 * np.array([1, 1, 1]),
        )
        ax.text(0, 1, letter, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig("figures/f06/f06c_map_both_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()
