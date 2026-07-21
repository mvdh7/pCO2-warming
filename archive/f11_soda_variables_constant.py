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
from takahashi93 import temperature as ex_temperature

# opt_k_carbonic = 10
opt_total_borate = 1
variable_var = "dic"

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

alkalinity = soda.talk.mean("time").data
if not variable_var == "alkalinity":
    alkalinity = [np.nanmean(alkalinity)]
dic = soda.dic.mean("time").data
if not variable_var == "dic":
    dic = [np.nanmean(dic)]
temperature = soda.temperature.mean("time").data
if not variable_var == "temperature":
    temperature = [np.nanmean(temperature)]
salinity = soda.salinity.mean("time").data
if not variable_var == "salinity":
    salinity = [np.nanmean(salinity)]

for opt_k_carbonic in [10]:  # range(1, 19):
    print(opt_k_carbonic)
    # Calculate surface field of dlnpCO2/dT
    results = pyco2.sys(
        par1=alkalinity,
        par2=dic,
        par1_type=1,
        par2_type=2,
        temperature=temperature,
        salinity=salinity,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        grads_of=["pCO2", *grads],
        grads_wrt=["temperature", *grads],
    )
    soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        results["dlnpCO2_dT"] * 100,
    )
    pCO2_wf = results["d_pCO2__d_temperature"] / results["pCO2"]
    pCO2_wf_components = {}
    pCO2_wf_percent = {}
    pCO2_wf_sum = 0
    pCO2_wf_percent_sum = 0
    for k in grads:
        k_comp = (
            results["d_" + k + "__d_temperature"] * results["d_pCO2__d_" + k]
        ) / results["pCO2"]
        pCO2_wf_components[k] = k_comp
        soda[k] = (("lat", "lon"), k_comp)
        pCO2_wf_percent[k] = 100 * k_comp / pCO2_wf
        pCO2_wf_sum += k_comp
        pCO2_wf_percent_sum += pCO2_wf_percent[k]

    # Calculate mean surface ocean conditions
    mean_surface = {gvar: soda[gvar].mean() for gvar in gvars}
    mean_k_percent = {k: np.nanmean(v) for k, v in pCO2_wf_percent.items()}
    std_k_percent = {k: np.nanstd(v) for k, v in pCO2_wf_percent.items()}
    p95_k_percent = {
        k: [np.percentile(v[~np.isnan(v)], 0.01), np.percentile(v[~np.isnan(v)], 0.99)]
        for k, v in pCO2_wf_percent.items()
    }

    # Simulate T93 experiment across the globe
    fits_linear = np.full_like(
        soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    fits_poly_const = np.full_like(
        soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    fits_poly_Tcoeff = np.full_like(
        soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    ex_pCO2 = np.full(
        (
            *soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].shape,
            ex_temperature.size,
        ),
        np.nan,
    )
    for i, t in enumerate(ex_temperature):
        print(i)
        ex_pCO2[:, :, i] = pyco2.sys(
            par1=alkalinity,
            par2=dic,
            par1_type=1,
            par2_type=2,
            temperature=t,
            salinity=salinity,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
        )["pCO2"]
    for i in range(180):
        print(i)
        for j in range(360):
            if ~np.isnan(ex_pCO2[i, j, 0]):
                ex_lnpCO2 = np.log(ex_pCO2[i, j, :])
                ex_linear = linregress(ex_temperature, ex_lnpCO2)
                fits_linear[i, j] = ex_linear.slope
                ex_poly = np.polyfit(ex_temperature, ex_lnpCO2, 2)
                fits_poly_const[i, j], fits_poly_Tcoeff[i, j] = ex_poly[:2]
    soda["fits_linear_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        fits_linear * 100,
    )

# %% Visualise - map
all_polyfits = []

for opt_k_carbonic in [10]:  # range(1, 19):
    # Get axis limits
    pt = soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    pt = pt[~np.isnan(pt)]
    fl = soda["fits_linear_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    fl = fl[~np.isnan(fl)]
    xlims = (
        min(np.quantile(fl, 0.005), np.quantile(pt, 0.005)),
        max(np.quantile(fl, 0.995), np.quantile(pt, 0.995)),
    )
    fig, axs = plt.subplots(
        dpi=300,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
        figsize=[17.4 / 2.54, 7 / 2.54],
        ncols=2,
    )
    for i, fvar in enumerate(
        [
            "dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic),
            "fits_linear_{:02.0f}".format(opt_k_carbonic),
        ]
    ):
        # fig, ax = plt.subplots(
        #     dpi=300, subplot_kw={"projection": ccrs.Robinson(central_longitude=205)}
        # )
        ax = axs[i]
        fm = soda[fvar].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            # vmin=3.95, vmax=4.7
            vmin=xlims[0],
            vmax=xlims[1],
        )
        ax.contour(
            "lon",
            "lat",
            fvar,
            [4.23],
            data=soda,
            colors="w",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
        )
        if fvar.startswith("dlnpCO2"):
            eta_type = "theoretical"
            letter = "(a)"
        elif fvar.startswith("fits_linear"):
            eta_type = "experimental, linear"
            letter = "(b)"
        plt.colorbar(
            fm,
            location="bottom",
            label="$η$ ({})".format(eta_type) + " / $10^{-2}$ °C$^{–1}$",
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
    fig.savefig("figures/f11_map_both_{}.png".format(variable_var))
    plt.show()
    plt.close()
