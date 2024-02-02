from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import pwtools

soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")
temperature_out = soda_monthly.temperature.mean("month").mean().data

# Convert with PyCO2SYS from TA and DIC
results_calc12 = pyco2.sys(
    par1=soda_monthly.talk.data,
    par1_type=1,
    par2=soda_monthly.dic.data,
    par2_type=2,
    salinity=soda_monthly.salinity.data,
    temperature=soda_monthly.temperature.data,
    temperature_out=temperature_out,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
soda_monthly["fCO2_dt_calc12"] = (
    ("month", "lat", "lon"),
    results_calc12["fCO2_out"] - results_calc12["fCO2"] + soda_monthly.fCO2.data,
)

# Convert with the various upsilon options
opt_at = [5, 1, 2, 6, 4]
results_dt = pyco2.sys(
    par1=soda_monthly.fCO2.data,
    par1_type=5,
    salinity=soda_monthly.salinity.data,
    temperature=soda_monthly.temperature.data,
    temperature_out=temperature_out,
    opt_k_carbonic=10,
    opt_total_borate=1,
    opt_adjust_temperature=np.reshape(np.array([[[opt_at]]]), [len(opt_at), 1, 1, 1]),
    bh_upsilon=soda_monthly.bh.data,
    opt_which_fCO2_insitu=1,
)
for i, at in enumerate(opt_at):
    soda_monthly["fCO2_dt_{}".format(at)] = (
        ("month", "lat", "lon"),
        results_dt["fCO2_out"][i],
    )
    soda_monthly["fCO2_dt_{}_diff".format(at)] = (
        soda_monthly["fCO2_dt_{}".format(at)] - soda_monthly.fCO2_dt_calc12
    )

# %% Print out statistics
for i, at in enumerate(opt_at):
    dtnp = soda_monthly["fCO2_dt_{}_diff".format(at)].to_numpy()
    dtnp = dtnp[~np.isnan(dtnp)]
    print("opt_adjust_temperature = {}".format(at))
    print(
        "Mean bias = {:.2f} µatm".format(
            soda_monthly["fCO2_dt_{}_diff".format(at)].mean().data
        )
    )
    print(
        "StD. bias = {:.2f} µatm".format(
            soda_monthly["fCO2_dt_{}_diff".format(at)].std().data
        )
    )
    print(
        "RMSD bias = {:.2f} µatm".format(
            np.sqrt((soda_monthly["fCO2_dt_{}_diff".format(at)] ** 2).mean().data)
        )
    )
    print(" 2.5% = {:.2f} µatm".format(np.percentile(dtnp, 2.5)))
    print("97.5% =  {:.2f} µatm".format(np.percentile(dtnp, 97.5)))
    print()

# %% Visualise
fig, axs = plt.subplots(
    dpi=300,
    nrows=2,
    figsize=(9 / 2.54, 13 / 2.54),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
)
letters = ["c", "d"]
abs_vmin_vmax = 10
for i, at in enumerate([5, 1]):
    ax = axs[i]
    ax.set_facecolor("xkcd:silver")
    fm = (
        soda_monthly["fCO2_dt_{}_diff".format(at)]
        .mean("month")
        .plot(
            ax=ax,
            vmin=-abs_vmin_vmax,
            vmax=abs_vmin_vmax,
            cmap="RdBu_r",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
    )
    soda_monthly.temperature.mean("month").plot.contour(
        levels=[temperature_out],
        colors="xkcd:dark",
        transform=ccrs.PlateCarree(),
        ax=ax,
        alpha=0.15,
    )
    ax.text(0, 1, "(" + letters[i] + ")", transform=ax.transAxes)
    if i == 0:
        plt.colorbar(
            fm,
            location="bottom",
            label="Bias in {f}CO$_2$($t_1$) / µatm".format(f=pwtools.f),
            pad=0.1,
            aspect=20,
            fraction=0.04,
            extend="both",
        )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor=0.1 * np.array([1, 1, 1]),
    )
fig.tight_layout()
fig.savefig("figures_final/figure5_part2.png")
