from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import pwtools

opt_k_carbonic = 10
opt_total_borate = 1
use_quickload = True

if not use_quickload:
    # Import OceanSODA
    soda = xr.open_dataset(
        "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
        + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
    )

    # Get monthly means of relevant fields
    soda["month"] = soda.time.dt.month
    soda = soda.set_coords("month")
    mvars = ["talk", "dic", "temperature", "salinity"]
    soda_monthly = xr.Dataset({v: soda[v].groupby("month").mean() for v in mvars})

    # Calculate monthly surface fields of dlnfCO2/dT and fCO2
    results = pyco2.sys(
        par1=soda_monthly.talk.data,
        par2=soda_monthly.dic.data,
        par1_type=1,
        par2_type=2,
        temperature=soda_monthly.temperature.data,
        salinity=soda_monthly.salinity.data,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
    )
    soda_monthly["dlnfCO2_dT"] = (("month", "lat", "lon"), results["dlnfCO2_dT"] * 1e3)
    soda_monthly["fCO2"] = (("month", "lat", "lon"), results["fCO2"])

    # Fit bh across the globe
    ex_temperature = np.linspace(-1.8, 35.83, num=50)
    soda_monthly["ex_temperature"] = ("ex_temperature", ex_temperature)
    soda_monthly = soda_monthly.set_coords("ex_temperature")
    ex_fCO2 = np.full((*soda_monthly.dlnfCO2_dT.shape, ex_temperature.size), np.nan)

    # This first loop, to calculate fCO2 across temperature, takes about 5 minutes
    for i, t in enumerate(ex_temperature):
        print(i + 1, "/", len(ex_temperature))
        ex_fCO2[:, :, :, i] = pyco2.sys(
            par1=soda_monthly.talk.data,
            par2=soda_monthly.dic.data,
            par1_type=1,
            par2_type=2,
            temperature=t,
            salinity=soda_monthly.salinity.data,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
        )["fCO2"]
    soda_monthly["ex_fCO2"] = (("month", "lat", "lon", "ex_temperature"), ex_fCO2)

    # This second loop, to fit values for bh, takes about 1.5 minutes
    fit_bh = np.full(soda_monthly.dlnfCO2_dT.shape, np.nan)
    for m in range(soda_monthly.month.size):
        print(m + 1, "/", soda_monthly.month.size)
        for i in range(soda.lat.size):
            for j in range(soda.lon.size):
                if ~np.isnan(ex_fCO2[m, i, j, 0]):
                    fit_bh[m, i, j] = pwtools.fit_vh_curve(
                        ex_temperature, ex_fCO2[m, i, j, :]
                    )[0][0]

    # Put fitted bh values into soda_monthly
    soda_monthly["bh"] = (("month", "lat", "lon"), fit_bh)

    # Save soda_monthly to file for convenience
    soda_monthly.to_zarr("quickload/soda_monthly.zarr")

else:
    soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")


# %% Make the parameterisation
def get_bh(t_s_fCO2, c, t, tt, s, ss, f, ff, ts, tf, sf):
    temperature, salinity, fCO2 = t_s_fCO2
    return (
        c
        + t * temperature
        + tt * temperature**2
        + s * salinity
        + ss * salinity**2
        + f * fCO2
        + ff * fCO2**2
        + ts * temperature * salinity
        + tf * temperature * fCO2
        + sf * salinity * fCO2
    )


# Prepare arguments for fitting
temperature = soda_monthly.temperature.data.ravel().astype(float)
salinity = soda_monthly.salinity.data.ravel().astype(float)
fCO2 = soda_monthly.fCO2.data.ravel()
bh = soda_monthly.bh.data.ravel()
L = ~np.isnan(temperature) & ~np.isnan(salinity) & ~np.isnan(fCO2) & ~np.isnan(bh)
temperature, salinity, fCO2, bh = temperature[L], salinity[L], fCO2[L], bh[L]
t_s_fCO2 = np.array([temperature, salinity, fCO2])

# Fit bh to the monthly means
bh_fit = curve_fit(get_bh, t_s_fCO2, bh, p0=(30000, 0, 0, 0, 0, 0, 0, 0, 0, 0))
bh_coeffs = bh_fit[0]
# bh_coeffs = np.array(
#     [
#         3.13184463e04,
#         1.39487529e02,
#         -1.21087624e00,
#         -4.22484243e00,
#         -6.52212406e-01,
#         -1.69522191e01,
#         -5.47585838e-04,
#         -3.02071783e00,
#         1.66972942e-01,
#         3.09654019e-01,
#     ]
# )
bh_predicted = get_bh(t_s_fCO2, *bh_coeffs)
soda_monthly["bh_predicted"] = get_bh(
    (soda_monthly.temperature, soda_monthly.salinity, soda_monthly.fCO2), *bh_coeffs
)
soda_monthly["bh_diff"] = soda_monthly.bh_predicted - soda_monthly.bh

# Plot the 1:1 fit for the parameterisation
fig, ax = plt.subplots(dpi=300)
ax.scatter(
    bh * 1e-3,
    bh_predicted * 1e-3,
    c="xkcd:dark",
    edgecolor="none",
    alpha=0.05,
    s=10,
)
ax.axline((29, 29), slope=1, c="xkcd:dark", lw=1.2)
axlims = (24.9, 30.8)
ax.set_xlim(axlims)
ax.set_ylim(axlims)
ax.set_aspect(1)
ax.set_xlabel("$b_h$ fitted to OceanSODA-ETZH / kJ mol$^{–1}$")
ax.set_ylabel("$b_h$ from parameterisation / kJ mol$^{–1}$")
fig.tight_layout()
fig.savefig("figures_si/predict_bh_line.png")

# %% Plot where bh does and doesn't work so well
fig, ax = plt.subplots(
    dpi=300, subplot_kw={"projection": ccrs.Robinson(central_longitude=205)}
)
fm = soda_monthly.bh_diff.mean("month").plot(
    ax=ax,
    vmin=-400,
    vmax=400,
    cmap="RdBu_r",
    add_colorbar=False,
    transform=ccrs.PlateCarree(),
)
ax.add_feature(
    cfeature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor=0.1 * np.array([1, 1, 1]),
)
plt.colorbar(
    fm,
    location="bottom",
    label="∆$b_h$ / J mol$^{-1}$",
    pad=0.05,
    aspect=20,
    fraction=0.05,
    extend="both",
)
fig.tight_layout()
fig.savefig("figures_si/predict_bh_map.png")
