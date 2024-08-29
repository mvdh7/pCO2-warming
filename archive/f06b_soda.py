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
from scipy.optimize import least_squares
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
    soda["pCO2_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        results["pCO2"],
    )

    # Fit b_h across the globe
    fit_bh = np.full(
        soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].shape,
        np.nan,
    )
    ex_temperature = np.linspace(-1.8, 35.83)
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
            par1=soda.talk.mean("time").data,
            par2=soda.dic.mean("time").data,
            par1_type=1,
            par2_type=2,
            temperature=t,
            salinity=soda.salinity.mean("time").data,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
        )["pCO2"]
    for i in range(180):
        print(i)
        for j in range(360):
            if ~np.isnan(ex_pCO2[i, j, 0]):
                ex_lnpCO2 = np.log(ex_pCO2[i, j, :])
                fit_bh[i, j] = pwtools.fit_pCO2_vh(ex_temperature, ex_lnpCO2)["x"][0]
    soda["bh_{:02.0f}".format(opt_k_carbonic)] = (("lat", "lon"), fit_bh)

# # %% Quick load
# soda = xr.open_dataset("quickload/f06_soda.zarr", engine="zarr")


# %% Predict bh
def get_bh(coeffs, temperature, salinity, pCO2):
    c, t, tt, s, ss, p, pp, ts, tp, sp = coeffs
    return (
        c
        + t * temperature
        + tt * temperature**2
        + s * salinity
        + ss * salinity**2
        + p * pCO2
        + pp * pCO2**2
        + ts * temperature * salinity
        + tp * temperature * pCO2
        + sp * salinity * pCO2
    )


def _lsqfun_get_bh(coeffs, temperature, salinity, pCO2, bh):
    return get_bh(coeffs, temperature, salinity, pCO2) - bh


temperature = soda.temperature.mean("time").to_numpy().ravel().astype(float)
salinity = soda.salinity.mean("time").to_numpy().ravel().astype(float)
pCO2 = soda.pCO2_10.to_numpy().ravel()
bh = soda.bh_10.to_numpy().ravel()
L = (
    ~np.isnan(temperature)
    & ~np.isnan(pCO2)
    & ~np.isnan(bh)
    & ~np.isnan(salinity)
    # & (salinity > 15)
)
temperature, pCO2, bh, salinity = temperature[L], pCO2[L], bh[L], salinity[L]
lon, lat = np.meshgrid(soda.lon.data, soda.lat.data)
lon = lon.ravel()[L]
lat = lat.ravel()[L]
optr = least_squares(
    _lsqfun_get_bh,
    [30000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    args=(temperature, salinity, pCO2, bh),
    method="lm",
)
bh_pred = get_bh(optr["x"], temperature, salinity, pCO2)

optr_x = np.array(
    [
        3.29346944e04,
        1.64051754e02,
        -1.19736777e00,
        -5.63073675e01,
        -8.85674870e-01,
        -2.24987770e01,
        -3.48315855e-03,
        -3.66862524e00,
        1.57507242e-01,
        5.37574317e-01,
    ]
)

fig, ax = plt.subplots(dpi=300)
ax.scatter(
    bh,
    bh_pred,
    c=lat,
    edgecolor="none",
    alpha=0.2,
    s=10,
)
ax.axline((29100, 29100), slope=1, c="xkcd:strawberry")
ax.set_aspect(1)

# %%
fig, ax = plt.subplots(dpi=300)
fsc = ax.scatter(
    lon, lat, c=bh - bh_pred, s=2, edgecolor="none", cmap="RdBu_r", vmin=-500, vmax=500
)
plt.colorbar(fsc)

# %%
fig, ax = plt.subplots(dpi=300)
ax.scatter(temperature, bh - bh_pred)

# %% Visualise - map
all_polyfits = []

for opt_k_carbonic in [10]:  # range(1, 19):
    # Get axis limits
    pt = soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    pt = pt[~np.isnan(pt)]
    fl = soda["bh_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    fl = fl[~np.isnan(fl)]
    fig, axs = plt.subplots(
        dpi=300,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=205)},
        figsize=[17.4 / 2.54, 7 / 2.54],
        ncols=2,
    )
    for i, fvar in enumerate(
        [
            "dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic),
            "bh_{:02.0f}".format(opt_k_carbonic),
        ]
    ):
        if fvar.startswith("bh"):
            xlims = (np.quantile(fl, 0.005), np.quantile(fl, 0.995))
        else:
            xlims = (np.quantile(pt, 0.005), np.quantile(pt, 0.995))
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
        elif fvar.startswith("bh"):
            letter = "(b)"
            cblabel = "$b_h$"
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
    fig.savefig("figures/f06/f06b_map_both_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()
