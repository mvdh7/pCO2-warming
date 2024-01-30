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

    # This second loop, to fit values for bh (and ch), takes about 1.5 minutes
    fit_bh = np.full(soda_monthly.dlnfCO2_dT.shape, np.nan)
    fit_ch = np.full(soda_monthly.dlnfCO2_dT.shape, np.nan)
    for m in range(soda_monthly.month.size):
        print(m + 1, "/", soda_monthly.month.size)
        for i in range(soda_monthly.lat.size):
            for j in range(soda_monthly.lon.size):
                if ~np.isnan(ex_fCO2[m, i, j, 0]):
                    fit_bh[m, i, j], fit_ch[m, i, j] = pwtools.fit_vh_curve(
                        ex_temperature, ex_fCO2[m, i, j, :]
                    )[0]

    # Put fitted bh and ch values into soda_monthly
    soda_monthly["bh"] = (("month", "lat", "lon"), fit_bh)
    soda_monthly["ch"] = (("month", "lat", "lon"), fit_ch)

    # Save soda_monthly to file for convenience
    soda_monthly.to_zarr("quickload/soda_monthly.zarr")

else:
    soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")

# %%
ex_temperature = np.reshape(ex_temperature, (1, 1, 1, 50))
ex_fCO2 = soda_monthly.ex_fCO2.data
fit_bh = np.reshape(soda_monthly.bh.data, (12, 180, 360, 1))
fit_ch = np.reshape(soda_monthly.ch.data, (12, 180, 360, 1))

fCO2_from_bhch = np.exp(
    fit_ch - fit_bh / (pwtools.Rgas * (ex_temperature + pwtools.tzero))
)
soda_monthly["fCO2_from_bhch"] = (
    ("month", "lat", "lon", "ex_temperature"),
    fCO2_from_bhch,
)

fit_bh_rmsd = np.sqrt(np.mean((fCO2_from_bhch - ex_fCO2) ** 2, axis=3))
soda_monthly["fit_bh_rmsd"] = (("month", "lat", "lon"), fit_bh_rmsd)
soda_monthly["dic_ta"] = soda_monthly.dic / soda_monthly.talk

# %% Calculate non-carbonate alkalinity
results = pyco2.sys(
    par1=soda_monthly.talk.data,
    par1_type=1,
    par2=soda_monthly.dic.data,
    par2_type=2,
    salinity=soda_monthly.salinity.data,
    temperature=soda_monthly.temperature.data,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
# %%
soda_monthly["alk_carb_bicarb"] = (
    ("month", "lat", "lon"),
    results["HCO3"] + 2 * results["CO3"],
)
soda_monthly["alk_non_carb_bicarb"] = soda_monthly.talk - soda_monthly.alk_carb_bicarb
for v in [
    "pH",
    "CO2",
    "alkalinity_borate",
    "isocapnic_quotient",
    "beta_dic",
    "gamma_dic",
    "beta_alk",
    "gamma_alk",
    "revelle_factor",
]:
    soda_monthly[v] = (("month", "lat", "lon"), results[v])
soda_monthly["CO2_dic_pct"] = 100 * soda_monthly.CO2 / soda_monthly.dic
soda_monthly["alk_non_carb_pct"] = (
    100 * soda_monthly.alk_non_carb_bicarb / soda_monthly.talk
)


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

# %%
baltic = soda_monthly.sel(lon=slice(13, 24), lat=slice(54, 65))

# Find max RMSD in Baltic
baltic_rmsd = baltic.fit_bh_rmsd.data.ravel()
baltic_rmsd[np.isnan(baltic_rmsd)] = 0
baltic_rmsd_argmax = np.argmax(baltic_rmsd)
m, i, j = np.unravel_index(baltic_rmsd_argmax, baltic.fit_bh_rmsd.data.shape)

fig, ax = plt.subplots(dpi=300)
ax.plot(
    ex_temperature.ravel(),
    baltic.ex_fCO2.data[m, i, j, :] - baltic.fCO2_from_bhch.data[m, i, j, :],
)
ax.axhline(c="k", lw=0.8)

# %%
baltic_shape = list(baltic.ex_fCO2.data.shape)
baltic_shape[-1] = 1
fig, ax = plt.subplots(dpi=300)
fs = ax.scatter(
    np.tile(ex_temperature, baltic_shape),
    baltic.ex_fCO2.data - baltic.fCO2_from_bhch.data,
    s=5,
    c=np.tile(baltic.dic_ta, (1, 1, 1, ex_temperature.size)),
)
plt.colorbar(fs, label=r"$T_\mathrm{C}$ / $A_\mathrm{T}$")
ax.axhline(c="k", lw=0.8)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("{f}CO$_2$($υ_h$) – {f}CO$_2$(Lu00) / µatm".format(f=pwtools.f))
ax.set_title("Baltic Sea")

# %%
sm_shape = list(soda_monthly.ex_fCO2.data.shape)
sm_shape[-1] = 1
lo_temperature = np.tile(ex_temperature, sm_shape)
lo_diff = soda_monthly.ex_fCO2.data - soda_monthly.fCO2_from_bhch.data
lo_dic_ta = soda_monthly.dic_ta.data
L = lo_dic_ta < 0.9
lo_temperature = lo_temperature[L]
lo_diff = lo_diff[L]
lo_dic_ta = lo_dic_ta[L]
fig, ax = plt.subplots(dpi=300)
ax.scatter(lo_temperature[::10], lo_diff[::10], s=5, alpha=0.5)
ax.axhline(c="k", lw=0.8)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("{f}CO$_2$($b_h$) – {f}CO$_2$(Lu00) / µatm".format(f=pwtools.f))
ax.set_title(r"$T_\mathrm{C}$ / $A_\mathrm{T}$ < 0.9")

# %%
fig, ax = plt.subplots(dpi=300)
ax.scatter(
    "dic_ta",
    "fit_bh_rmsd",
    data=soda_monthly,
    s=5,
    alpha=0.2,
    c="salinity",
    edgecolor="none",
)
ax.scatter(
    "dic_ta",
    "fit_bh_rmsd",
    data=baltic,
    s=5,
    c="xkcd:strawberry",
    alpha=0.5,
    edgecolor="none",
)
ax.set_xlabel(r"$T_\mathrm{C}$ / $A_\mathrm{T}$")
ax.set_ylabel("RMSD of $b_h$ fit / µatm")
ax.set_ylim((0, 5.2))

# %%
fig, ax = plt.subplots(dpi=300)
ax.scatter(
    "alk_non_carb_pct",
    "fit_bh_rmsd",
    data=soda_monthly,
    s=5,
    alpha=0.2,
    c="salinity",
    edgecolor="none",
)
# ax.scatter(
#     "alk_non_carb_bicarb",
#     "fit_bh_rmsd",
#     data=baltic,
#     s=5,
#     c="xkcd:strawberry",
#     alpha=0.5,
#     edgecolor="none",
# )
ax.set_xlabel(r"[($A_\mathrm{T}$ $-$ $A_x$) / $A_\mathrm{T}$] / %")
ax.set_ylabel("RMSD of $b_h$ fit / µatm")
ax.set_ylim((0, 5.2))

# %%
fig, ax = plt.subplots(dpi=300)
ax.scatter(
    "pH",
    "fit_bh_rmsd",
    data=soda_monthly,
    s=5,
    alpha=0.2,
    c="alk_non_carb_bicarb",
    edgecolor="none",
)
ax.set_xlabel(r"pH$_\mathrm{T}$")
ax.set_ylabel("RMSD of $b_h$ fit / µatm")
ax.set_ylim((0, 5.2))

# %%
fig, ax = plt.subplots(dpi=300)
fs = ax.scatter(
    "CO2_dic_pct",
    "fit_bh_rmsd",
    data=soda_monthly,
    s=5,
    alpha=0.2,
    c="salinity",
    edgecolor="none",
)
plt.colorbar(fs)
ax.set_xlabel(r"{[CO$_2$(aq)] / $T_\mathrm{C}$} / %")
ax.set_ylabel("RMSD of $b_h$ fit / µatm")
ax.set_ylim((0, 5.2))

# %%
for m in range(1, 13):
    fig, ax = plt.subplots(dpi=300)
    baltic.fit_bh_rmsd.sel(month=m).plot(ax=ax, vmin=0, vmax=5)
