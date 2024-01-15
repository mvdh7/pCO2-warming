import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import pwtools

# Import Wanninkhof et al. (2022) dataset
wkf = pd.read_excel("data/AOML_All-Data_combined_UW_fCO2_disc_fCO2wCO2Sys.xlsx").rename(
    columns={
        "fCO2(20)disc_UATM": "fCO2_d20",
        "UWfCO2w_situ_UATM": "fCO2_uw",
        "UW_TSG_Temp_DEG_C": "temperature",
        "TSG_Salinity": "salinity",
        "TA_UMOL/KG": "alkalinity",
        "DIC_UMOL/KG": "dic",
        "TA/DIC": "ta_dic",
    }
)

# Fix bugs
wkf.loc[wkf.fCO2_uw.apply(type) == str, "fCO2_uw"] = np.nan
wkf["fCO2_uw"] = wkf.fCO2_uw.astype(float)

# Calculate fraction
wkf["fr_fCO2"] = wkf.fCO2_uw / wkf.fCO2_d20
wkf["temperature_diff"] = wkf.temperature - 20

# Get bins
fstep = 2
fx = np.arange(-25.0, 15, fstep)
wkf["npts_bin"] = 0
for i in range(len(fx) - 1):
    L = (
        (wkf.temperature_diff >= fx[i])
        & (wkf.temperature_diff < fx[i + 1])
        & (wkf.fCO2_d20.notnull() & wkf.fCO2_uw.notnull())
    )
    wkf.loc[L, "npts_bin"] = L.sum()

# Find best fits to this dataset
t0 = 20
L = wkf.temperature.notnull() & wkf.fr_fCO2.notnull()
t1 = wkf.temperature[L].to_numpy()
expUPS = wkf.fr_fCO2[L].to_numpy()
fCO2_d20 = wkf.fCO2_d20[L].to_numpy()
fCO2_uw = wkf.fCO2_uw[L].to_numpy()
npts_bin = wkf.npts_bin[L].to_numpy()


def fit_l(bl, t0, t1):
    return np.exp(pwtools.get_H_l(bl, t0, t1))


def _lsqfun_fit_l(bl, t0, t1):
    # return fit_l(bl, t0, t1) - expUPS
    return (fCO2_d20 * fit_l(bl, t0, t1) - fCO2_uw) / npts_bin


optr_l = least_squares(_lsqfun_fit_l, 0.0423, args=(t0, t1))


def fit_h(bh, t0, t1):
    return np.exp(pwtools.get_H_h(bh, t0, t1))


def _lsqfun_fit_h(bh, t0, t1):
    # return fit_h(bh, t0, t1) - expUPS
    return (fCO2_d20 * fit_h(bh, t0, t1) - fCO2_uw) / npts_bin


optr_h = least_squares(_lsqfun_fit_h, pwtools.bh_best, args=(t0, t1))

# Set which bh and bl to use for figures
bh = pwtools.bh_best
bl = 0.0423
# bh = optr_h["x"]
# bl = optr_l["x"]

# Calculate various differences
wkf["UPS_h"] = pwtools.get_H_h(bh, 20, wkf.temperature)
wkf["fCO2_h"] = wkf.fCO2_d20 * np.exp(wkf.UPS_h)
wkf["fCO2_h_diff"] = wkf.fCO2_h - wkf.fCO2_uw
wkf["UPS_l"] = pwtools.get_H_l(bl, 20, wkf.temperature)
wkf["fCO2_l"] = wkf.fCO2_d20 * np.exp(wkf.UPS_l)
wkf["fCO2_l_diff"] = wkf.fCO2_l - wkf.fCO2_uw

# Visualise
fx = np.linspace(wkf.temperature_diff.min(), wkf.temperature_diff.max())
fy_h = np.exp(pwtools.get_H_h(bh, 20, 20 + fx))
fy_l = np.exp(pwtools.get_H_l(bl, 20, 20 + fx))
fig, ax = plt.subplots(dpi=300)
ax.scatter("temperature_diff", "fr_fCO2", data=wkf)
ax.plot(fx, fy_h, c="r")
ax.plot(fx, fy_l, c="k")
ax.set_xlabel("∆$t$ / °C")
ax.set_ylabel(r"exp($\Upsilon$) SHOULD BE ITALICISED")

# %% visualise with fitted bh from f06b_soda.py


def get_bh(temperature, salinity, pCO2):
    c, t, tt, s, ss, p, pp, ts, tp, sp = np.array(
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


wkf["bh_fitted_uw"] = get_bh(wkf.temperature, wkf.salinity, wkf.fCO2_uw)
wkf["fCO2_bh_fitted"] = wkf.fCO2_d20 * np.exp(
    pwtools.get_H_h(wkf.bh_fitted_uw, 20, wkf.temperature)
)
wkf["fCO2_bh_fitted_diff"] = wkf.fCO2_bh_fitted - wkf.fCO2_uw

fig, ax = plt.subplots(dpi=300)
ax.scatter("temperature_diff", "fCO2_bh_fitted_diff", data=wkf, s=20, edgecolor="none")
ax.set_ylim((-50, 50))
ax.axhline(0, c="k", lw=0.8)

# %% Visualise another way
fstep = 1.5
fx = np.arange(-25.0, 15, fstep)
fy_h = np.full(fx.size - 1, np.nan)
fy_hf = np.full(fx.size - 1, np.nan)
fy_l = np.full(fx.size - 1, np.nan)
fy_sum = np.full(fx.size - 1, 0)
for i in range(len(fx) - 1):
    L = (
        (wkf.temperature_diff >= fx[i])
        & (wkf.temperature_diff < fx[i + 1])
        & (wkf.fCO2_h_diff.notnull())
    )
    if L.sum() > 3:
        fy_h[i] = wkf.fCO2_h_diff[L].median()
        fy_hf[i] = wkf.fCO2_bh_fitted_diff[L].median()
        fy_l[i] = wkf.fCO2_l_diff[L].median()
    fy_sum[i] = L.sum()
fxp = fx[:-1] + fstep / 2

fig, ax = plt.subplots(dpi=300)
ax.scatter(
    "temperature_diff",
    # "fCO2_h_diff",
    "fCO2_bh_fitted_diff",
    data=wkf,
    s=20,
    edgecolor="none",
    # c="xkcd:navy",
    c=wkf.ta_dic,
    alpha=0.5,
)
# ax.scatter(
#     "temperature_diff",
#     "fCO2_l_diff",
#     data=wkf,
#     s=20,
#     edgecolor="none",
#     c="xkcd:strawberry",
#     alpha=0.2,
# )
ax.plot(fxp, fy_h, c="xkcd:navy")
ax.plot(fxp, fy_l, c="xkcd:strawberry")
ax.plot(fxp, fy_hf, c="xkcd:brown")
ax.axhline(0, c="k", lw=0.8)
ax.set_ylim([-25, 25])
