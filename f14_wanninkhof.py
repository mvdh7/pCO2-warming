import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy import stats
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

# Get W22 cutoff
L = wkf.temperature_diff.notnull() & wkf.fr_fCO2.notnull()
lr = stats.linregress(wkf.temperature_diff[L], np.log(wkf.fr_fCO2[L]))
wkf["ln_fr_fCO2_linear_fit_diff"] = np.log(wkf.fr_fCO2) - (
    lr.slope * wkf.temperature_diff + lr.intercept
)
wkf_cutoff = wkf.ln_fr_fCO2_linear_fit_diff.std() * 2

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


# %%
def std_Sn(a):
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/diffsn.htm
    #
    # Peter J. Rousseuw and Christophe Croux (1993), "Alternatives to the Median
    # Absolute Deviation," Journal of the American Statistical Association, Vol. 88,
    # No. 424, pp. 1273-1283.
    a = a[~np.isnan(a)].ravel()
    sn0 = np.full((np.size(a), np.size(a)), np.nan)
    for i in range(len(a)):
        sn0[i] = np.abs(a[i] - a)
    np.fill_diagonal(sn0, np.nan)
    sn1 = np.nanmedian(sn0, axis=1)
    sn = 1.1926 * np.median(sn1)
    return sn


# %% Visualise another way
fwindow = 4  # °C
fx = np.linspace(
    np.min(wkf.temperature_diff) + 0.1, np.max(wkf.temperature_diff) - 0.1, num=50
)
fy_h = np.full(fx.size, np.nan)
fy_hf = np.full(fx.size, np.nan)
fy_l = np.full(fx.size, np.nan)
fu_h = np.full(fx.size, np.nan)
fu_hf = np.full(fx.size, np.nan)
fu_l = np.full(fx.size, np.nan)
ttp_h = np.full(fx.size, np.nan)
ttp_hf = np.full(fx.size, np.nan)
ttp_l = np.full(fx.size, np.nan)
fy_sum = np.full(fx.size, 0)
for i in range(len(fx)):
    L = (
        (wkf.temperature_diff > fx[i] - fwindow)
        & (wkf.temperature_diff < fx[i] + fwindow)
        & wkf.fCO2_h_diff.notnull()
        & wkf.fCO2_bh_fitted_diff.notnull()
        & wkf.fCO2_l_diff.notnull()
        & (wkf.fCO2_bh_fitted_diff.abs() < 25)
        # & (wkf.ln_fr_fCO2_linear_fit_diff <= wkf_cutoff)
    )
    ttp_h[i] = stats.ttest_1samp(wkf.fCO2_h_diff[L], popmean=0).pvalue
    ttp_hf[i] = stats.ttest_1samp(wkf.fCO2_bh_fitted_diff[L], popmean=0).pvalue
    ttp_l[i] = stats.ttest_1samp(wkf.fCO2_l_diff[L], popmean=0).pvalue
    fy_h[i] = wkf.fCO2_h_diff[L].mean()
    fy_hf[i] = wkf.fCO2_bh_fitted_diff[L].mean()
    fy_l[i] = wkf.fCO2_l_diff[L].mean()
    fy_sum[i] = L.sum()
    fu_h[i] = wkf.fCO2_h_diff.std() / np.sqrt(fy_sum[i])
    fu_hf[i] = wkf.fCO2_bh_fitted_diff.std() / np.sqrt(fy_sum[i])
    fu_l[i] = wkf.fCO2_l_diff.std() / np.sqrt(fy_sum[i])

fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 8 / 2.54))
ax.scatter(
    "temperature_diff",
    # "fCO2_h_diff",
    "fCO2_bh_fitted_diff",
    data=wkf,
    s=10,
    edgecolor="none",
    c="xkcd:navy",
    # c=wkf.ta_dic,
    alpha=0.2,
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

confidence = 99  # percent confidence that we can reject the null hypothesis.
# The null hypothesis is that the mean of the data is zero.
confidence_fr = 1 - confidence / 100

fy_hf_zero = fy_hf.copy()
fy_hf_zero[ttp_hf <= confidence_fr] = np.nan
fy_hf_nonzero = fy_hf.copy()
fy_hf_nonzero[ttp_hf > confidence_fr] = np.nan
ax.plot(fx, fy_hf, c="xkcd:navy", lw=1.5, alpha=0.8)
# ax.plot(fx, fy_hf_zero, c="xkcd:navy", lw=2)
# ax.plot(fx, fy_hf_nonzero, c="xkcd:navy", lw=2, dashes=(2, 1.2))
ax.fill_between(
    fx, fy_hf - fu_hf, fy_hf + fu_hf, color="xkcd:navy", edgecolor="none", alpha=0.3
)

fy_h_zero = fy_h.copy()
fy_h_zero[ttp_h <= confidence_fr] = np.nan
fy_h_nonzero = fy_h.copy()
fy_h_nonzero[ttp_h > confidence_fr] = np.nan
ax.plot(fx, fy_h, c="xkcd:azure", lw=1.5, alpha=0.8)
# ax.plot(fx, fy_h_zero, c="xkcd:azure", lw=2)
# ax.plot(fx, fy_h_nonzero, c="xkcd:azure", lw=2, dashes=(2, 1.2))
ax.fill_between(
    fx, fy_h - fu_h, fy_h + fu_h, color="xkcd:azure", edgecolor="none", alpha=0.2
)

fy_l_zero = fy_l.copy()
fy_l_zero[ttp_l <= confidence_fr] = np.nan
fy_l_nonzero = fy_l.copy()
fy_l_nonzero[ttp_l > confidence_fr] = np.nan
ax.plot(fx, fy_l, c="xkcd:strawberry", lw=1.5, alpha=0.8)
# ax.plot(fx, fy_l_zero, c="xkcd:strawberry", lw=2)
# ax.plot(fx, fy_l_nonzero, c="xkcd:strawberry", lw=2, dashes=(2, 1.2))
ax.fill_between(
    fx, fy_l - fu_l, fy_l + fu_l, color="xkcd:strawberry", edgecolor="none", alpha=0.2
)

ax.axhline(0, c="k", lw=0.8, zorder=-1)
ax.set_ylim([-12, 12])
ax.set_xticks(np.arange(-20, 15, 5))
ax.set_xlim([-21, 12])
ax.set_xlabel("∆$t$ / °C")
ax.tick_params(top=True, labeltop=False)
for t in np.arange(0, 35, 5):
    ax.text(t - 20, 13, "{}".format(t), ha="center", va="bottom")
ax.text(0.5, 1.11, "$t_1$ / °C", ha="center", va="bottom", transform=ax.transAxes)
ax.grid(alpha=0.2)
ax.set_ylabel("[" + pwtools.thinspace + "$ƒ$CO$_2$($t_1$) – $ƒ$CO$_2$(20 °C)] / µatm")
fig.tight_layout()
fig.savefig("figures/f14_wanninkhof.png")
