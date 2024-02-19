from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import pandas as pd
import numpy as np
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
        "fCO2(20)(TA,DIC)_UATM": "fCO2_12_20",
        "fCO2(SST)(TA,DIC)_UATM": "fCO2_12_insitu",
    }
)

# Fix bugs
wkf.loc[wkf.fCO2_uw.apply(type) == str, "fCO2_uw"] = np.nan
wkf["fCO2_uw"] = wkf.fCO2_uw.astype(float)

# Calculate differences
t0 = 20  # °C --- all fCO2_d20 samples are at 20 °C (this is t0; t1 is the underway)
wkf["temperature_diff"] = wkf.temperature - t0
# wkf["fCO2_12_insitu_diff"] = wkf.fCO2_12_insitu - wkf.fCO2_uw

# Correct with PyCO2SYS
results_15 = pyco2.sys(
    par1=wkf.fCO2_d20.to_numpy(),
    par1_type=5,
    par2=wkf.alkalinity.to_numpy(),
    par2_type=1,
    salinity=wkf.salinity.to_numpy(),
    temperature=t0,
    temperature_out=wkf.temperature.to_numpy(),
    opt_k_carbonic=10,
    opt_total_borate=1,
)
wkf["fCO2_calc15"] = results_15["fCO2_out"]
results_25 = pyco2.sys(
    par1=wkf.fCO2_d20.to_numpy(),
    par1_type=5,
    par2=wkf.dic.to_numpy(),
    par2_type=2,
    salinity=wkf.salinity.to_numpy(),
    temperature=t0,
    temperature_out=wkf.temperature.to_numpy(),
    opt_k_carbonic=10,
    opt_total_borate=1,
)
wkf["fCO2_calc25"] = results_25["fCO2_out"]

# %% Calculate the fCO2 adjustment with different methods
opt_at = [1, 2, 5, 6]
results = pyco2.sys(
    par1=wkf.fCO2_d20.to_numpy(),
    par1_type=5,
    salinity=wkf.salinity.to_numpy(),
    temperature=t0,
    temperature_out=wkf.temperature.to_numpy(),
    opt_adjust_temperature=np.vstack(opt_at),
    opt_which_fCO2_insitu=2,
)
wkf["fCO2_uw_1"], wkf["fCO2_uw_2"], wkf["fCO2_uw_5"], wkf["fCO2_uw_6"] = results[
    "fCO2_out"
]

# Visualise
fwindow = 4  # °C
fx = np.linspace(
    np.min(wkf.temperature_diff) + 0.1, np.max(wkf.temperature_diff) - 0.1, num=100
)
fvars = [
    "fCO2_uw_1",
    "fCO2_uw_2",
    "fCO2_uw_5",
    "fCO2_uw_6",
    "fCO2_calc15",
    "fCO2_calc25",
]
for v in fvars:
    wkf[v + "_diff"] = wkf[v] - wkf.fCO2_uw
fy = {v: np.full(fx.size, np.nan) for v in fvars}
fu = {v: np.full(fx.size, np.nan) for v in fvars}
fs = np.full(fx.size, np.nan)
for i in range(len(fx)):
    L = (
        (wkf.temperature_diff > fx[i] - fwindow)
        & (wkf.temperature_diff < fx[i] + fwindow)
        & wkf.fCO2_uw_1_diff.notnull()
        & wkf.fCO2_uw_2_diff.notnull()
        & wkf.fCO2_uw_5_diff.notnull()
        & (wkf.fCO2_uw_1_diff.abs() < 25)
    )
    fs[i] = L.sum()
    for v in fvars:
        fy[v][i] = wkf[v + "_diff"][L].mean()
        fu[v][i] = wkf[v + "_diff"].std() / np.sqrt(fs[i])

# Draw the figure
fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 15 / 2.54))
# ax.scatter(
#     "temperature_diff",
#     "fCO2_uw_1_diff",
#     data=wkf,
#     s=10,
#     edgecolor="none",
#     c="xkcd:navy",
#     alpha=0.2,
#     label=None,
# )
ax.plot(
    fx,
    fy["fCO2_uw_5"],
    c=pwtools.dark,
    lw=1.5,
    # alpha=0.8,
    label="$υ_l$ (Ta93, linear)",
    zorder=-1,
)
ax.plot(
    fx,
    fy["fCO2_uw_6"],
    c=pwtools.dark,
    lw=1.5,
    ls=(0, (6, 2)),
    label="$υ_q$ (Ta93, quadratic)",
    zorder=2,
)
ax.plot(
    fx,
    fy["fCO2_uw_2"],
    c=pwtools.blue,
    lw=1.5,
    ls=(0, (3, 1)),
    # alpha=0.8,
    label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
    zorder=1,
)
ax.plot(
    fx,
    fy["fCO2_uw_1"],
    c=pwtools.blue,
    lw=1.5,
    # alpha=0.8,
    label="$υ_p$ (van 't Hoff, $b_h$ parameterised)",
    zorder=0,
)
# ax.plot(
#     fx,
#     fy["fCO2_calc15"],
#     c="xkcd:green",
#     lw=1.5,
#     alpha=0.8,
#     label=r"$υ_\mathrm{Lu00}$ (PyCO2SYS, with $A_\mathrm{T}$)",
# )
ax.plot(
    fx,
    fy["fCO2_calc25"],
    c=pwtools.pink,
    lw=1.5,
    ls=(2, (2,)),
    # alpha=0.8,
    label=r"$υ_\mathrm{Lu00}$ (PyCO2SYS, with $T_\mathrm{C}$)",
    zorder=3,
)
ax.fill_between(
    fx,
    -2 * fu["fCO2_uw_1"],
    2 * fu["fCO2_uw_1"],
    color=pwtools.dark,
    edgecolor="none",
    alpha=0.2,
    label="2$σ$ uncertainty in Wa22 compilation mean",
    zorder=-2,
)
ax.axhline(0, c="k", lw=0.8, zorder=-1)
ax.set_yticks(np.arange(-12, 12, 2))
ax.set_ylim([-8, 8])
ax.set_xticks(np.arange(-20, 15, 5))
ax.set_xlim([-20, 11.3])
ax.set_xlabel("∆$t$ / °C")
ax.tick_params(top=True, labeltop=False)
for t in np.arange(0, 35, 5):
    ax.text(t - 20, 8.5, "{}".format(t), ha="center", va="bottom")
ax.text(0.5, 1.1, "$t_1$ / °C", ha="center", va="bottom", transform=ax.transAxes)
ax.grid(alpha=0.2)
ax.set_ylabel(
    "[{sp}{f}CO$_2$($t_1$) – {f}CO$_2$(20 °C)] / µatm".format(
        sp=pwtools.thinspace, f=pwtools.f
    )
)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), edgecolor="k", ncol=1, fontsize=9
)
fig.tight_layout()
# fig.savefig("figures_final/figure2.png")
