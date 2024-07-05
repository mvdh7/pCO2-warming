from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pwtools

# Freshwater, vary TC/AT, see how much less accurate Ax and Tx become
alkalinity = 2000
dic = np.linspace(1700, 2200, num=251)
temperatures = np.vstack(np.linspace(-1, 35, num=37))
results = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    temperature=temperatures,
    opt_k_carbonic=8,
    salinity=0,
)

alkalinity_x = results["HCO3"] + 2 * results["CO3"]
dic_x = results["HCO3"] + results["CO3"]

fit_ch = np.full(dic.shape, np.nan)
fCO2_ch = np.full(results["fCO2"].shape, np.nan)
fit_bhch = np.full((dic.size, 2), np.nan)
fCO2_bhch = np.full(results["fCO2"].shape, np.nan)
for i in range(len(dic)):
    fit_ch[i] = pwtools.fit_fCO2_vht(
        temperatures.ravel(), np.log(results["fCO2"][:, i])
    )["x"][0]
    fCO2_ch[:, i] = np.exp(pwtools.get_lnfCO2_vht(fit_ch[i], temperatures.ravel()))
    fit_bhch[i] = pwtools.fit_fCO2_vh(
        temperatures.ravel(), np.log(results["fCO2"][:, i])
    )["x"]
    fCO2_bhch[:, i] = np.exp(pwtools.get_lnfCO2_vh(fit_bhch[i], temperatures.ravel()))
fCO2_ch_rmsd = np.sqrt(np.mean((fCO2_ch - results["fCO2"]) ** 2, axis=0))
fCO2_bhch_rmsd = np.sqrt(np.mean((fCO2_bhch - results["fCO2"]) ** 2, axis=0))
fCO2_range = np.max(results["fCO2"], axis=0) - np.min(results["fCO2"], axis=0)
fCO2_ch_rmsd_norm = fCO2_ch_rmsd * np.median(fCO2_range) / fCO2_range
fCO2_bhch_rmsd_norm = fCO2_bhch_rmsd * np.median(fCO2_range) / fCO2_range

fig, axs = plt.subplots(dpi=300, nrows=3, figsize=(4, 8))

ax = axs[0]
for i, temperature in enumerate(temperatures):
    ax.plot(
        dic / alkalinity,
        results["fCO2"][i],
        # fCO2_bhch[i],
        c=mpl.cm.turbo((temperature + 1) / 37),
    )
ax.set_ylabel("{f}CO$_2$ / µatm".format(f=pwtools.f))
ax.text(0, 1.05, "(a)", transform=ax.transAxes)

ax = axs[1]
for i, temperature in enumerate(temperatures):
    ax.plot(
        dic / alkalinity,
        alkalinity_x[i] - alkalinity,
        label=r"$A_\mathrm{T}$",
        c=mpl.cm.turbo((temperature + 1) / 37),
    )
    ax.plot(
        dic / alkalinity,
        dic_x[i] - dic,
        label=r"$T_\mathrm{C}$",
        c=mpl.cm.turbo((temperature + 1) / 37),
        zorder=10,
    )
ax.set_ylabel("$V_x$ – $V$ / µmol kg$^{-1}$")
ax.text(1.09, -19, r"$A_\mathrm{T}$", ha="right")
ax.text(1.09, -200, r"$T_\mathrm{C}$", ha="right")
ax.text(0, 1.05, "(b)", transform=ax.transAxes)

ax = axs[2]
ax.plot(dic / alkalinity, fCO2_ch_rmsd, label="Fixed $b_h$", c="xkcd:dark", ls="--")
ax.plot(dic / alkalinity, fCO2_ch_rmsd_norm, label="Fixed $b_h$, norm.", c="xkcd:dark")
ax.plot(
    dic / alkalinity, fCO2_bhch_rmsd, label="Fitted $b_h$", c="xkcd:cerulean", ls="--"
)
ax.plot(
    dic / alkalinity,
    fCO2_bhch_rmsd_norm,
    label="Fitted $b_h$, norm.",
    c="xkcd:cerulean",
)
ax.set_ylabel("(N)RMSD / µatm")
ax.set_yscale("log")
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.3), edgecolor="k", ncol=2, fontsize=9
)
ax.text(0, 1.05, "(c)", transform=ax.transAxes)

for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_xlabel(r"$T_\mathrm{C} / A_\mathrm{T}$")
    ax.set_xlim(np.min(dic / alkalinity), np.max(dic / alkalinity))

fig.tight_layout()

# %%
i = 0
fig, axs = plt.subplots(dpi=300)
ax = axs
# ax.plot(temperatures.ravel(), results['fCO2'][:, i], label="PyCO2SYS", c='xkcd:dark')
ax.plot(
    temperatures.ravel(),
    fCO2_ch[:, i] - results["fCO2"][:, i],
    label="Fixed $b_h$",
    c="xkcd:cerulean",
    ls=":",
)
ax.plot(
    temperatures.ravel(),
    fCO2_bhch[:, i] - results["fCO2"][:, i],
    label="Fitted $b_h$",
    c="xkcd:cerulean",
    ls="--",
)
ax.legend()
ax.axhline(0, c="k", lw=0.8)
ax.set_xlim(-1, 35)
ax.set_title((dic[i] / alkalinity))
