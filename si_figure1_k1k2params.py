from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from scipy.stats import linregress
import matplotlib
from matplotlib import pyplot as plt
import takahashi93 as t93
import pwtools

# %% Do own "linear regression" (forced to T93 slope) and quadratic fit
fCO2 = t93.get_fCO2(10, 1)
lr_pCO2 = linregress(
    t93.temperature, np.log(t93.pCO2)
)  # slope comes out as 42.2 not 42.3!
lr_fCO2 = linregress(t93.temperature, np.log(fCO2))  # slope comes out as 42.2 not 42.3!
lr_slope = 42.3e-3
lr_intercept = np.mean(np.log(fCO2) - t93.temperature * lr_slope)

# Get alkalinity
opt_total_borate = 1
alkalinity = {}
for okc in range(1, 19):
    alkalinity[okc] = t93.get_alkalinity(okc, opt_total_borate)[0]

# Calculate fCO2 variation with temperature with different approaches
v_temperature = np.linspace(-1.8, 35.83, num=100)
f_linear = np.exp(v_temperature * lr_slope + lr_intercept)

# - with PyCO2SYS (autograd approach)
v_results = {}
v_fCO2 = {}
v_lnfCO2_PyCO2SYS = {}
for okc in range(1, 19):
    v_results[okc] = pyco2.sys(
        par1=alkalinity[okc],
        par1_type=1,
        par2=t93.dic,
        par2_type=2,
        temperature=v_temperature,
        opt_k_carbonic=okc,
        opt_total_borate=opt_total_borate,
        **t93.tak93,
    )
    v_fCO2[okc] = v_results[okc]["fCO2"]
    v_lnfCO2_PyCO2SYS[okc] = np.log(v_fCO2[okc])

# %% Visualise similar to Takahashi et al. (1993) Figure A1
cmap = matplotlib.colormaps["tab20"]
cvalues = np.linspace(0, 1, num=20)
cvalues = [cmap(cv) for cv in cvalues]
cvalues = [
    "purple",
    "green",
    "blue",
    "pink",
    "brown",
    "red",
    "light blue",
    "teal",
    "orange",
    "turquoise",
    "magenta",
    "tan",
    "sky blue",
    "grey",
    "mauve",
    "light purple",
    "violet",
    "dark green",
]
cvalues = ["xkcd:" + cv for cv in cvalues]

groups = {}
groups[0] = [6, 7]
groups[1] = [4, 10, 11, 13, 14, 15, 17, 9]
groups[2] = [1, 2, 3, 5]
groups[3] = [
    i
    for i in range(1, 19)
    if i not in groups[0] and i not in groups[1] and i not in groups[2]
]
style_vh = dict(
    c="xkcd:dark",
    # label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (3, 1)),
    zorder=10,
)
letters = ["a", "b", "c", "d"]

fig, axs = plt.subplots(dpi=300, nrows=2, ncols=2, figsize=(18.4 / 2.54, 16 / 2.54))

for i, ax in enumerate(axs.ravel()):
    ax.text(0, 1.05, "(" + letters[i] + ")", transform=ax.transAxes)
    ax.errorbar(
        t93.temperature,
        fCO2 - np.exp(t93.temperature * lr_slope + lr_intercept),
        np.sqrt(8),
        c="xkcd:dark",
        ls="none",
        zorder=10,
    )
    ax.scatter(
        t93.temperature,
        fCO2 - np.exp(t93.temperature * lr_slope + lr_intercept),
        c="xkcd:dark",
        # label="Ta93 measured",
        s=50,
        zorder=10,
        # alpha=0.9,
        edgecolor="none",
    )
    for okc in groups[i]:
        ax.plot(
            v_temperature,
            np.exp(v_lnfCO2_PyCO2SYS[okc]) - f_linear,
            lw=2,
            c=cvalues[okc - 1],
            label=pwtools.okc_codes[okc],
        )
    ax.plot(
        v_temperature,
        np.exp(pwtools.get_lnfCO2_vh([pwtools.bh_best, pwtools.ch_best], v_temperature))
        - f_linear,
        **style_vh,
    )
    ax.set_xlabel("Temperature / °C")
    ax.set_ylabel(
        "[{sp}{f}CO$_2$ $-$ {f}CO$_2$($υ_l$)] / µatm".format(
            sp=pwtools.thinspace, f=pwtools.f
        )
    )
    ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))
    ax.set_ylim((-43, 23))
    ncol = 1
    if i in [1, 2]:
        ncol = 2
    ax.legend(loc="lower left", ncol=ncol, fontsize=9)
    ax.grid(alpha=0.2)

# ax = axs[1]
# ax.text(0, 1.05, "(b)", transform=ax.transAxes)
# ax.axhline(1e3 * pyco2.upsilon.ups_linear_TOG93(), **style_linear)
# ax.plot(
#     v_temperature,
#     1e3 * pyco2.upsilon.ups_enthalpy_H24(v_temperature, pwtools.Rgas * 10),
#     **style_vht,
# )
# ax.plot(v_temperature, 1e3 * v_results_08["dlnfCO2_dT"], **style_pyco2_08)
# ax.plot(
#     v_temperature,
#     1e3 * pyco2.upsilon.ups_quadratic_TOG93(v_temperature),
#     **style_poly,
# )
# ax.plot(
#     v_temperature,
#     1e3 * pyco2.upsilon.ups_TOG93_H24(v_temperature, pwtools.Rgas * 10),
#     **style_vh,
# )
# ax.plot(v_temperature, 1e3 * v_results_10["dlnfCO2_dT"], **style_pyco2_10)
# ax.set_xlabel("Temperature / °C")
# ax.set_ylabel("$υ$ / k°C$^{–1}$")
# ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))
# ax.set_ylim([30, 50])

# ax.legend(
#     loc="upper center", bbox_to_anchor=(0.5, -0.3), edgecolor="k", ncol=3, fontsize=9
# )

fig.tight_layout()
fig.savefig("figures_si/figure1_k1k2params.png")
