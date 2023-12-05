from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from itertools import cycle
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np

soda = xr.open_dataset("quickload/f06_soda.nc")
opt_total_borate = 1

grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
]

pCO2_wf = {}
pCO2_wf_components = {}
pCO2_wf_percent = {}
pCO2_wf_sum = {}
pCO2_wf_percent_sum = {}

for opt_k_carbonic in range(1, 19):
    o = opt_k_carbonic
    print(o)
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
        results["dlnpCO2_dT"] * 100,
    )
    pCO2_wf[o] = results["d_pCO2__d_temperature"] / results["pCO2"]
    pCO2_wf_components[o] = {}
    pCO2_wf_percent[o] = {}
    pCO2_wf_sum[o] = 0
    pCO2_wf_percent_sum[o] = 0
    for k in grads:
        k_comp = (
            results["d_" + k + "__d_temperature"] * results["d_pCO2__d_" + k]
        ) / results["pCO2"]
        pCO2_wf_components[o][k] = k_comp
        soda[k] = (("lat", "lon"), k_comp)
        pCO2_wf_percent[o][k] = 100 * k_comp / pCO2_wf[o]
        pCO2_wf_sum[o] += k_comp
        pCO2_wf_percent_sum[o] += pCO2_wf_percent[o][k]

# %% Visualise - violins
colours = cycle(
    ["xkcd:sea blue", "xkcd:grey green", "xkcd:light orange", "xkcd:dusty rose"]
)
fig, ax = plt.subplots(dpi=300, figsize=(17.4 / 2.54, 11 / 2.54))
fvars = ["k_CO2", "k_borate", "k_water"]
widths = 0.85
for i, var in enumerate(fvars):
    fvar = pCO2_wf_components[10][var] * 100
    parts = ax.violinplot(
        fvar[~np.isnan(fvar)],
        [i],
        showextrema=False,
        points=100,
        widths=widths,
    )
    parts["bodies"][0].set_facecolor("xkcd:dark")
    parts["bodies"][0].set_alpha(0.8)
okc_order = [4, 10, 11, 13, 14, 15, 17, 6, 7, 12, 16, 1, 2, 3, 5, 9, 8, 18]
okc_codes = {
    1: "Ro93",
    2: "GP89",
    3: "DM87-H",
    4: "DM87-M",
    5: "DM87-HM",
    6: "Me73",
    7: "Me73-P",
    8: "Mi79",
    9: "CW98",
    10: "Lu00",
    11: "MM02",
    12: "Mi02",
    13: "Mi06",
    14: "Mi10",
    15: "Wa13",
    16: "Su20",
    17: "SB21",
    18: "Pa18",
}
assert len(set(okc_order)) == 18
assert np.all(np.isin(okc_order, range(1, 19)))
for x, o in enumerate(okc_order):
    xpos = i + x + 1
    ax.text(
        xpos, 6.5, okc_codes[o], ha="center", va="bottom", rotation=90, fontsize=8.5
    )
    c = next(colours)
    fvar = pCO2_wf_components[o]["k_carbonic_1"] * 100
    parts = ax.violinplot(
        fvar[~np.isnan(fvar)],
        [xpos],
        showextrema=False,
        points=100,
        widths=widths,
    )
    parts["bodies"][0].set_facecolor(c)
    parts["bodies"][0].set_alpha(0.6)
    fvar = pCO2_wf_components[o]["k_carbonic_2"] * 100
    parts = ax.violinplot(
        fvar[~np.isnan(fvar)],
        [i + x + 1],
        showextrema=False,
        points=100,
        widths=widths,
    )
    parts["bodies"][0].set_facecolor(c)
    parts["bodies"][0].set_alpha(0.4)
    fvar = pCO2_wf[o] * 100
    parts = ax.violinplot(
        fvar[~np.isnan(fvar)],
        [i + x + 1],
        showextrema=False,
        points=100,
        widths=widths,
    )
    parts["bodies"][0].set_facecolor(c)
    parts["bodies"][0].set_alpha(0.8)
ax.plot([2.4, 20.6], [4.23, 4.23], c="xkcd:dark", alpha=0.8, ls=":", lw=1.5)
ax.grid(alpha=0.3, axis="y")
ax.grid(alpha=0.05, axis="x")
ax.set_xlim((-1, 21))
ax.axhline(0, c="k", lw=0.8)
ax.set_yticks(np.arange(-6, 7, 1.5))
ax.set_ylim([-4.5, 6])
ax.set_xticks(np.arange(0, 21))
ax.text(2.45, 3.95, "Total →", va="center", ha="right", fontsize=9)
ax.text(2.45, -2.2, "$K_1^*$ →", va="center", ha="right", fontsize=9)
ax.text(2.45, 2.35, "$K_2^*$ →", va="center", ha="right", fontsize=9)
fxlabels = [r"$K_{\mathrm{CO}_2}^*$", r"$K_\mathrm{B}^*$", "$K_w^*$"]
for i, fxl in enumerate(fxlabels):
    ax.text(i, 6.4, fxl, ha="center", va="bottom")
for o in okc_order:
    fxlabels.append(str(o))
ax.set_xticklabels(fxlabels)
ax.set_ylabel("Contribution to $η$ / 10$^{-2}$ °C$^{–1}$")
ax.tick_params(top=True, labeltop=False)
brackets = dict(
    xycoords="data",
    textcoords="data",
    arrowprops=dict(
        arrowstyle="-",
        connectionstyle="arc, armA=15, armB=15, angleA=-90, angleB=-90, rad=35",
        ec="xkcd:grey blue",
    ),
    annotation_clip=False,
)
anx = -6
ax.annotate("", xy=(3, anx), xytext=(9, anx), **brackets)
ax.text(6, anx - 0.6, "Mehrbach", ha="center", va="top", c="xkcd:grey blue")
ax.annotate("", xy=(10, anx), xytext=(11, anx), **brackets)
ax.text(10.5, anx - 0.6, "GEOSECS", ha="center", va="top", c="xkcd:grey blue")
ax.annotate("", xy=(14, anx), xytext=(17, anx), **brackets)
ax.text(15.5, anx - 0.6, "Synthetic", ha="center", va="top", c="xkcd:grey blue")
fig.tight_layout()
fig.savefig("figures/f07_violins_soda.png")
plt.show()
plt.close()
