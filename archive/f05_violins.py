from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from itertools import cycle
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np

glodap = xr.open_dataset("quickload/f03_glodap.nc")
glodap_i = glodap.copy()
glodap_i["silicate"] = glodap.silicate.interpolate_na("lat").interpolate_na("lon")
glodap_i["PO4"] = glodap.PO4.interpolate_na("lat").interpolate_na("lon")
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

for opt_k_carbonic in range(1, 18):
    o = opt_k_carbonic
    print(o)
    # Calculate surface field of dlnpCO2/dT
    results = pyco2.sys(
        par1=glodap.isel(depth_surface=0).TAlk.data,
        par2=glodap.isel(depth_surface=0).TCO2.data,
        par1_type=1,
        par2_type=2,
        temperature=glodap.isel(depth_surface=0).temperature.data,
        salinity=glodap.isel(depth_surface=0).salinity.data,
        total_silicate=glodap_i.isel(depth_surface=0).silicate.data,
        total_phosphate=glodap_i.isel(depth_surface=0).PO4.data,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        grads_of=["pCO2", *grads],
        grads_wrt=["temperature", *grads],
    )
    glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)] = (
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
        glodap[k] = (("lat", "lon"), k_comp)
        pCO2_wf_percent[o][k] = 100 * k_comp / pCO2_wf[o]
        pCO2_wf_sum[o] += k_comp
        pCO2_wf_percent_sum[o] += pCO2_wf_percent[o][k]

# %% Visualise - violins
colours = cycle(
    ["xkcd:sea blue", "xkcd:grey green", "xkcd:light orange", "xkcd:dusty rose"]
)
fig, ax = plt.subplots(dpi=300, figsize=(16 / 2.54, 8 / 2.54))
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
okc_order = [4, 10, 11, 13, 14, 15, 17, 6, 7, 12, 16, 1, 2, 3, 5, 9, 8]
for x, o in enumerate(okc_order):
    c = next(colours)
    fvar = pCO2_wf_components[o]["k_carbonic_1"] * 100
    parts = ax.violinplot(
        fvar[~np.isnan(fvar)],
        [i + x + 1],
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
ax.plot([2.4, 19.6], [4.23, 4.23], c="xkcd:dark", alpha=0.8, ls=":", lw=1)
ax.grid(alpha=0.2, axis="y")
ax.set_xlim((-1, 20))
ax.axhline(0, c="k", lw=0.8)
ax.set_yticks(np.arange(-6, 7, 1.5))
ax.set_ylim([-4.5, 6])
ax.set_xticks(np.arange(0, 20))
ax.text(2.45, 3.95, "Total →", va="center", ha="right", fontsize=9)
ax.text(2.45, -2.2, "$K_1^*$ →", va="center", ha="right", fontsize=9)
ax.text(2.45, 2.35, "$K_2^*$ →", va="center", ha="right", fontsize=9)
fxlabels = [r"$K_{\mathrm{CO}_2}^*$", r"$K_\mathrm{B}^*$", "$K_w^*$"]
for o in okc_order:
    fxlabels.append(str(o))
ax.set_xticklabels(fxlabels)
ax.set_ylabel("Contribution to $η$ / 10$^{-2}$ °C$^{–1}$")
ax.tick_params(top=True, labeltop=True)
fig.tight_layout()
fig.savefig("figures/f05_violins.png")
plt.show()
plt.close()
