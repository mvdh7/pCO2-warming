from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt

# Import OceanSODA-ETZH product
soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")

# Calculate upsilon lines
dic = 2045  # soda_monthly.dic.median() = 2045
dic_alk = np.vstack(np.arange(0.6, 1.201, 0.05))
alkalinity = dic / dic_alk
temperature = np.linspace(-3, 35)
results = pyco2.sys(
    par1=alkalinity,
    par1_type=1,
    par2=dic,
    par2_type=2,
    temperature=temperature,
    opt_k_carbonic=10,
    opt_total_borate=1,
)

# Draw figure
fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 7 / 2.54))
for i, da in enumerate(dic_alk.ravel()):
    # Make lines that are labelled on the colorbar slightly thicker
    if i % 2 == 0:
        lw = 0.8
    else:
        lw = 0.5
    # Plot lines
    fl = ax.plot(
        temperature,
        results["dlnfCO2_dT"][i] * 100,
        c=mpl.cm.plasma((da - 0.6) / 0.6),
        lw=lw,
        alpha=1,
    )

# Scatter OceanSODA-ETZH product
fc = ax.scatter(
    soda_monthly.temperature.values.ravel(),
    soda_monthly.dlnfCO2_dT.values.ravel() / 10,
    s=1,
    c="xkcd:cloudy blue",
    edgecolor="none",
    alpha=0.05,
)

# Figure settings
ax.set_ylim(3.8, 4.8)
ax.set_xlim(temperature.min(), temperature.max())
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(r"$υ_\mathrm{Lu00}$ / % °C$^{-1}$")

# Create colorbar
cmap = mpl.cm.plasma
bounds = np.array([*(dic_alk.ravel() - 0.025), dic_alk.max() + 0.025])
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="plasma"),
    ax=ax,
    orientation="vertical",
    label=r"$T_\mathrm{C}$ / $A_\mathrm{T}$",
    ticks=dic_alk.ravel()[::2],
)
cb.ax.yaxis.set_ticks(dic_alk.ravel()[1::2], minor=True)

# # Check the cancelling-out effect of Wanninkhof et al. (2022)
# ax.grid()
# ax.scatter([30, 30, -1.7, -1.7], [3.85, 3.97, 4.72, 4.6], marker='x')

# Finish off
fig.tight_layout()
fig.savefig("figures_final/figure7r.png")
