from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pwtools

# Generated with si_predict_bh.py
soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")

# Get fit RMSD
ex_temperature = np.reshape(soda_monthly.ex_temperature.data, (1, 1, 1, 50))
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

# %% Draw figure
sm_shape = list(soda_monthly.ex_fCO2.data.shape)
sm_shape[-1] = 1
lo_temperature = np.tile(ex_temperature, sm_shape)
lo_diff = soda_monthly.ex_fCO2.data - soda_monthly.fCO2_from_bhch.data
# lo_diff = soda_monthly.HCO3.data
lo_dic_ta = soda_monthly.dic_ta.data
lo_colour = soda_monthly.dic_ta.data
L = lo_dic_ta < 9
lo_temperature = lo_temperature[L]
lo_diff = lo_diff[L]
lo_dic_ta = lo_dic_ta[L]
lo_colour = lo_colour[L]

fig, axs = plt.subplots(dpi=300, figsize=(17.4 / 2.54, 10 / 2.54), ncols=2)

ax = axs[0]
coarsen = 5  # 5 for final figure, 50 for quick testing
fx = lo_temperature[::coarsen].ravel()
fy = lo_diff[::coarsen].ravel()
fc = np.tile(lo_colour, (50, 1)).T[::coarsen].ravel()
fix = np.argsort(fc)
fx = fx[fix][::-1]
fy = fy[fix][::-1]
fc = fc[fix][::-1]
fs = ax.scatter(fx, fy, c=fc, s=5, cmap="plasma")
plt.colorbar(fs, label=r"$T_\mathrm{C}$ / $A_\mathrm{T}$", location="top", shrink=0.7)
# M = (fc >= 0.9365) & (fc <= 0.9366)
# ax.scatter(fx[M], fy[M], c='xkcd:strawberry')
ax.axhline(c="k", lw=0.8)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(
    "[{sp}{f}CO$_2$($υ_p$) $-$ {f}CO$_2$".format(f=pwtools.f, sp=pwtools.thinspace)
    + r"($A_\mathrm{T}$, $T_\mathrm{C}$)] / µatm"
)
ax.text(0, 1.06, "(a)", transform=ax.transAxes)

ax = axs[1]
fx2 = soda_monthly.dic_ta.data
fy2 = soda_monthly.fit_bh_rmsd.data
fc2 = soda_monthly.salinity.data
L2 = ~np.isnan(fy2)
fx2, fy2, fc2 = fx2[L2], fy2[L2], fc2[L2]
fix2 = np.argsort(fc2)
fx2 = fx2[fix2][::-1]
fy2 = fy2[fix2][::-1]
fc2 = fc2[fix2][::-1]
fs2 = ax.scatter(
    fx2,
    fy2,
    s=5,
    c=fc2,
    edgecolor="none",
    cmap="viridis",
)
ax.set_xlabel(r"$T_\mathrm{C}$ / $A_\mathrm{T}$")
ax.set_ylabel("RMSD of $b_h$ fit / µatm")
ax.set_ylim((0, 5.2))
ax.text(0, 1.06, "(b)", transform=ax.transAxes)
plt.colorbar(fs2, label="Practical salinity", location="top", shrink=0.7)

for ax in axs:
    ax.grid(alpha=0.2)

# ax.set_ylim([-10.5, 6.5])
# ax.set_title(r"$T_\mathrm{C}$ / $A_\mathrm{T}$ < 0.9")
fig.tight_layout()
fig.savefig("figures_final/figure3.png")
