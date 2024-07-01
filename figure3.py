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
show_highlights = False
show_cutoff_rmsd = True

cutoff_dic_ta = 0.9
percent_under_cutoff_dic_ta = (
    100
    * (soda_monthly.dic_ta < cutoff_dic_ta).sum()
    / soda_monthly.dic_ta.notnull().sum()
).data
print(
    "Over {:.0f}% of all data have DIC/TA < {}.  These are the purple points in (a).".format(
        np.floor(percent_under_cutoff_dic_ta), cutoff_dic_ta
    )
)
cutoff_rmsd = 1
percent_under_cutoff_rmsd = (
    100
    * (soda_monthly.fit_bh_rmsd < cutoff_rmsd).sum()
    / soda_monthly.dic_ta.notnull().sum()
).data
print(
    "Over {:.0f}% of all data have RMSD < {} µatm.".format(
        np.floor(percent_under_cutoff_rmsd), cutoff_rmsd
    )
)

sm_shape = list(soda_monthly.ex_fCO2.data.shape)
sm_shape[-1] = 1
lo_temperature = np.tile(ex_temperature, sm_shape)
lo_diff = soda_monthly.ex_fCO2.data - soda_monthly.fCO2_from_bhch.data
# lo_diff = soda_monthly.HCO3.data
lo_dic_ta = soda_monthly.dic_ta.data
lo_colour = soda_monthly.dic_ta.data
lo_rmsd = soda_monthly.fit_bh_rmsd.data
L = lo_dic_ta < 9
lo_temperature = lo_temperature[L]
lo_diff = lo_diff[L]
lo_dic_ta = lo_dic_ta[L]
lo_colour = lo_colour[L]
lo_rmsd = lo_rmsd[L]

v_temperature = lo_temperature[0]
v_diff_cutoff_min = np.full_like(v_temperature, np.nan)
v_diff_cutoff_max = np.full_like(v_temperature, np.nan)
for t in range(len(v_temperature)):
    # v_diff_cutoff_min[t] = np.min(lo_diff[lo_dic_ta < cutoff_dic_ta, t])
    # v_diff_cutoff_max[t] = np.max(lo_diff[lo_dic_ta < cutoff_dic_ta, t])
    v_diff_cutoff_min[t] = np.min(lo_diff[lo_rmsd < 1, t])
    v_diff_cutoff_max[t] = np.max(lo_diff[lo_rmsd < 1, t])

ix_lo = 47030
ix_hi = 111430

fig, axs = plt.subplots(dpi=300, figsize=(17.4 / 2.54, 10 / 2.54), ncols=2)

ax = axs[0]
coarsen = 500  # 5 for final figure (~90 seconds), 50 for quick testing
fx = lo_temperature[::coarsen].ravel()
fy = lo_diff[::coarsen].ravel()
fc = np.tile(lo_colour, (50, 1)).T[::coarsen].ravel()
fix = np.argsort(fc)
fx = fx[fix][::-1]
fy = fy[fix][::-1]
fc = fc[fix][::-1]
fs = ax.scatter(fx, fy, c=fc, s=8, cmap="plasma")
plt.colorbar(fs, label=r"$T_\mathrm{C}$ / $A_\mathrm{T}$", location="top", shrink=0.7)
if show_highlights:
    ax.plot(v_temperature, lo_diff[ix_lo, :], c="xkcd:bright pink", lw=1.2, ls="--")
    ax.plot(v_temperature, lo_diff[ix_hi, :], c="xkcd:bright pink", lw=1.2)
if show_cutoff_rmsd:
    offset = 0.15
    # ^ this shifts the lines so they fall at the edges of the points to be enclosed
    #   instead of running through the middle of them.  It does not invalidate the
    #   statements made about theh lines (by increasing the space between them,
    #   even more than 97% of the data will be enclosed)
    ax.plot(
        v_temperature,
        v_diff_cutoff_min - offset,
        c="xkcd:aqua blue",
        lw=1.5,
        alpha=0.8,
    )
    ax.plot(
        v_temperature,
        v_diff_cutoff_max + offset,
        c="xkcd:aqua blue",
        lw=1.5,
        alpha=0.8,
    )
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
if show_highlights:
    ax.scatter(lo_dic_ta[ix_hi], lo_rmsd[ix_hi], c="xkcd:bright pink", marker="s", s=20)
    ax.scatter(
        lo_dic_ta[ix_lo],
        lo_rmsd[ix_lo],
        c="none",
        marker="s",
        s=20,
        edgecolor="xkcd:bright pink",
    )
if show_cutoff_rmsd:
    ax.axhline(cutoff_rmsd, c="xkcd:aqua blue", lw=1.5, alpha=0.8)
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
# if show_highlights:
#     fig.savefig("figures_final/figure3_hl.png")
# else:
#     fig.savefig("figures_final/figure3.png")
