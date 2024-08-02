from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares
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

# %% Calculate non-carbonate alkalinity
results = pyco2.sys(
    par1=soda_monthly.talk.data,
    par1_type=1,
    par2=soda_monthly.dic.data,
    par2_type=2,
    salinity=soda_monthly.salinity.data,
    temperature=soda_monthly.temperature.data,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
# %%
soda_monthly["alk_carb_bicarb"] = (
    ("month", "lat", "lon"),
    results["HCO3"] + 2 * results["CO3"],
)
soda_monthly["alk_non_carb_bicarb"] = soda_monthly.talk - soda_monthly.alk_carb_bicarb
for v in [
    # "pH",
    "CO2",
    # "alkalinity_borate",
    # "isocapnic_quotient",
    # "beta_dic",
    # "gamma_dic",
    # "beta_alk",
    # "gamma_alk",
    # "revelle_factor",
    "HCO3",
    "CO3",
]:
    soda_monthly[v] = (("month", "lat", "lon"), results[v])
soda_monthly["CO2_dic_pct"] = 100 * soda_monthly.CO2 / soda_monthly.dic
soda_monthly["alk_non_carb_pct"] = (
    100 * soda_monthly.alk_non_carb_bicarb / soda_monthly.talk
)
soda_monthly["alk_approx_error"] = soda_monthly.alk_carb_bicarb - soda_monthly.talk
soda_monthly["dic_approx_error"] = (
    soda_monthly.HCO3 + soda_monthly.CO3 - soda_monthly.dic
)
soda_monthly["alk_approx_error_pct"] = (
    100 * soda_monthly.alk_approx_error / soda_monthly.talk
)
soda_monthly["dic_approx_error_pct"] = (
    100 * soda_monthly.dic_approx_error / soda_monthly.dic
)

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
for vt in range(len(v_temperature)):
    # v_diff_cutoff_min[t] = np.min(lo_diff[lo_dic_ta < cutoff_dic_ta, t])
    # v_diff_cutoff_max[t] = np.max(lo_diff[lo_dic_ta < cutoff_dic_ta, t])
    v_diff_cutoff_min[vt] = np.min(lo_diff[lo_rmsd < 1, vt])
    v_diff_cutoff_max[vt] = np.max(lo_diff[lo_rmsd < 1, vt])

ix_lo = 47030
ix_hi = 111430

daxlim = (0.81, 1.015)

fig, axs = plt.subplots(
    dpi=300, figsize=(12 / 2.54, 18 / 2.54), nrows=3, layout="constrained"
)

ax = axs[0]
coarsen = 5  # 5 for final figure (~90 seconds), 50 for quick testing
fx = lo_temperature[::coarsen].ravel()
fy = lo_diff[::coarsen].ravel()
fc = np.tile(lo_colour, (50, 1)).T[::coarsen].ravel()
fix = np.argsort(fc)
fx = fx[fix][::-1]
fy = fy[fix][::-1]
fc = fc[fix][::-1]
fs = ax.scatter(fx, fy, c=fc, s=8, cmap="plasma")
plt.colorbar(fs, label=r"$T_\mathrm{C}$ / $A_\mathrm{T}$", location="right")
if show_highlights:
    ax.plot(v_temperature, lo_diff[ix_lo, :], c="xkcd:bright pink", lw=1.2, ls="--")
    ax.plot(v_temperature, lo_diff[ix_hi, :], c="xkcd:bright pink", lw=1.2)
if show_cutoff_rmsd:
    offset = 0.2
    # ^ this shifts the lines so they fall at the edges of the points to be enclosed
    #   instead of running through the middle of them.  It does not invalidate the
    #   statements made about the lines (by increasing the space between them,
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
# ax.set_ylabel(
#     "[{sp}{f}CO$_2$($υ_p$) $-$ {f}CO$_2$".format(f=pwtools.f, sp=pwtools.thinspace)
#     + r"($A_\mathrm{T}$, $T_\mathrm{C}$)]" + "\n/ µatm"
# )
ax.set_ylabel("Error in {}CO$_2$($υ_p$) / µatm".format(pwtools.f))
ax.text(0, 1.05, "(a)", transform=ax.transAxes)
ax.set_ylim(-12, 6)
ax.set_yticks(range(-12, 9, 3))

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
ax.set_ylim(0, 5.3)
ax.set_xlim(daxlim)
ax.text(0, 1.05, "(b)", transform=ax.transAxes)
plt.colorbar(fs2, label="Practical salinity", location="right", ticks=(10, 20, 30, 39))

ax = axs[2]
ax.scatter(
    soda_monthly.dic_ta.values.ravel(),
    soda_monthly.dic_approx_error_pct.values.ravel(),
    s=8,
    alpha=0.2,
    edgecolor="none",
    # label=r"$T_\mathrm{C}$",
    c="xkcd:dark",
)
ax.scatter(
    soda_monthly.dic_ta.values.ravel(),
    soda_monthly.alk_approx_error_pct.values.ravel(),
    s=8,
    alpha=0.2,
    edgecolor="none",
    # label=r"$A_\mathrm{T}$",
    c="xkcd:cloudy blue",
)
ax.scatter(
    1,
    1,
    s=15,
    edgecolor="none",
    label=r"$T_x - T_\mathrm{C}$",
    c="xkcd:dark",
    alpha=0.8,
)
ax.scatter(
    1,
    1,
    s=15,
    edgecolor="none",
    label=r"$A_x - A_\mathrm{T}$",
    c="xkcd:cloudy blue",
    alpha=0.8,
)
ax.set_xlim(daxlim)
ax.set_ylim(-6, 0)
ax.set_xlabel(r"$T_\mathrm{C}$ / $A_\mathrm{T}$")
ax.set_ylabel("Error in approximation / %")
ax.text(0, 1.05, "(c)", transform=ax.transAxes)
ax.legend()

for ax in axs:
    ax.grid(alpha=0.2)

# ax.set_ylim([-10.5, 6.5])
# ax.set_title(r"$T_\mathrm{C}$ / $A_\mathrm{T}$ < 0.9")
# fig.tight_layout()
# if show_highlights:
#     fig.savefig("figures_final/figure3_hl.png")
# else:
#     fig.savefig("figures_final/figure3.png")
fig.savefig("figures_final/figure1r.png")

# %% Do pure water comparison
# # rng = np.random.default_rng(7)
# t = np.reshape(ex_temperature, (50, 1, 1))
# dic = np.linspace(1, 3000, num=120)
# alkalinity = np.linspace(-1000, 3000, num=160)
# dic, alkalinity = np.meshgrid(dic, alkalinity)
# rpure = pyco2.sys(
#     par1=np.array([dic]),
#     par2=np.array([alkalinity]),
#     par1_type=2,
#     par2_type=1,
#     temperature=t,
#     opt_k_carbonic=8,
#     salinity=0,
#     # opt_k_carbonic=10,
#     # salinity=35,
# )

# pwfit_x = np.full(dic.shape, np.nan)
# pwfit_rmsd = np.full(dic.shape, np.nan)
# pwfit_rmsd_norm = np.full(dic.shape, np.nan)
# pwfit_rmsd_pct = np.full(dic.shape, np.nan)
# for r in range(dic.shape[0]):
#     for c in range(dic.shape[1]):
#         pwfit = pwtools.fit_fCO2_vht(t.ravel(), np.log(rpure["fCO2"][:, r, c].ravel()))
#         pwfit_x[r, c] = pwfit["x"][0]
#         pred_fCO2 = np.exp(pwtools.get_lnfCO2_vht(pwfit["x"][0], t.ravel()))
#         pwfit_rmsd[r, c] = np.sqrt(
#             np.mean((rpure["fCO2"][:, r, c].ravel() - pred_fCO2) ** 2)
#         )
#         pwfit_rmsd_norm[r, c] = pwfit_rmsd[r, c] / (
#             np.max(rpure["fCO2"][:, r, c]) - np.min(rpure["fCO2"][:, r, c])
#         )
#         # pwfit_rmsd_pct[r, c] = np.sqrt(
#         #     np.mean(
#         #         (
#         #             100
#         #             * (rpure["fCO2"][:, r, c].ravel() - pred_fCO2)
#         #             / rpure["fCO2"][:, r, c].ravel()
#         #         )
#         #         ** 2
#         #     )
#         # )

# # %%
# pvar = np.log10(pwfit_rmsd_norm * 650)

# fig, ax = plt.subplots(dpi=300)
# vmin = 0
# vmax = 2
# fc = ax.contourf(alkalinity, dic, pvar, 1024, cmap="turbo", vmin=vmin, vmax=vmax)

# # fig.colorbar(fc)

# cb = fig.colorbar(
#     mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap="turbo"),
#     ax=ax,
#     orientation="vertical",
#     label="log$_{10}$ (RMSD / µatm)",
#     extend="both",
#     shrink=0.8,
#     # ticks=range(-3, 4),
# )
# # cb.ax.set_yticklabels(["10$^{" + "{}".format(i) + "}$" for i in range(-3, 4)])
# # calpha = 0.9
# # fc = ax.contour(
# #     alkalinity,
# #     dic,
# #     np.log10(pvar),
# #     [-3, -2, -1],
# #     colors="w",
# #     linestyles="--",
# #     linewidths=1,
# #     alpha=calpha,
# # )
# # fc = ax.contour(
# #     alkalinity,
# #     dic,
# #     np.log10(pvar),
# #     [1, 2, 3],
# #     colors="w",
# #     linestyles=":",
# #     linewidths=1,
# #     alpha=calpha,
# # )
# # # ax.clabel(fc, [-3, -2, -1, 1, 2, 3], fontsize=8)
# # fc = ax.contour(
# #     alkalinity,
# #     dic,
# #     np.log10(pvar),
# #     [0],
# #     colors="w",
# #     linestyles="-",
# #     linewidths=1,
# #     alpha=calpha,
# # )
# # # ax.axline((0, 0), slope=1.4, c="k", ls="--")  # slope=0.75 ~matches the 0 contour
# # # ax.clabel(fc, [0], fontsize=8, inline_spacing=5)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 3000)
# ax.set_aspect(1)
# ax.set_xlabel(r"$A_\mathrm{T}$ / µmol kg-sw$^{-1}$")
# ax.set_ylabel(r"$T_\mathrm{C}$ / µmol kg-sw$^{-1}$")
# fig.tight_layout()


# # %%
# dars = [0.015, 0.1, 0.36, 0.756, 1.00725, 1.0461, 1.438]
# fig, ax = plt.subplots(dpi=300)
# for i, dic_alk_ratio in enumerate(dars):
#     s_rpure = pyco2.sys(
#         par1=1500 * dic_alk_ratio,
#         par2=1500,
#         par1_type=2,
#         par2_type=1,
#         temperature=t.ravel(),
#         opt_k_carbonic=8,
#         salinity=0,
#     )

#     s_pwfit = pwtools.fit_fCO2_vht(t.ravel(), np.log(s_rpure["fCO2"]))
#     s_pwfit_x = s_pwfit["x"][0]
#     s_pred_fCO2 = np.exp(pwtools.get_lnfCO2_vht(s_pwfit["x"][0], t.ravel()))
#     s_pwfit_rmsd = np.sqrt(np.mean((s_rpure["fCO2"] - s_pred_fCO2) ** 2))
#     print(s_pwfit_rmsd)

#     if i < 3:
#         ls = "--"
#     elif i > 3:
#         ls = ":"
#     else:
#         ls = "-"
#     ax.plot(
#         t.ravel(),
#         s_pred_fCO2 - s_rpure["fCO2"],
#         c=mpl.cm.turbo((np.log10(s_pwfit_rmsd) + 3) / 6),
#         # label="{:.0f}".format(np.log10(s_pwfit_rmsd)),
#         label="{:.3f}".format(dic_alk_ratio),
#         ls=ls,
#         lw=2,
#     )
# ax.axhline(0, c="k", lw=0.8)
# ax.set_xlim(np.min(t), np.max(t))
# ax.set_ylim(-12, 12)
# ax.set_yticks((-12, -8, -4, 0, 4, 8, 12))
# ax.set_xlabel("Temperature / °C")
# ax.set_ylabel(
#     r"[{sp}{f}CO$_2(υ_h)$ – {f}CO$_2(υ_\mathrm".format(
#         f=pwtools.f, sp=pwtools.thinspace
#     )
#     + "{Mi79})$] / µatm"
# )
# ax.legend(ncols=2)
# fig.tight_layout()
# fig.savefig("figures_si/figure3_pure_misfit_examples.png")

# %%
# dars = [0.8, 1, 1.2]
# letters = ["a", "b", "c"]
# fig, axs = plt.subplots(dpi=300, nrows=1, ncols=3, figsize=(7, 2.6))
# for i, ax in enumerate(axs.ravel()):
#     dic_alk_ratio = dars[i]
#     s_rpure = pyco2.sys(
#         par1=2000 * dic_alk_ratio,
#         par2=2000,
#         par1_type=2,
#         par2_type=1,
#         temperature=t.ravel(),
#         # opt_k_carbonic=8,
#         # salinity=0,
#         opt_k_carbonic=10,
#         salinity=35,
#     )

#     s_pwfit = pwtools.fit_fCO2_vht(t.ravel(), np.log(s_rpure["fCO2"]))
#     s_pwfit_x = s_pwfit["x"][0]
#     s_pred_fCO2 = np.exp(pwtools.get_lnfCO2_vht(s_pwfit["x"][0], t.ravel()))
#     s_pwfit_rmsd = np.sqrt(np.mean((s_rpure["fCO2"] - s_pred_fCO2) ** 2))
#     ax.plot(t.ravel(), s_rpure["fCO2"], c="xkcd:dark")
#     ax.plot(t.ravel(), s_pred_fCO2, c="xkcd:steel", ls=":")
#     ax.set_xlabel("$t$ / °C")
#     ax.set_ylabel("{f}CO$_2$ / µatm".format(f=pwtools.f))
#     # ax.set_title("{:.0f}".format(np.log10(s_pwfit_rmsd)))
#     ax.text(
#         0,
#         1.05,
#         "("
#         + letters[i]
#         + r") $T_\mathrm{C}/A_\mathrm{T}$ "
#         + "= {:.1f}".format(dic_alk_ratio),
#         transform=ax.transAxes,
#     )
#     # print(s_rpure['fCO2'].max() - s_rpure['fCO2'].min())
# fig.tight_layout()

# # %%
# r_opt_k_carbonic = 8
# r_par1 = {8: 2072, 10: 2263}
# r_pyco2 = pyco2.sys(
#     par1=r_par1[r_opt_k_carbonic],
#     par2=2074,
#     par1_type=1,
#     par2_type=2,
#     salinity=35.38,
#     temperature=np.linspace(-1, 35),
#     opt_k_carbonic=r_opt_k_carbonic,
# )
# pwfit_ch = pwtools.fit_fCO2_vht(r_pyco2["temperature"], np.log(r_pyco2["fCO2"]))
# pwfit_bhch = pwtools.fit_fCO2_vh(r_pyco2["temperature"], np.log(r_pyco2["fCO2"]))
# r_pyco2["fCO2_ch"] = np.exp(
#     pwtools.get_lnfCO2_vht(pwfit_ch["x"], r_pyco2["temperature"])
# )
# r_pyco2["fCO2_bhch"] = np.exp(
#     pwtools.get_lnfCO2_vh(pwfit_bhch["x"], r_pyco2["temperature"])
# )


# fig, ax = plt.subplots(dpi=300)
# ax.plot("temperature", "fCO2", data=r_pyco2, label="Lu00")
# ax.plot("temperature", "fCO2_ch", data=r_pyco2, label="ch", ls=":")
# ax.plot("temperature", "fCO2_bhch", data=r_pyco2, label="bhch", ls=":")
# ax.legend()

# # %%
# v_alkalinity = np.arange(0, 3000)
# v_dic = 1500  # 3000 - v_alkalinity
# v_dar = v_dic / v_alkalinity
# v_temperature = np.vstack(t.ravel())
# v_results = pyco2.sys(
#     par1=v_dic,
#     par2=v_alkalinity,
#     par1_type=2,
#     par2_type=1,
#     temperature=v_temperature,
#     opt_k_carbonic=8,
#     salinity=0,
# )
# v_pwfit_x = np.full(v_dar.shape, np.nan)
# v_pwfit_rmsd = np.full(v_dar.shape, np.nan)
# v_pred_fCO2 = np.full(v_results["fCO2"].shape, np.nan)
# for i in range(len(v_alkalinity)):
#     v_pwfit = pwtools.fit_fCO2_vht(
#         v_temperature.ravel(), np.log(v_results["fCO2"][:, i])
#     )
#     v_pwfit_x[i] = v_pwfit["x"][0]
#     v_pred_fCO2[:, i] = np.exp(
#         pwtools.get_lnfCO2_vht(v_pwfit["x"][0], v_temperature.ravel())
#     )
#     v_pwfit_rmsd[i] = np.sqrt(
#         np.mean((v_results["fCO2"][:, i] - v_pred_fCO2[:, i]) ** 2)
#     )

# # %%
# fig, ax = plt.subplots(dpi=300)
# # ax.plot(np.log10(v_dar), np.log10(v_pwfit_rmsd))
# # ax.set_xlim(-0.1, 0.1)
# ax.plot(v_dar, np.log10(v_pwfit_rmsd))
# ax.axhline(1)
# ax.set_xlim(0.8, 1.2)
# ax.axvline(0, c="k", lw=0.8)
