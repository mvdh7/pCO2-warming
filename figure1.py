from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from scipy.stats import linregress
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
alkalinity_10 = t93.get_alkalinity(10, opt_total_borate)[0]
alkalinity_08 = t93.get_alkalinity(8, opt_total_borate)[0]

# Calculate fCO2 variation with temperature with different approaches
v_temperature = np.linspace(-1.8, 35.83, num=100)

# - with PyCO2SYS (autograd approach)
v_results_10 = pyco2.sys(
    par1=alkalinity_10,
    par1_type=1,
    par2=t93.dic,
    par2_type=2,
    temperature=v_temperature,
    opt_k_carbonic=10,
    **t93.tak93,
)
v_fCO2_10 = v_results_10["fCO2"]
v_lnfCO2_PyCO2SYS_10 = np.log(v_fCO2_10)
v_results_08 = pyco2.sys(
    par1=alkalinity_08,
    par1_type=1,
    par2=t93.dic,
    par2_type=2,
    temperature=v_temperature,
    opt_k_carbonic=8,
    **t93.tak93,
)
v_fCO2_08 = v_results_08["fCO2"]
v_lnfCO2_PyCO2SYS_08 = np.log(v_fCO2_08)

# - with the linear temperature sensitivity of T93
zero_lnfCO2 = np.log(fCO2) - 0.0423 * t93.temperature
v_lnfCO2_linear = np.mean(zero_lnfCO2) + v_temperature * 0.0423

# - with the polynomial temperature sensitivity of T93
zero_lnfCO2_poly = (
    np.log(fCO2) - 0.0433 * t93.temperature + 0.5 * 8.7e-5 * t93.temperature**2
)
v_lnfCO2_poly = (
    np.mean(zero_lnfCO2_poly)
    + v_temperature * 0.0433
    - 0.5 * 8.7e-5 * v_temperature**2
)
f_linear = np.exp(v_temperature * lr_slope + lr_intercept)

# Visualise similar to Takahashi et al. (1993) Figure A1
style_linear = dict(
    c=pwtools.dark,
    label="$υ_l$ (Ta93, linear)",
    # alpha=0.9,
    lw=1.5,
)
style_poly = dict(
    c=pwtools.dark,
    label="$υ_q$ (Ta93, quadratic)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (6, 2)),
    zorder=2,
)
style_vh = dict(
    c=pwtools.blue,
    label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (3, 1)),
    zorder=1,
)
style_pyco2_10 = dict(
    c=pwtools.pink,
    label=r"$υ_\mathrm{Lu00}$ (PyCO2SYS, Lu00)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (2,)),
    zorder=3,
)
style_vht = style_vh.copy()
style_vht.update(
    dict(ls=(0, (3, 1, 1, 1)), label="$υ_x$ (van 't Hoff, ∆$_rH^⦵$)"), lw=1.5
)
style_pyco2_08 = dict(
    c=pwtools.pink,
    label=r"$υ_\mathrm{Mi79}$ (PyCO2SYS, Mi79)",
    # alpha=0.8,
    lw=1.5,
    ls=":",
    zorder=3,
)
fig, axs = plt.subplots(dpi=300, nrows=2, figsize=(12 / 2.54, 18 / 2.54))

ax = axs[0]
ax.text(0, 1.05, "(a)", transform=ax.transAxes)
ax.errorbar(
    t93.temperature,
    fCO2 - np.exp(t93.temperature * lr_slope + lr_intercept),
    np.sqrt(8),
    c="xkcd:dark",
    ls="none",
    zorder=-1,
)
ax.scatter(
    t93.temperature,
    fCO2 - np.exp(t93.temperature * lr_slope + lr_intercept),
    c="xkcd:dark",
    label="Ta93 measured",
    s=50,
    zorder=0,
    # alpha=0.9,
    edgecolor="none",
)
ax.axhline(0, **style_linear)
ax.plot(
    v_temperature,
    np.exp(v_lnfCO2_poly) - f_linear,
    **style_poly,
)
ax.plot(
    v_temperature,
    np.exp(pwtools.get_lnfCO2_vh([pwtools.bh_best, pwtools.ch_best], v_temperature))
    - f_linear,
    **style_vh,
)
ax.plot(
    v_temperature,
    np.exp(
        pwtools.get_lnfCO2_vh(
            [pwtools.bh_theory, pwtools.ch_theory_best], v_temperature
        )
    )
    - f_linear,
    **style_vht,
)
ax.plot(
    v_temperature,
    np.exp(v_lnfCO2_PyCO2SYS_10) - f_linear,
    **style_pyco2_10,
)
ax.plot(
    v_temperature,
    np.exp(v_lnfCO2_PyCO2SYS_08) - f_linear,
    **style_pyco2_08,
)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(
    "[{sp}{f}CO$_2$ $-$ {f}CO$_2$($υ_l$)] / µatm".format(
        sp=pwtools.thinspace, f=pwtools.f
    )
)
ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))
ax.set_ylim((-43, 23))

ax = axs[1]
ax.text(0, 1.05, "(b)", transform=ax.transAxes)
ax.axhline(1e2 * pyco2.upsilon.ups_linear_TOG93(), **style_linear)
ax.plot(
    v_temperature,
    1e2 * pyco2.upsilon.ups_enthalpy_H24(v_temperature, pwtools.Rgas * 10),
    **style_vht,
)
ax.plot(v_temperature, 1e2 * v_results_08["dlnfCO2_dT"], **style_pyco2_08)
ax.plot(
    v_temperature,
    1e2 * pyco2.upsilon.ups_quadratic_TOG93(v_temperature),
    **style_poly,
)
ax.plot(
    v_temperature,
    1e2 * pyco2.upsilon.ups_TOG93_H24(v_temperature, pwtools.Rgas * 10),
    **style_vh,
)
ax.plot(v_temperature, 1e2 * v_results_10["dlnfCO2_dT"], **style_pyco2_10)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("$υ$ / % °C$^{–1}$")
ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))
ax.set_ylim([3, 5])

ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.3), edgecolor="k", ncol=2, fontsize=9
)

for ax in axs:
    ax.grid(alpha=0.2)

fig.tight_layout()
# fig.savefig("figures_final/figure1.png")
