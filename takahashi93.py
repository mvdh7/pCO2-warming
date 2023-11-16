import PyCO2SYS as pyco2
import numpy as np
from matplotlib import pyplot as plt

# Data from Takahashi et al. (1993) Table A1
dic = 2074
pCO2 = np.array([571.2, 376.1, 267, 322.1, 461.6, 688.2, 571, 526.3])
temperature = np.array([20, 10.05, 2.1, 6.33, 15.01, 24.5, 20, 18])
tak93 = dict(
    salinity=35.38,
    total_silicate=0,
    total_phosphate=0,
    opt_k_carbonic=10,
)

# Determine alkalinity
r = pyco2.sys(
    par1=dic,
    par1_type=2,
    par2=pCO2,
    par2_type=4,
    temperature=temperature,
    **tak93,
)
alkalinity = np.mean(r["alkalinity"])
alkalinity_std = np.std(r["alkalinity"])

# Calculate pCO2 variation with temperature with different approaches
v_temperature = np.linspace(-1.8, 30, num=100)
v_results = pyco2.sys(
    par1=alkalinity,
    par1_type=1,
    par2=dic,
    par2_type=2,
    temperature=v_temperature,
    **tak93,
)
zero_lnpCO2 = np.log(pCO2) - 0.0423 * temperature
zero_lnpCO2_poly = np.log(pCO2) - 0.0433 * temperature + 8.7e-5 * temperature**2
v_pCO2 = v_results["pCO2"]
v_lnpCO2_PyCO2SYS = np.log(v_pCO2)
v_lnpCO2_linear = np.mean(zero_lnpCO2) + v_temperature * 0.0423
v_lnpCO2_poly = (
    np.mean(zero_lnpCO2_poly) + v_temperature * 0.0433 - 8.7e-5 * v_temperature**2
)
# TODO Takahashi et al. (2009) have 4.35e-5 instead of 8.7e-5 above

# Calculate 1-degree warming correction (i.e., seawater has warmed up by 1 °C before
# being measured)
warm1_PyCO2SYS = (
    pyco2.sys(
        par1=alkalinity,
        par1_type=1,
        par2=dic,
        par2_type=2,
        temperature=v_temperature - 1,
        **tak93,
    )["pCO2"]
    - v_pCO2
)
warm1_linear = v_pCO2 * np.exp(0.0423 * (v_temperature - (v_temperature + 1))) - v_pCO2
warm1_poly = (
    v_pCO2
    * np.exp(
        0.0433 * (v_temperature - (v_temperature + 1))
        - 8.7e-5 * (v_temperature**2 - (v_temperature + 1) ** 2)
    )
    - v_pCO2
)

# Visualise similar to Takahashi et al. (1993) Figure A1
style_linear = dict(c="k", label="T93 linear fit", alpha=0.9, lw=2.5)
style_poly = dict(label="T93 quadratic fit", alpha=0.9, lw=2.5, ls="--")
fig, axs = plt.subplots(dpi=300, nrows=3, figsize=(10 / 2.54, 21 / 2.54))
ax = axs[0]
ax.text(0, 1.03, "(a)", transform=ax.transAxes)
ax.scatter(
    temperature,
    # np.log(pCO2),
    pCO2,
    c="k",
    label="T93 measurements",
    s=90,
    zorder=0,
    alpha=0.9,
    edgecolor="none",
)
ax.plot(
    v_temperature,
    # v_lnpCO2_linear,
    np.exp(v_lnpCO2_linear),
    **style_linear,
)
ax.plot(
    v_temperature,
    # v_lnpCO2_poly,
    np.exp(v_lnpCO2_poly),
    **style_poly,
)
ax.plot(
    v_temperature,
    # v_lnpCO2_PyCO2SYS,
    np.exp(v_lnpCO2_PyCO2SYS),
    c="xkcd:coral",
    label="PyCO2SYS",
    alpha=0.9,
    lw=2.5,
    ls=":",
)
ax.legend()
ax.set_xlabel("Equilibration temperature / °C")
# ax.set_ylabel("ln ($p$CO$_2$ / µatm)")
ax.set_ylabel("$p$CO$_2$ / µatm")
ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))
ax = axs[1]
# TODO Plot the derivative against temperature here after adding it to PyCO2SYS properly
ax.text(0, 1.03, "(b)", transform=ax.transAxes)
ax.axhline(100 * 0.0423, **style_linear)
ax.plot(
    v_temperature, 100 * (0.0433 - 8.7e-5 * v_temperature), c="xkcd:azure", **style_poly
)
# ax.legend()
ax.set_xlabel("Equilibration temperature / °C")
ax.set_ylabel("100 × ∂(ln $p$CO$_2$)/∂$T$")
ax = axs[2]
# TODO Sanity check these calculations!!!
# TODO Repeat this calculation as a map of the global ocean --- is it the high pCO2
# or the high temperature that's causing the worst error?  Check the whole phase space!
ax.text(0, 1.03, "(c)", transform=ax.transAxes)
ax.plot(v_temperature, warm1_linear - warm1_PyCO2SYS, **style_linear)
ax.plot(v_temperature, warm1_poly - warm1_PyCO2SYS, **style_poly)
ax.axhline(0, c="k", lw=0.8)
ax.set_xlabel("Equilibration temperature / °C")
ax.set_ylabel("Warming corr. error / µatm")
fig.tight_layout()

# %% Calculate with PyCO2SYS
grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
    "k_bisulfate",
    "k_fluoride",
    "fugacity_factor",
    "k_silicate",
    "k_phosphoric_1",
    "k_phosphoric_2",
    "k_phosphoric_3",
]
rr = pyco2.sys(
    par1=alkalinity,
    par1_type=1,
    par2=dic,
    par2_type=2,
    temperature=25,
    **tak93,
    grads_of=["pCO2", *grads],
    grads_wrt=["temperature", *grads],
)
pCO2_wf = rr["d_pCO2__d_temperature"] / rr["pCO2"]
pCO2_wf_components = {}
pCO2_wf_percent = {}
pCO2_wf_sum = 0
pCO2_wf_percent_sum = 0
for k in grads:
    k_comp = (rr["d_" + k + "__d_temperature"] * rr["d_pCO2__d_" + k]) / rr["pCO2"]
    pCO2_wf_components[k] = k_comp
    pCO2_wf_percent[k] = 100 * k_comp / pCO2_wf
    pCO2_wf_sum += k_comp
    pCO2_wf_percent_sum += pCO2_wf_percent[k]
