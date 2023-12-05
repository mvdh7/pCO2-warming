from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from takahashi93 import get_alkalinity, dic, tak93, pCO2, temperature

rng = np.random.default_rng(1)
opt_total_borate = 1

# Calculate the precision of Takahashi's pCO2 measurements
for opt_k_carbonic in [10]:  # range(1, 18):
    # opt_k_carbonic=10 gives the least pattern in pCO2_error with temperature,
    # except for maybe 6 (which also has slightly better precision...)
    # T93 note that the Mehrbach constants (4) fit well to the experimental data
    # (and that Hansson (3/5) and Goyet/Poisson (2) do not)
    alkalinity, alkalinity_std = get_alkalinity(opt_k_carbonic, opt_total_borate)
    results = pyco2.sys(
        par1=alkalinity,
        par1_type=1,
        par2=dic,
        par2_type=2,
        temperature=temperature,
        opt_k_carbonic=opt_k_carbonic,
        **tak93,
    )
    pCO2_error = pCO2 - results["pCO2"]
    pCO2_precision = pCO2_error.std()
    # ^ 2.6 µatm for opt_k_carbonic=10; T93 state "precision... approximately ±2 µatm"
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(temperature, pCO2_error)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("Temperature / °C")
    ax.set_ylabel("$p$CO$_2$ error / µatm")
    ax.set_title("opt_k_carbonic = {}".format(opt_k_carbonic))
    fig.tight_layout()
    plt.show()
    plt.close()

# %% Simulate the experiment with uncertainties in pCO2 and temperature
# pCO2_precision = 2  # from study - use value from above instead
pCO2_bias = 2  # guessed
temperature_precision = 0.005  # based on decimal places of reporting, uniform distro
temperature_bias = 0.005
# ^ WOCE-era accuracy from
# https://odv.awi.de/fileadmin/user_upload/odv/data/Gouretski-Koltermann-2004/ ...
# ... BSH35_report_final.pdf

nreps = 10_000
sim_pCO2 = results["pCO2"] * np.ones((nreps, 1))
# Add the random uncertainty in pCO2
sim_pCO2 += rng.normal(loc=0, scale=pCO2_precision, size=(nreps, len(pCO2)))
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_pCO2 += rng.normal(loc=0, scale=pCO2_bias, size=(nreps, 1))
sim_temperature = temperature * np.ones((nreps, 1))
# Add the random uncertainty in temperature (from decimal places)
sim_temperature += rng.uniform(
    low=-temperature_precision, high=temperature_precision, size=(nreps, len(pCO2))
)
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_temperature += rng.normal(loc=0, scale=temperature_bias, size=(nreps, 1))
sim_slope = np.full(nreps, np.nan)
sim_intercept = np.full(nreps, np.nan)
sim_polyfit = []
for i in range(nreps):
    i_lr = linregress(sim_temperature[i], np.log(sim_pCO2[i]))
    sim_slope[i] = i_lr.slope
    sim_intercept[i] = i_lr.intercept
    sim_polyfit.append(np.polyfit(sim_temperature[i], np.log(sim_pCO2[i]), 2))
sim_polyfit = np.array(sim_polyfit)

# Get statistics
sim_slope_precision = sim_slope.std()
print("Linear slope precision:  {:.5f}".format(sim_slope_precision))
sim_poly_slope_precision = sim_polyfit[:, 1].std()
sim_poly_squared_precision = sim_polyfit[:, 0].std()
sim_poly_covmx = np.cov(sim_polyfit[:, :2].T)
print("Poly   slope precision:  {:.5f}".format(sim_poly_slope_precision))
print("Poly squared precision:  {:.5f}".format(sim_poly_squared_precision))
print("Poly        covariance: {:.2e}".format(sim_poly_covmx[0, 1]))

# %% Polynomial uncertainty propagation
fxl = np.linspace(np.min(temperature), np.max(temperature))
jac_poly = np.array([fxl, np.ones_like(fxl)]).T
uncert_poly = np.sqrt(np.diag(jac_poly @ sim_poly_covmx @ jac_poly.T))

fxl_soda = np.linspace(-1.8, 35.83)
jac_poly_soda = np.array([fxl_soda, np.ones_like(fxl_soda)]).T
uncert_poly_soda = np.sqrt(np.diag(jac_poly_soda @ sim_poly_covmx @ jac_poly_soda.T))

fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 8 / 2.54))
ax.fill_betweenx(
    [0, 2], fxl[0], fxl[-1], facecolor="xkcd:navy", alpha=0.2, label="T93 $T$ range"
)
ax.plot(fxl_soda, uncert_poly_soda * 100, c="xkcd:navy", lw=2, label="Quadratic fit")
ax.set_xlim([fxl_soda[0], fxl_soda[-1]])
ax.set_ylim([0.03, 0.18])
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("1$σ$ uncertainty in $η$ / 10$^{-2}$ °C$^{–1}$")
ax.axhline(sim_slope_precision * 100, c="xkcd:navy", lw=2, ls=":", label="Linear fit")
ax.legend()
fig.tight_layout()
fig.savefig("figures/f04_uncertainty.png")

# %% Visualise
fx = np.array([np.min(temperature), np.max(temperature)])
fig, ax = plt.subplots(dpi=300)
for i in range(nreps):
    if i % 100 == 0:
        # ax.plot(
        #     fx, fx * sim_slope[i] + sim_intercept[i], c="xkcd:navy", alpha=0.1, lw=2
        # )
        ax.plot(fxl, np.polyval(sim_polyfit[i], fxl), c="xkcd:navy", alpha=0.1, lw=2)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("ln ($p$CO$_2$ / µatm)")
fig.tight_layout()
