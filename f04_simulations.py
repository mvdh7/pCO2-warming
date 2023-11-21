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
pCO2_bias = 2
temperature_precision = 0.05
temperature_bias = 0.1

nreps = 100
sim_pCO2 = results["pCO2"] * np.ones((nreps, 1))
# Add the random uncertainty in pCO2
sim_pCO2 += rng.normal(loc=0, scale=pCO2_precision, size=(nreps, len(pCO2)))
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_pCO2 += rng.normal(loc=0, scale=pCO2_bias, size=(nreps, 1))
sim_temperature = temperature * np.ones((nreps, 1))
# Add the random uncertainty in temperature
sim_temperature += rng.normal(
    loc=0, scale=temperature_precision, size=(nreps, len(pCO2))
)
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_temperature += rng.normal(loc=0, scale=temperature_bias, size=(nreps, 1))
sim_slope = np.full(nreps, np.nan)
sim_intercept = np.full(nreps, np.nan)
for i in range(nreps):
    i_lr = linregress(sim_temperature[i], np.log(sim_pCO2[i]))
    sim_slope[i] = i_lr.slope
    sim_intercept[i] = i_lr.intercept

# Get statistics
sim_slope_precision = sim_slope.std()
print(sim_slope_precision)

# Visualise
fig, ax = plt.subplots(dpi=300)
fx = np.array([np.min(temperature), np.max(temperature)])
for i in range(nreps):
    ax.plot(fx, fx * sim_slope[i] + sim_intercept[i], c="xkcd:navy", alpha=0.1, lw=2)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("ln ($p$CO$_2$ / µatm)")
fig.tight_layout()
