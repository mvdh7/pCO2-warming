from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from matplotlib import pyplot as plt

# Humphreys et al. (2018, Mar. Chem.), Appendix C, equation C.1:
#   pCO2 = (b**2 / c) * K2 / (K0 * K1)
# And (still following H18) we can approximate the following:
#   b = 2TC - AT
#   c = AT - TC
# both of which are constants (if AT and TC are constant)!
# Which means that pCO2 is controlled by K2 / (K0 * K1).
# So how does K2 / (K0 * K1) respond to temperature?
# First we'll just look at it with PyCO2SYS, then take a theoretical approach with the
# van 't Hoff equation.

opt_k_carbonic = 10

temperature = np.linspace(-1.8, 35.83)
alkalinity = 2250
dic = 2150
results = pyco2.sys(
    par1=alkalinity,
    par1_type=1,
    par2=dic,
    par2_type=2,
    temperature=temperature,
    opt_k_carbonic=opt_k_carbonic,
    opt_total_borate=1,
)
pCO2 = results["pCO2"]
K0 = results["k_CO2"]
K1 = results["k_carbonic_1"]
K2 = results["k_carbonic_2"]
eta = results["dlnpCO2_dT"]
Rgas = results["gas_constant"] / 10
Kfrac = K2 / (K0 * K1)
b = 2 * dic - alkalinity
c = alkalinity - dic
multiplier = b**2 / c

# Standard enthalpies of formation
ef_CO2_g = -393.476
ef_CO2_aq = -413.196
ef_H2O = -285.800
ef_HCO3 = -689.862
ef_CO3 = -675.160
ef_H = 0

# Standard enthalpies of reaction
er_K0 = ef_CO2_aq - ef_CO2_g
er_K1 = ef_HCO3 + ef_H - (ef_CO2_aq + ef_H2O)
er_K2 = ef_CO3 + ef_H - ef_HCO3

enthalpy_sum = (er_K2 - er_K0 - er_K1) * 1e3  # J/mol
print("Enthalpy sum = {:.1f}".format(enthalpy_sum))
dlnKfrac_dT = enthalpy_sum / (Rgas * (273.15 + temperature) ** 2)

vanthoff_centre = 0
tdiff = np.diff(temperature)[0]
vanthoff = np.cumsum(tdiff * dlnKfrac_dT) + np.log(pCO2[vanthoff_centre])

fig, axs = plt.subplots(dpi=300, figsize=(12 / 2.54, 16 / 2.54), nrows=2)
ax = axs[0]
ax.plot(temperature, np.log(pCO2), label="$p$CO$_2$")
ax.plot(temperature, np.log(Kfrac * multiplier), label="$K$ equation")
ax.plot(temperature, vanthoff, label="van 't Hoff")
ax.legend()
# ax.set_ylim([5.5, 8])
ax = axs[1]
ax.plot(temperature, eta, label="$η$ from PyCO2SYS")
ax.plot(temperature, dlnKfrac_dT, label="van 't Hoff")
ax.legend()
