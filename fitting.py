import takahashi93 as t93
import numpy as np
from matplotlib import pyplot as plt
import pwtools

temperature = t93.temperature
fCO2 = t93.get_fCO2(10, 1)

# # Data from Lee and Millero (1995) Table 6
# temperature = np.array(
#     [
#         5.05,
#         10,
#         14.98,
#         20,
#         25,
#         30.04,
#         35,
#     ]
# )
# fCO2 = np.array(
#     [
#         140,
#         175,
#         218,
#         270,
#         329,
#         410,
#         504,
#     ]
# )

pf_linear = np.polyfit(temperature, np.log(fCO2), 1)
rmsd_linear = np.sqrt(np.mean((np.exp(np.polyval(pf_linear, temperature)) - fCO2) ** 2))

pf_quadratic = np.polyfit(temperature, np.log(fCO2), 2)
rmsd_quadratic = np.sqrt(
    np.mean((np.exp(np.polyval(pf_quadratic, temperature)) - fCO2) ** 2)
)

fit_vh = pwtools.fit_fCO2_vh(temperature, np.log(fCO2))
fit_vh_curve = pwtools.fit_vh_curve(temperature, fCO2)
# fit_vh_pCO2 = pwtools.fit_fCO2_vh(temperature, np.log(t93.pCO2))
# rmsd_vh = np.sqrt(
#     np.mean(
#         (np.exp(pwtools.get_lnfCO2_vh(fit_vh["x"], temperature)) - t93.pCO2) ** 2
#     )
# )
bh = fit_vh["x"][0]
bh_std = np.sqrt(fit_vh_curve[1][0][0])

fit_vht = pwtools.fit_fCO2_vht(temperature, np.log(fCO2))
fit_vht_pCO2 = pwtools.fit_fCO2_vht(temperature, np.log(t93.pCO2))

# Just a 1:1 plot
ft = np.linspace(-1.8, 35.83)
fig, ax = plt.subplots(dpi=300)
ax.scatter(temperature, np.log(t93.pCO2))
ax.plot(ft, np.polyval(pf_linear, ft))
ax.plot(ft, np.polyval(pf_quadratic, ft))
ax.plot(ft, pwtools.get_lnfCO2_vh(fit_vh["x"], ft))

# Check the fits look okay and to make sure that we haven't rounded off too many decimal
# places (compare the 'exact' with 'manual' lines - should be indistinguishable)
fig, ax = plt.subplots(dpi=300)
flin = np.exp(np.polyval(pf_linear, ft))
ax.scatter(temperature, t93.pCO2 - np.exp(np.polyval(pf_linear, temperature)))
ax.errorbar(
    temperature,
    t93.pCO2 - np.exp(np.polyval(pf_linear, temperature)),
    np.sqrt(8),
    ls="none",
)
ax.plot(ft, np.exp(np.polyval(pf_quadratic, ft)) - flin, label="quadratic")
ax.plot(ft, np.exp(pwtools.get_lnfCO2_vh(fit_vh["x"], ft)) - flin, label="vH - exact")
ax.plot(
    ft, np.exp(pwtools.get_lnfCO2_vh([28995, 18.242], ft)) - flin, label="vH - manual"
)
ax.plot(
    ft, np.exp(pwtools.get_lnfCO2_vht(fit_vht["x"], ft)) - flin, label="theory - exact"
)
ax.plot(ft, np.exp(pwtools.get_lnfCO2_vht(16.709, ft)) - flin, label="theory - manual")
ax.legend()
