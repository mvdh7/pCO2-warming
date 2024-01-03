import takahashi93 as t93
import numpy as np
from matplotlib import pyplot as plt
import pwtools

pf_linear = np.polyfit(t93.temperature, np.log(t93.pCO2), 1)
rmsd_linear = np.sqrt(
    np.mean((np.exp(np.polyval(pf_linear, t93.temperature)) - t93.pCO2) ** 2)
)
pf_quadratic = np.polyfit(t93.temperature, np.log(t93.pCO2), 2)
rmsd_quadratic = np.sqrt(
    np.mean((np.exp(np.polyval(pf_quadratic, t93.temperature)) - t93.pCO2) ** 2)
)

opt_result = pwtools.fit_pCO2_vh(t93.temperature, np.log(t93.pCO2))
rmsd_vh = np.sqrt(
    np.mean(
        (np.exp(pwtools.get_lnpCO2_vh(opt_result["x"], t93.temperature)) - t93.pCO2)
        ** 2
    )
)

optr_vht = pwtools.fit_pCO2_vht(t93.temperature, np.log(t93.pCO2))


ft = np.linspace(-1.8, 35.83)
fig, ax = plt.subplots(dpi=300)
ax.scatter(t93.temperature, np.log(t93.pCO2))
ax.plot(ft, np.polyval(pf_linear, ft))
ax.plot(ft, np.polyval(pf_quadratic, ft))
ax.plot(ft, pwtools.get_lnpCO2_vh(opt_result["x"], ft))

# %%
fig, ax = plt.subplots(dpi=300)
flin = np.exp(np.polyval(pf_linear, ft))
ax.scatter(t93.temperature, t93.pCO2 - np.exp(np.polyval(pf_linear, t93.temperature)))
ax.errorbar(
    t93.temperature,
    t93.pCO2 - np.exp(np.polyval(pf_linear, t93.temperature)),
    np.sqrt(8),
    ls="none",
)
ax.plot(ft, np.exp(np.polyval(pf_quadratic, ft)) - flin, label="quadratic")
ax.plot(ft, np.exp(pwtools.get_lnpCO2_vh(opt_result["x"], ft)) - flin, label="vH")
ax.plot(ft, np.exp(pwtools.get_lnpCO2_vht(optr_vht["x"], ft)) - flin, label="theory")
ax.plot(ft, np.exp(pwtools.get_lnpCO2_vht(16.713, ft)) - flin, label="manual")
ax.legend()

# ax.set_ylim([-20, 10])
