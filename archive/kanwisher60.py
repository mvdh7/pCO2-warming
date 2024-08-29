import numpy as np
from matplotlib import pyplot as plt

# import takahashi93 as t93
import pwtools

# Data extracted from Kanwisher (1960) Fig. 2 using plotdigitizer.com
temperature = np.array(
    [
        11.024958402662229,
        12.04991680532446,
        13.960066555740433,
        15.194675540765392,
        17.943427620632278,
        19.62063227953411,
        21.46089850249584,
    ]
)
pCO2 = np.array(
    [
        290.33674963396777,
        303.22108345534406,
        330.4538799414348,
        345.9736456808199,
        390.77598828696927,
        420.93704245973646,
        442.89897510980967,
    ]
)
fit = np.polyfit(temperature, np.log(pCO2), 1)

pCO2_fit = np.exp(fit[0] * temperature + fit[1])
fit_vh = pwtools.fit_vh_curve(temperature, np.log(pCO2))
pCO2_vh = np.exp(pwtools.get_lnfCO2_vh(fit_vh[0], temperature))

fx = np.linspace(np.min(temperature), np.max(temperature))
fy = np.exp(pwtools.get_lnfCO2_vh(fit_vh[0], fx))
fy_linear = fit[0] * fx + fit[1]

fig, ax = plt.subplots(dpi=300)
ax.scatter(temperature, pCO2 - pCO2_fit)
ax.plot(fx, fy - fy_linear)

# ax.scatter(t93.temperature, np.log(t93.pCO2))
# ax.plot(temperature, np.polyval(fit, temperature))
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("ln($p$CO$_2$ / µatm)")
# ax.set_ylim((-3, 3))


# Kanwisher data for Fig. 1 --- plot k60_temperature vs (k60_pCO2 - k60_norm)
k60_temperature = np.array(
    [
        11.024958402662229,
        12.04991680532446,
        13.960066555740433,
        15.194675540765392,
        17.943427620632278,
        19.62063227953411,
        21.46089850249584,
    ]
)
k60_pCO2 = np.array(
    [
        290.33674963396777,
        303.22108345534406,
        330.4538799414348,
        345.9736456808199,
        390.77598828696927,
        420.93704245973646,
        442.89897510980967,
    ]
)
k60_lnpCO2 = np.log(k60_pCO2)
k60_intercept = np.mean(k60_lnpCO2 - 0.0423 * k60_temperature)
k60_norm = np.exp(k60_temperature * 0.0423 + k60_intercept)
