import numpy as np
from matplotlib import pyplot as plt

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

fig, ax = plt.subplots(dpi=300)
ax.scatter(temperature, np.log(pCO2))
ax.plot(temperature, np.polyval(fit, temperature))
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("ln($p$CO$_2$ / µatm)")
