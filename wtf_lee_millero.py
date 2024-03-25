import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import PyCO2SYS as pyco2

temperature = np.array(
    [
        5.05,
        10,
        14.98,
        20,
        25,
        30.04,
        35,
    ]
)

fCO2 = np.array(
    [
        140,
        175,
        218,
        270,
        329,
        410,
        504,
    ]
)

ln_fCO2 = np.log(fCO2)

r = pyco2.sys(
    par1=2320.8,
    par2=1966.8,  # - del_dic,
    par1_type=1,
    par2_type=2,
    salinity=34.95,
    temperature=temperature,
    opt_k_carbonic=1,
)
r_fCO2 = r["fCO2"]

lr = linregress(temperature, ln_fCO2)
lr_ln_fCO2 = lr.intercept + temperature * lr.slope
lr_fCO2 = np.exp(lr_ln_fCO2)

fig, ax = plt.subplots(dpi=300)
# ax.scatter(temperature, ln_fCO2)
# ax.scatter(temperature, lr_ln_fCO2)
ax.scatter(temperature, fCO2 - lr_fCO2)
ax.scatter(temperature, r_fCO2 - lr_fCO2)
ax.axhline(0, c="k", lw=0.8)

# %% WT correction
xCO2_headspace = fCO2.copy() * 1e-6 * 101.325
xCO2_standard = 340e-6 * 101.325
v_water = 500
v_air = 50  # complete guess!
pressure = 101.325  # kPa
R = 8.3145  # J / (K * mol)

del_dic = (
    (xCO2_headspace - xCO2_standard)
    * pressure
    * v_air
    / (R * (temperature + 273.15) * v_water)
) * 1e6
