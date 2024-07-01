import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.optimize import least_squares
import PyCO2SYS as pyco2
import pwtools
import takahashi93 as tak93

# Data from Lee and Millero (1995) Table 6
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
alkalinity = 2320.8
dic = 1966.8
salinity = 34.95

ln_fCO2 = np.log(fCO2)

r = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=temperature,
    opt_k_carbonic=1,
    opt_total_borate=1,
)
r_fCO2 = r["fCO2"]

# Linear regression with own slope and intercept
lr = linregress(temperature, ln_fCO2)
lr_ln_fCO2 = lr.intercept + temperature * lr.slope
lr_fCO2 = np.exp(lr_ln_fCO2)


# Linear regression with Ta93 slope and own intercept
def get_linear(intercept):
    return 0.0423 * temperature + intercept


def _lsqfun_get_linear(intercept):
    return get_linear(intercept) - ln_fCO2


optresult = least_squares(_lsqfun_get_linear, 0)
linear_intercept = optresult["x"][0]
linear_fCO2 = np.exp(linear_intercept + temperature * 0.0423)

curve_vh = pwtools.fit_vh_curve(temperature, fCO2)
fx = np.linspace(np.min(temperature), np.max(temperature))
fy = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], fx))
fy_linear = np.exp(lr.intercept + fx * lr.slope)

vh_fCO2 = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], temperature))


def get_quadratic(qint):
    return (
        tak93.quadratic[0] * temperature**2 + tak93.quadratic[1] * temperature + qint
    )


def _lsqfun_get_quadratic(qint):
    return get_quadratic(qint) - ln_fCO2


optr_qint = least_squares(_lsqfun_get_quadratic, 0)
qint = optr_qint["x"][0]

fy_quadratic = np.exp(tak93.quadratic[0] * fx**2 + tak93.quadratic[1] * fx + qint)

qfit = np.polyfit(temperature, ln_fCO2, 2)
fy_quadratic_here = np.exp(np.polyval(qfit, fx))

fy_t93bh = np.exp(pwtools.get_lnfCO2_vh((pwtools.bh_best, pwtools.ch_best), fx))

fig, ax = plt.subplots(dpi=300)
# ax.scatter(temperature, ln_fCO2)
# ax.scatter(temperature, lr_ln_fCO2)
ax.scatter(temperature, fCO2 - lr_fCO2)
ax.scatter(temperature, r_fCO2 - lr_fCO2)
ax.plot(fx, fy - fy_linear, label="bh fit to this dataset")
ax.plot(fx, fy_quadratic - fy_linear, label="q fit with T93 coeffs")
ax.plot(fx, fy_quadratic_here - fy_linear, label="q fit to this dataset")
# ax.scatter(temperature, fCO2 - linear_fCO2)
ax.axhline(0, c="k", lw=0.8)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(
    "[{sp}{f}CO$_2$ $-$ {f}CO$_2$($υ_l$)] / µatm".format(
        sp=pwtools.thinspace, f=pwtools.f
    )
)
ax.set_ylim((-10, 10))
rmsd_linear = np.sqrt(np.mean((fCO2 - lr_fCO2) ** 2))
rmsd_vh = np.sqrt(np.mean((fCO2 - vh_fCO2) ** 2))
ax.legend()

# %% WT correction
xCO2_headspace = fCO2 * 1e-6
xCO2_standard = 340e-6
v_water = 460
v_air = 120
pressure = 101325  # Pa
R = 8.3145  # J / (K * mol)

del_dic = (
    (xCO2_headspace - xCO2_standard)
    * pressure
    * v_air
    / (R * (temperature + 273.15) * v_water)
) * 1e3  # µmol/l

results_wtcorr = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    temperature=temperature,
    salinity=salinity,
    opt_k_carbonic=1,
)
results_wtcorr_del = pyco2.sys(
    par1=alkalinity,
    par2=dic + del_dic,
    par1_type=1,
    par2_type=2,
    temperature=temperature,
    salinity=salinity,
    opt_k_carbonic=1,
)

r_wtcorr_fCO2 = results_wtcorr["fCO2"]
r_wtcorr_dd_fCO2 = results_wtcorr_del["fCO2"]
del_fCO2 = r_wtcorr_fCO2 - r_wtcorr_dd_fCO2
