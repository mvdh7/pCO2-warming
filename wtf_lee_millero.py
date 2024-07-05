from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.optimize import least_squares
import PyCO2SYS as pyco2
import pwtools
import takahashi93 as tak93


def rmsd(x):
    return np.sqrt(np.mean((x - fCO2) ** 2))


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

# Calculate fCO2 from DIC and alkalinity with PyCO2SYS
opt_k_carbonic = 1
r = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=temperature,
    opt_k_carbonic=opt_k_carbonic,
    opt_total_borate=1,
)
r_fCO2 = r["fCO2"]
r_rmsd = rmsd(r_fCO2)
print(
    "PyCO2SYS with opt_k_carbonic={} gives RMSD = {:.1f} µatm.\n".format(
        opt_k_carbonic, r_rmsd
    )
)

# Prepare for plotting
fx = np.linspace(np.min(temperature), np.max(temperature))
rx = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=fx,
    opt_k_carbonic=opt_k_carbonic,
    opt_total_borate=1,
)
ry = rx["fCO2"]

# Linear regression with own slope and intercept
linear_full = linregress(temperature, ln_fCO2)
linear_full_ln_fCO2 = linear_full.intercept + temperature * linear_full.slope
linear_full_fCO2 = np.exp(linear_full_ln_fCO2)
linear_full_rmsd = rmsd(linear_full_fCO2)
fy_full = np.exp(linear_full.intercept + fx * linear_full.slope)
print(
    "Refitting LM95's dataset, minimising ln(fCO2), returns υl = {:.2f}%/°C,".format(
        100 * linear_full.slope
    )
)
print("with an RMSD of {:.1f} µatm.\n".format(linear_full_rmsd))


def get_linear_LM95(coeffs):
    slope, intercept = coeffs
    return slope * temperature + intercept


def _lsqfun_get_linear_LM95(coeffs):
    return np.exp(get_linear_LM95(coeffs)) - fCO2


optresult_linear_LM95 = least_squares(_lsqfun_get_linear_LM95, [0, 0])
linear_LM95_fCO2 = np.exp(get_linear_LM95(optresult_linear_LM95["x"]))
linear_LM95_rmsd = rmsd(linear_LM95_fCO2)
fy_LM95 = np.exp(optresult_linear_LM95["x"][0] * fx + optresult_linear_LM95["x"][1])
print(
    "Refitting LM95's dataset, but minimising fCO2, returns υl = {:.2f}%/°C,".format(
        100 * optresult_linear_LM95["x"][0]
    )
)
print("with an RMSD of {:.1f} µatm.\n".format(linear_LM95_rmsd))


# Linear regression with Ta93 slope and own intercept
def get_linear_T93(intercept):
    return 0.0423 * temperature + intercept


def _lsqfun_get_linear_T93(intercept):
    return np.exp(get_linear_T93(intercept)) - fCO2


optresult_linear_T93 = least_squares(_lsqfun_get_linear_T93, 0)
linear_T93_intercept = optresult_linear_T93["x"][0]
linear_T93_fCO2 = np.exp(linear_T93_intercept + temperature * 0.0423)
linear_T93_rmsd = rmsd(linear_T93_fCO2)
fy_T93 = np.exp(linear_T93_intercept + fx * 0.0423)
print("That's virtually identical to using the T93 value of 4.23%/°C directly.\n")

curve_vh = pwtools.fit_vh_curve(temperature, fCO2)
vh_fCO2 = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], temperature))
vh_rmsd = rmsd(vh_fCO2)
fy_vh = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], fx))
print("Fitting the 1/tK curve gives a bh of {:.0f} J/mol,".format(curve_vh[0][0]))
print("with and RMSD of {:.1f} µatm.\n".format(vh_rmsd))

# fy = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], fx))
# fy_linear = np.exp(lr.intercept + fx * lr.slope)


def get_quadratic(qint):
    return (
        tak93.quadratic[0] * temperature**2 + tak93.quadratic[1] * temperature + qint
    )


def _lsqfun_get_quadratic(qint):
    return np.exp(get_quadratic(qint)) - fCO2


optr_qint = least_squares(_lsqfun_get_quadratic, 0)
qint = optr_qint["x"][0]
qint_fCO2 = np.exp(
    tak93.quadratic[0] * temperature**2 + tak93.quadratic[1] * temperature + qint
)
qint_rmsd = rmsd(qint_fCO2)
fy_qint = np.exp(tak93.quadratic[0] * fx**2 + tak93.quadratic[1] * fx + qint)
print(
    "The T93 quadratic fit has an RMSD of {:.1f} µatm for the LM95 data.".format(
        qint_rmsd
    )
)

qfit = np.polyfit(temperature, ln_fCO2, 2)
qfit_fCO2 = np.exp(np.polyval(qfit, temperature))
qfit_rmsd = rmsd(qfit_fCO2)
fy_qfit = np.exp(np.polyval(qfit, fx))
print(
    "Fitting a new quadratic fit to LM95 data gives RMSD = {:.1f} µatm.\n".format(
        qfit_rmsd
    )
)

# Which one to normalise everything to
fnorm = fy_full
fnorm_fCO2 = linear_full_fCO2

fig, ax = plt.subplots(dpi=300)
ax.scatter(temperature, fCO2 - fnorm_fCO2)
# ax.plot(fx, fy_full - fnorm, label="LM95 refit to ln(fCO2)")
ax.plot(fx, fy_T93 - fnorm, label="T93 linear")
ax.plot(fx, fy_LM95 - fnorm, label="LM95 refit to fCO2")
ax.plot(fx, fy_vh - fnorm, label="bh refit to LM95")
ax.plot(fx, fy_qint - fnorm, label="T93 quadratic")
ax.plot(fx, fy_qfit - fnorm, label="LM95 quadratic")
ax.plot(fx, ry - fnorm, label="PyCO2SYS okc={}".format(opt_k_carbonic))
ax.axhline(0, c="k", lw=0.8)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(
    "[{sp}{f}CO$_2$ $-$ {f}CO$_2$($υ_l$)] / µatm".format(
        sp=pwtools.thinspace, f=pwtools.f
    )
)
ax.set_ylim((-12, 12))
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
