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
lm95 = dict(
    salinity=salinity,
    total_silicate=0,
    total_phosphate=0,
)


def get_alkalinity(opt_k_carbonic, opt_total_borate):
    """Determine alkalinity in the experiment as the best-fitting alkalinity to match
    all the experimental fCO2 points.
    """

    def fCO2_from_alkalinity(alkalinity):
        return pyco2.sys(
            par1=dic,
            par1_type=2,
            par2=alkalinity,
            par2_type=1,
            temperature=temperature,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
            **lm95,
        )["fCO2"]

    def _lsqfun_fCO2_from_alkalinity(alkalinity, fCO2):
        return fCO2_from_alkalinity(alkalinity) - fCO2

    opt_result = least_squares(_lsqfun_fCO2_from_alkalinity, [2300], args=(fCO2,))
    return opt_result["x"][0], np.sqrt(np.mean(opt_result.fun**2))


alkalinity_fitted = get_alkalinity(10, 1)
# alkalinity = alkalinity_fitted[0]

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

# Calculate with Lueker et al. (2000)
r10 = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=temperature,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
r10_fCO2 = r10["fCO2"]
r10_rmsd = rmsd(r10_fCO2)

# Prepare for plotting
fx = np.linspace(0, 50, num=500)
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

rx10 = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=fx,
    opt_k_carbonic=10,
    opt_total_borate=1,
)
ry10 = rx10["fCO2"]

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

# Find bh based on the DIC and alkalinity reported
ex_temperature = np.linspace(-1.8, 35.83, num=50)
ex_results = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    temperature=ex_temperature,
    **lm95,
)
ex_fCO2 = ex_results["fCO2"]
fit_bh, fit_ch = pwtools.fit_vh_curve(ex_temperature, ex_fCO2)[0]

curve_vh = pwtools.fit_vh_curve(temperature, fCO2)
vh_fCO2 = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], temperature))
vh_rmsd = rmsd(vh_fCO2)
fy_vh = np.exp(pwtools.get_lnfCO2_vh(curve_vh[0], fx))
print("Fitting the 1/tK curve gives a bh of {:.0f} J/mol,".format(curve_vh[0][0]))
print("with an RMSD of {:.1f} µatm.\n".format(vh_rmsd))
print("Based on the DIC and alkalinity reported by LM95, with K1/K2 of Lu00,")
print("we would have expected a bh of {:.0f} J/mol.\n".format(fit_bh))

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

style_linear = dict(
    c=pwtools.dark,
    label="$υ_l$ (Ta93, linear)",
    # alpha=0.9,
    lw=1.5,
)
style_poly = dict(
    c=pwtools.dark,
    label="$υ_q$ (Ta93, quadratic)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (6, 2)),
    zorder=2,
)
style_pyco2_1 = dict(
    c=pwtools.pink,
    label=r"$υ_\mathrm{Ro93}$ (PyCO2SYS, Ro93)",
    # alpha=0.8,
    lw=1.5,
    ls="-",
    zorder=3,
)
style_pyco2_10 = dict(
    c=pwtools.pink,
    label=r"$υ_\mathrm{Lu00}$ (PyCO2SYS, Lu00)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (2,)),
    zorder=3,
)
style_vh = dict(
    c=pwtools.green,
    label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (3, 1)),
    zorder=1,
)
style_linear_LM95 = dict(
    c=pwtools.green,
    label="$υ_l$ (LM95, linear refitted)",
    # alpha=0.9,
    ls=(0, (1, 2)),
    lw=2,
)
style_poly_LM95 = dict(
    c=pwtools.green,
    label="$υ_q$ (LM95, quadratic)",
    # alpha=0.8,
    lw=1.5,
    ls=(0, (6, 2)),
    zorder=2,
)


fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 14 / 2.54))
ax.errorbar(
    temperature,
    fCO2 - fnorm_fCO2,
    3,
    c=pwtools.dark,
    ls="none",
    zorder=-1,
)
ax.scatter(
    temperature,
    fCO2 - fnorm_fCO2,
    c=pwtools.dark,
    # label="LM95 measured",
    s=50,
    zorder=0,
    # alpha=0.9,
    edgecolor="none",
)
# ax.scatter(temperature, fCO2 - fnorm_fCO2)
# ax.plot(fx, fy_full - fnorm, label="LM95 refit to ln(fCO2)")
ax.plot(fx, fy_T93 - fnorm, **style_linear)
ax.plot(fx, fy_qint - fnorm, **style_poly)
ax.plot(fx, ry - fnorm, **style_pyco2_1)
ax.plot(fx, ry10 - fnorm, **style_pyco2_10)
ax.axhline(0, c="k", lw=0.8)
ax.plot(fx, fy_LM95 - fnorm, **style_linear_LM95)
ax.plot(fx, fy_qfit - fnorm, **style_poly_LM95)
ax.plot(fx, fy_vh - fnorm, **style_vh)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel(
    "[{sp}{f}CO$_2$ $-$ {f}CO$_2$($υ_l$)] / µatm".format(
        sp=pwtools.thinspace, f=pwtools.f
    )
)
ax.set_xlim(3.5, 36.5)
ax.set_ylim((-22, 12))
ax.legend(
    loc="upper center", bbox_to_anchor=(0.42, -0.15), edgecolor="k", ncol=2, fontsize=9
)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("figures_final/extra.png")

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
