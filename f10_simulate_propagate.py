from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
from autograd import numpy as np, grad, jacobian
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
# pCO2_precision = 2  # from study - use value from above instead
pCO2_bias = 2  # guessed
temperature_precision = 0.005  # based on decimal places of reporting, uniform distro
temperature_bias = 0.005
# ^ WOCE-era accuracy from
# https://odv.awi.de/fileadmin/user_upload/odv/data/Gouretski-Koltermann-2004/ ...
# ... BSH35_report_final.pdf

nreps = 100_000
sim_pCO2 = results["pCO2"] * np.ones((nreps, 1))
# Add the random uncertainty in pCO2
sim_pCO2 += rng.normal(loc=0, scale=pCO2_precision, size=(nreps, len(pCO2)))
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_pCO2 += rng.normal(loc=0, scale=pCO2_bias, size=(nreps, 1))
sim_temperature = temperature * np.ones((nreps, 1))
# Add the random uncertainty in temperature (from decimal places)
sim_temperature += rng.uniform(
    low=-temperature_precision, high=temperature_precision, size=(nreps, len(pCO2))
)
# Add the systematic uncertainty in pCO2 (need to fix scale below)
sim_temperature += rng.normal(loc=0, scale=temperature_bias, size=(nreps, 1))
sim_slope = np.full(nreps, np.nan)
sim_intercept = np.full(nreps, np.nan)
sim_polyfit = []
for i in range(nreps):
    i_lr = linregress(sim_temperature[i], np.log(sim_pCO2[i]))
    sim_slope[i] = i_lr.slope
    sim_intercept[i] = i_lr.intercept
    sim_polyfit.append(np.polyfit(sim_temperature[i], np.log(sim_pCO2[i]), 2))
sim_polyfit = np.array(sim_polyfit)

# Get statistics
sim_slope_precision = sim_slope.std()
print("Linear slope precision:  {:.5f}".format(sim_slope_precision))
sim_poly_slope_precision = sim_polyfit[:, 1].std()
sim_poly_squared_precision = sim_polyfit[:, 0].std()
sim_poly_covmx = np.cov(sim_polyfit[:, :2].T)
# ^ this is for the squared then slope terms in the quadratic equation
print("Poly   slope precision:  {:.5f}".format(sim_poly_slope_precision))
print("Poly squared precision:  {:.5f}".format(sim_poly_squared_precision))
print("Poly        covariance: {:.2e}".format(sim_poly_covmx[0, 1]))

# # %% Polynomial uncertainty propagation
# fxl = np.linspace(np.min(temperature), np.max(temperature))
# jac_poly = np.array([fxl, np.ones_like(fxl)]).T
# uncert_poly_mx = jac_poly @ sim_poly_covmx @ jac_poly.T
# uncert_poly = np.sqrt(np.diag(uncert_poly_mx))

# fxl_soda = np.linspace(-1.8, 35.83)
# jac_poly_soda = np.array([fxl_soda, np.ones_like(fxl_soda)]).T
# uncert_poly_soda = np.sqrt(np.diag(jac_poly_soda @ sim_poly_covmx @ jac_poly_soda.T))

# # %% Visualise
# fx = np.array([np.min(temperature), np.max(temperature)])
# fig, ax = plt.subplots(dpi=300)
# for i in range(nreps):
#     if i % 100 == 0:
#         # ax.plot(
#         #     fx, fx * sim_slope[i] + sim_intercept[i], c="xkcd:navy", alpha=0.1, lw=2
#         # )
#         ax.plot(fxl, np.polyval(sim_polyfit[i], fxl), c="xkcd:navy", alpha=0.1, lw=2)
# ax.set_xlabel("Temperature / °C")
# ax.set_ylabel("ln ($p$CO$_2$ / µatm)")
# fig.tight_layout()
#
# # %% Propagate through to a correction
t0 = 1
t1 = 0
dt = t1 - t0


def get_linear(coeff, dt):
    return np.exp(coeff * dt)


def get_grad_linear(coeff, dt):
    return grad(get_linear)(coeff, dt)


sim_linear = get_linear(sim_slope, dt)
var_linear_sim = np.var(sim_linear)

fx_linear = np.exp(0.0423 * dt)
jac_linear = dt * fx_linear
jac_linear_auto = get_grad_linear(0.0423, dt)
assert np.isclose(jac_linear, jac_linear_auto)
var_linear = jac_linear**2 * sim_slope_precision**2


fx_quad = np.exp(0.0433 * dt - 4.35e-5 * (t1**2 - t0**2))


def get_quad(coeffs, t0, t1):
    squared, slope = coeffs
    return np.exp(slope * (t1 - t0) - squared * (t1**2 - t0**2))


def get_jac_quad(coeffs, t0, t1):
    return jacobian(get_quad)(coeffs, t0, t1)


sim_quad = get_quad(sim_polyfit[:, :2].T, t0, t1)
var_quad_sim = np.var(sim_quad)

jac_quad = np.array([[-(t1**2 - t0**2) * fx_quad, dt * fx_quad]])
jac_quad_auto = get_jac_quad(np.array([4.35e-5, 0.0433]), t0, t1)
assert np.allclose(jac_quad, jac_quad_auto)
var_quad = (jac_quad @ sim_poly_covmx @ jac_quad.T)[0][0]

print(np.sqrt(var_linear), np.sqrt(var_quad))
print(np.sqrt(var_linear_sim), np.sqrt(var_quad_sim))
# ^ these are the uncertainties in the term that gets multiplied by pCO2 to find the
# t-corrected pCO2
print(100 * np.sqrt(var_linear) / fx_linear, 100 * np.sqrt(var_quad) / fx_quad)

# For a cooling correction (t0 > t1):
# direct calculation slightly underestimates simulated error for linear
# direct calculation slightly  overestimates simulated error for quadratic
# due to curvature? error is bigger at higher dt and virtually disappears at small dt
# (small = e.g. 0.01 °C)
# see e.g. https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Caveats_and_warnings
# the propagation equation includes a truncated Taylor series expansion
# so that's the "problem" => can probably ignore

# %% Plot final uncertainties against temperature
u_t = np.linspace(-1.8, 35.83)
u_dt = 1
quad_coeffs = np.array([4.35e-5, 0.0433])
u_quad_jac = get_jac_quad(quad_coeffs, u_t, u_t + u_dt)
u_quad_covmx = u_quad_jac @ sim_poly_covmx @ u_quad_jac.T
u_quad_std = np.sqrt(np.diag(u_quad_covmx))
u_factor = get_quad(quad_coeffs, u_t, u_t + u_dt)
u_quad_pct = 100 * u_quad_std / u_factor

u_linear = np.exp(0.0423 * u_dt)
u_linear_grad = u_dt * u_linear
u_linear_var = u_linear_grad**2 * sim_slope_precision**2
u_linear_pct = 100 * np.sqrt(u_linear_var) / u_linear

fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 8 / 2.54))
ax.plot(u_t, u_quad_pct, label="Quadratic fit", c="xkcd:navy", lw=2)
ax.axhline(u_linear_pct, label="Linear fit", c="xkcd:navy", lw=2, ls=":")
ax.set_xlim(u_t[0], u_t[-1])
ax.set_ylim([0, 0.6])
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("Uncertainty in $p$CO$_2$ after\na {:+} °C correction / %".format(u_dt))
ax.legend()
fig.tight_layout()
fig.savefig("figures/f10_uncertainty_factor.png")
