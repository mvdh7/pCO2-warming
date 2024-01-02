from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

from matplotlib import pyplot as plt
import numpy as np
import takahashi93 as t93
import pwtools

# Initialise random number generator
rng = np.random.default_rng(1)

# Determine settings
n_reps = int(1e6)  # use 1e6 for the manuscript
n_meas = len(t93.pCO2)
opt_k_carbonic = 10
opt_total_borate = 1

# Define uncertainties
u_pCO2_random = 2  # µatm
u_pCO2_bias = 2  # µatm
u_temperature_random = 0.01  # °C
u_temperature_bias = 0.005  # °C

# Initialise simulated fields ----------------------------------------- NEEDS UPDATING?
sim_pCO2 = t93.pCO2 * np.ones((n_reps, 1))
sim_temperature = t93.temperature * np.ones((n_reps, 1))

# Generate and add uncertainties (comment lines out to omit the corresponding terms)
sim_pCO2 += rng.normal(loc=0, scale=u_pCO2_random, size=(n_reps, n_meas))
sim_pCO2 += rng.normal(loc=0, scale=u_pCO2_bias, size=(n_reps, 1))
sim_temperature += rng.uniform(
    low=-u_temperature_random / 2, high=u_temperature_random / 2, size=(n_reps, n_meas)
)
sim_temperature += rng.normal(loc=0, scale=u_temperature_bias, size=(n_reps, 1))

# Make fits
sim_lnpCO2 = np.log(sim_pCO2)
fit_l_bc = np.full((n_reps, 2), np.nan)
fit_q_abc = np.full((n_reps, 3), np.nan)
for i in range(n_reps):
    fit_l_bc[i] = np.polyfit(sim_temperature[i], sim_lnpCO2[i], 1)
    fit_q_abc[i] = np.polyfit(sim_temperature[i], sim_lnpCO2[i], 2)

# Get fit statistics
sim_bl = fit_l_bc[:, 0]
sim_aq_bq = fit_q_abc[:, :2].T
sim_var_bl = np.var(sim_bl)  # σ^2(b_l) for linear propagation
sim_cov_q_ab = np.cov(sim_aq_bq)  # Σ_q for quadratic propagation

# %% Set conditions for a test set of propagations
t0 = 25  # °C; asserts were set up with 25 (other values may still work)
t1 = 20  # °C; asserts were set up with 20 (other values may still work)

# Propagate through to η and H --- simulations
sim_eta_l = pwtools.get_eta_l(sim_bl)
sim_eta_q = pwtools.get_eta_q(sim_aq_bq, t0)
sim_H_l = pwtools.get_H_l(sim_bl, t0, t1)
sim_H_q = pwtools.get_H_q(sim_aq_bq, t0, t1)
var_eta_l__sim = np.var(sim_eta_l)
var_eta_q__sim = np.var(sim_eta_q)
var_H_l__sim = np.var(sim_H_l)
var_H_q__sim = np.var(sim_H_q)

# Propagate through to η and H --- direct equations from manuscript
var_eta_l__eq = pwtools.get_var_eta_l(sim_var_bl)
var_eta_q__eq = pwtools.get_var_eta_q(sim_cov_q_ab, t0)
var_H_l__eq = pwtools.get_var_H_l(sim_var_bl, t0, t1)
var_H_q__eq = pwtools.get_var_H_q(sim_cov_q_ab, t0, t1)

# Propagate through to η and H --- autograd
var_eta_l__ag = pwtools.get_grad_eta_l(sim_bl.mean()) ** 2 * sim_var_bl
var_H_l__ag = pwtools.get_grad_H_l(sim_bl.mean(), t0, t1) ** 2 * sim_var_bl
jac_eta_q__ag = pwtools.get_jac_eta_q(sim_aq_bq.mean(axis=1), t0)
var_eta_q__ag = jac_eta_q__ag @ sim_cov_q_ab @ jac_eta_q__ag.T
jac_H_q_ag = pwtools.get_jac_H_q(sim_aq_bq.mean(axis=1), t0, t1)
var_H_q__ag = jac_H_q_ag @ sim_cov_q_ab @ jac_H_q_ag.T

# Test that all three approaches give similar results (worst case is 0.01% agreement)
assert np.isclose(var_eta_l__eq, var_eta_l__ag, rtol=1e-16 / 100, atol=0)
assert np.isclose(var_eta_l__eq, var_eta_l__sim, rtol=1e-16 / 100, atol=0)
assert np.isclose(var_eta_q__eq, var_eta_q__ag, rtol=1e-12 / 100, atol=0)
assert np.isclose(var_eta_q__eq, var_eta_q__sim, rtol=0.01 / 100, atol=0)
assert np.isclose(var_H_l__eq, var_H_l__ag, rtol=1e-16, atol=0)
assert np.isclose(var_H_l__eq, var_H_l__sim, rtol=1e-12, atol=0)
assert np.isclose(var_H_q__eq, var_H_q__ag, rtol=1e-12, atol=0)
assert np.isclose(var_H_q__eq, var_H_q__sim, rtol=1e-4, atol=0)

# Print out uncertainties
print("σ(η_l)      =  {:.2f} / k°C".format(1e3 * np.sqrt(var_eta_l__sim)))
print("σ(b_q)      =  {:.2f} / k°C".format(1e3 * np.sqrt(sim_cov_q_ab[1, 1])))
print("σ(a_q)      =  {:.1f} / k°C**2".format(1e6 * np.sqrt(sim_cov_q_ab[0, 0])))
print("σ(a_q, b_q) = {:.0f}   / k°C**3".format(1e9 * sim_cov_q_ab[1, 0]))

# %% Visualise
f_sim_cov_q_ab = sim_cov_q_ab.copy()
# f_sim_cov_q_ab[0, 1] = f_sim_cov_q_ab[1, 0] = 0  # see effect of ignoring covariance

f_t_soda = np.linspace(-1.8, 35.83)  # temperature range in OceanSODA
f_jac_eta_q = np.array([2 * f_t_soda, np.ones_like(f_t_soda)]).T
f_std_eta_q = np.sqrt(np.diag(f_jac_eta_q @ f_sim_cov_q_ab @ f_jac_eta_q.T))

f_dt = +1
f_t1_soda = f_t_soda + f_dt
f_std_H_l = np.sqrt(sim_var_bl) * np.abs(f_dt)
f_jac_H_q = np.array([f_t1_soda**2 - f_t_soda**2, f_t1_soda - f_t_soda]).T
f_std_H_q = np.sqrt(np.diag(f_jac_H_q @ f_sim_cov_q_ab @ f_jac_H_q.T))

f_H_l = pwtools.get_H_l(sim_bl.mean(), f_t_soda, f_t1_soda)
f_H_q = pwtools.get_H_q(sim_aq_bq.mean(axis=1), f_t_soda, f_t1_soda)
f_std_exp_H_l = np.exp(f_H_l) * f_std_H_l
f_std_exp_H_q = np.exp(f_H_q) * f_std_H_q

fig, axs = plt.subplots(dpi=300, figsize=(12 / 2.54, 16 / 2.54), nrows=2)

ax = axs[0]
ax.text(0, 1.05, "(a)", transform=ax.transAxes)
ax.plot(f_t_soda, 1e3 * f_std_eta_q, c="xkcd:navy", lw=2, label="Quadratic fit")
ax.axhline(
    1e3 * np.sqrt(var_eta_l__sim), c="xkcd:navy", lw=2, ls=":", label="Linear fit"
)
ax.fill_betweenx(
    [0, 2],
    np.min(t93.temperature),
    np.max(t93.temperature),
    facecolor="xkcd:navy",
    alpha=0.2,
    label="Ta93 $t$ range",
)
ax.set_ylabel("$σ(η)$ / k°C$^{-1}$")
ax.set_yticks(np.arange(0, 3, 0.4))
ax.set_ylim([0, 2])
ax.legend(loc="upper left")

ax = axs[1]
ax.text(0, 1.05, "(b)", transform=ax.transAxes)
ax.plot(f_t_soda, 100 * f_std_exp_H_q, c="xkcd:navy", lw=2, label="Quadratic fit")
ax.plot(f_t_soda, 100 * f_std_exp_H_l, c="xkcd:navy", lw=2, ls=":", label="Linear fit")
ax.fill_betweenx(
    [0, 0.2],
    np.min(t93.temperature),
    np.max(t93.temperature),
    facecolor="xkcd:navy",
    alpha=0.2,
    label="Ta93 $t$ range",
)
ax.set_ylabel("$σ$($p$CO$_2$) for ∆t = ${:+}$ °C / %".format(f_dt))
ax.set_yticks(np.arange(0, 0.3, 0.04))
ax.set_ylim([0, 0.2])

for ax in axs:
    ax.set_xlabel("Temperature / °C")
    ax.set_xticks(np.arange(0, 40, 5))
    ax.set_xlim(f_t_soda[[0, -1]])
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/simulate.png")
