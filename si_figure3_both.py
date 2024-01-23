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
use_quickload = True

# %% Determine settings
n_reps = int(1e6)  # use 1e6 for the manuscript --- takes about 10 minutes
opt_k_carbonic = 10
opt_total_borate = 1
fCO2 = t93.get_fCO2(opt_k_carbonic, opt_total_borate)
n_meas = len(fCO2)

# Define uncertainties
u_fCO2_random = 2  # µatm
u_fCO2_bias = 2  # µatm
u_temperature_random = 0.01  # °C
u_temperature_bias = 0.005  # °C

# Initialise simulated fields
sim_temperature = t93.temperature * np.ones((n_reps, 1))
sim_fCO2 = fCO2 * np.ones((n_reps, 1))

# Generate and add uncertainties (comment lines out to omit the corresponding terms)
sim_fCO2 += rng.normal(loc=0, scale=u_fCO2_random, size=(n_reps, n_meas))
sim_fCO2 += rng.normal(loc=0, scale=u_fCO2_bias, size=(n_reps, 1))
sim_temperature += rng.uniform(
    low=-u_temperature_random / 2, high=u_temperature_random / 2, size=(n_reps, n_meas)
)
sim_temperature += rng.normal(loc=0, scale=u_temperature_bias, size=(n_reps, 1))

# Make fits
sim_lnfCO2 = np.log(sim_fCO2)
if not use_quickload:
    fit_l_bc = np.full((n_reps, 2), np.nan)
    fit_q_abc = np.full((n_reps, 3), np.nan)
    fit_h_bc = np.full((n_reps, 2), np.nan)
    for i in range(n_reps):
        if i % 1000 == 0:
            print(i, "/", n_reps)
        fit_l_bc[i] = np.polyfit(sim_temperature[i], sim_lnfCO2[i], 1)
        fit_q_abc[i] = np.polyfit(sim_temperature[i], sim_lnfCO2[i], 2)
        fit_h_bc[i] = pwtools.fit_fCO2_vh(sim_temperature[i], sim_lnfCO2[i])["x"]
    np.save("quickload/fit_l_bc.npy", fit_l_bc)
    np.save("quickload/fit_q_abc.npy", fit_q_abc)
    np.save("quickload/fit_h_bc.npy", fit_h_bc)
else:
    fit_l_bc = np.load("quickload/fit_l_bc.npy")
    fit_q_abc = np.load("quickload/fit_q_abc.npy")
    fit_h_bc = np.load("quickload/fit_h_bc.npy")

# Get fit statistics
sim_bl = fit_l_bc[:, 0]
sim_aq_bq = fit_q_abc[:, :2].T
sim_bh = fit_h_bc[:, 0]
sim_var_bl = np.var(sim_bl)  # σ^2(b_l) for linear propagation
sim_cov_q_ab = np.cov(sim_aq_bq)  # Σ_q for quadratic propagation
sim_var_bh = np.var(sim_bh)  # σ^2(b_h) for van 't Hoff propagation

# %% Set conditions for a test set of propagations
t0 = 25  # °C; asserts were set up with 25 (other values may still work)
t1 = 20  # °C; asserts were set up with 20 (other values may still work)

# Propagate through to η and H --- simulations
sim_eta_l = pwtools.get_eta_l(sim_bl)
sim_eta_q = pwtools.get_eta_q(sim_aq_bq, t0)
sim_eta_h = pwtools.get_eta_h(sim_bh, t0)
sim_H_l = pwtools.get_H_l(sim_bl, t0, t1)
sim_H_q = pwtools.get_H_q(sim_aq_bq, t0, t1)
sim_H_h = pwtools.get_H_h(sim_bh, t0, t1)
var_eta_l__sim = np.var(sim_eta_l)
var_eta_q__sim = np.var(sim_eta_q)
var_eta_h__sim = np.var(sim_eta_h)
var_H_l__sim = np.var(sim_H_l)
var_H_q__sim = np.var(sim_H_q)
var_H_h__sim = np.var(sim_H_h)

# Propagate through to η and H --- direct equations from manuscript
var_eta_l__eq = pwtools.get_var_eta_l(sim_var_bl)
var_eta_q__eq = pwtools.get_var_eta_q(sim_cov_q_ab, t0)
var_eta_h__eq = pwtools.get_var_eta_h(sim_var_bh, t0)
var_H_l__eq = pwtools.get_var_H_l(sim_var_bl, t0, t1)
var_H_q__eq = pwtools.get_var_H_q(sim_cov_q_ab, t0, t1)
var_H_h__eq = pwtools.get_var_H_h(sim_var_bh, t0, t1)

# Propagate through to η and H --- autograd
var_eta_l__ag = pwtools.get_grad_eta_l(sim_bl.mean()) ** 2 * sim_var_bl
var_H_l__ag = pwtools.get_grad_H_l(sim_bl.mean(), t0, t1) ** 2 * sim_var_bl
jac_eta_q__ag = pwtools.get_jac_eta_q(sim_aq_bq.mean(axis=1), t0)
var_eta_q__ag = jac_eta_q__ag @ sim_cov_q_ab @ jac_eta_q__ag.T
jac_H_q_ag = pwtools.get_jac_H_q(sim_aq_bq.mean(axis=1), t0, t1)
var_H_q__ag = jac_H_q_ag @ sim_cov_q_ab @ jac_H_q_ag.T
var_eta_h__ag = pwtools.get_grad_eta_h(sim_bh.mean(), t0) ** 2 * sim_var_bh
var_H_h__ag = pwtools.get_grad_H_h(sim_bh.mean(), t0, t1) ** 2 * sim_var_bh

# Test that all three approaches give similar results (worst case is 0.01% agreement)
assert np.isclose(var_eta_l__eq, var_eta_l__ag, rtol=1e-16 / 100, atol=0)
assert np.isclose(var_eta_l__eq, var_eta_l__sim, rtol=1e-16 / 100, atol=0)
assert np.isclose(var_eta_q__eq, var_eta_q__ag, rtol=1e-12 / 100, atol=0)
assert np.isclose(var_eta_q__eq, var_eta_q__sim, rtol=0.01 / 100, atol=0)
assert np.isclose(var_eta_h__eq, var_eta_h__ag, rtol=1e-16 / 100, atol=0)
assert np.isclose(var_eta_h__eq, var_eta_h__sim, rtol=1e-12 / 100, atol=0)
assert np.isclose(var_H_l__eq, var_H_l__ag, rtol=1e-16, atol=0)
assert np.isclose(var_H_l__eq, var_H_l__sim, rtol=1e-12, atol=0)
assert np.isclose(var_H_q__eq, var_H_q__ag, rtol=1e-12, atol=0)
assert np.isclose(var_H_q__eq, var_H_q__sim, rtol=1e-4, atol=0)
assert np.isclose(var_H_h__eq, var_H_h__ag, rtol=1e-12, atol=0)
assert np.isclose(var_H_h__eq, var_H_h__sim, rtol=1e-12, atol=0)

# Print out uncertainties
print("σ(η_l)      =  {:.2f} / k°C".format(1e3 * np.sqrt(var_eta_l__sim)))
print("σ(b_q)      =  {:.2f} / k°C".format(1e3 * np.sqrt(sim_cov_q_ab[1, 1])))
print("σ(a_q)      =  {:.1f} / k°C**2".format(1e6 * np.sqrt(sim_cov_q_ab[0, 0])))
print("σ(a_q, b_q) = {:.0f}   / k°C**3".format(1e9 * sim_cov_q_ab[1, 0]))
print("σ(b_h)      = {:.1f} J/mol".format(np.sqrt(sim_var_bh)))

# %% Visualise
f_sim_cov_q_ab = sim_cov_q_ab.copy()
# f_sim_cov_q_ab[0, 1] = f_sim_cov_q_ab[1, 0] = 0  # see effect of ignoring covariance

f_t_soda = np.linspace(-1.8, 35.83)  # temperature range in OceanSODA
f_jac_eta_q = np.array([2 * f_t_soda, np.ones_like(f_t_soda)]).T
f_std_eta_q = np.sqrt(np.diag(f_jac_eta_q @ f_sim_cov_q_ab @ f_jac_eta_q.T))

f_dt = +1  # °C
f_t1_soda = f_t_soda + f_dt
f_std_H_l = np.sqrt(pwtools.get_var_H_l(sim_var_bl, f_t_soda, f_t1_soda))
f_std_H_q = np.sqrt(pwtools.get_var_H_q(sim_cov_q_ab, f_t_soda, f_t1_soda))
f_std_H_h = np.sqrt(pwtools.get_var_H_h(sim_var_bh, f_t_soda, f_t1_soda))
f_H_l = pwtools.get_H_l(sim_bl.mean(), f_t_soda, f_t1_soda)
f_H_q = pwtools.get_H_q(sim_aq_bq.mean(axis=1), f_t_soda, f_t1_soda)
f_H_h = pwtools.get_H_h(sim_bh.mean(), f_t_soda, f_t1_soda)
f_std_exp_H_l = np.exp(f_H_l) * f_std_H_l
f_std_exp_H_q = np.exp(f_H_q) * f_std_H_q
f_std_exp_H_h = np.exp(f_H_h) * f_std_H_h

fig, axs = plt.subplots(dpi=300, figsize=(12 / 2.54, 16 / 2.54), nrows=2)

ax = axs[0]
ax.text(0, 1.05, "(a)", transform=ax.transAxes)
ax.axhline(
    1e3 * np.sqrt(var_eta_l__sim),
    c=pwtools.dark,
    lw=1.5,
    # label="Linear fit",
)
ax.plot(
    f_t_soda,
    1e3 * f_std_eta_q,
    c=pwtools.dark,
    lw=2,
    ls=(0, (6, 2)),
    # label="Quadratic fit",
)
ax.plot(
    f_t_soda,
    1e3 * np.sqrt(pwtools.get_var_eta_h(sim_var_bh, f_t_soda)),
    c=pwtools.blue,
    lw=2,
    ls=(0, (3, 1)),
    # label="van 't Hoff fit",
)
ax.fill_betweenx(
    [0, 2],
    np.min(t93.temperature),
    np.max(t93.temperature),
    facecolor=pwtools.dark,
    alpha=0.2,
    # label="Ta93 $t$ range",
)
ax.set_ylabel("$σ(υ)$ / k°C$^{-1}$")
ax.set_yticks(np.arange(0, 3, 0.4))
ax.set_ylim([0, 2])

ax = axs[1]
ax.text(0, 1.05, "(b)", transform=ax.transAxes)
ax.plot(
    f_t_soda,
    100 * f_std_exp_H_l,
    c=pwtools.dark,
    lw=1.5,
    label="$υ_l$ (Ta93, linear)",
)
ax.plot(
    f_t_soda,
    100 * f_std_exp_H_q,
    c=pwtools.dark,
    lw=2,
    ls=(0, (6, 2)),
    label="$υ_q$ (Ta93, quadratic)",
)
ax.plot(
    f_t_soda,
    100 * f_std_exp_H_h,
    c=pwtools.blue,
    lw=2,
    ls=(0, (3, 1)),
    label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
)
ax.fill_betweenx(
    [0, 0.2],
    np.min(t93.temperature),
    np.max(t93.temperature),
    facecolor=pwtools.dark,
    alpha=0.3,
    label="Ta93 $t$ range",
)
ax.set_ylabel(r"$σ$[exp($Υ$)] for ∆$t$ = " + "${dt:+}$ °C / %".format(dt=f_dt))
ax.set_yticks(np.arange(0, 0.3, 0.04))
ax.set_ylim([0, 0.2])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, edgecolor="k")

for ax in axs:
    ax.set_xlabel("Temperature / °C")
    ax.set_xticks(np.arange(0, 40, 5))
    ax.set_xlim(f_t_soda[[0, -1]])
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures_si/figure3_both.png")
