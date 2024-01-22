from autograd import grad, jacobian
from scipy.optimize import least_squares, curve_fit
import numpy as np

Rgas = 8.314462618
tzero = 273.15
okc_codes = {
    1: "Ro93",
    2: "GP89",
    3: "DM87-H",
    4: "DM87-M",
    5: "DM87-HM",
    6: "Me73",
    7: "Me73-P",
    8: "Mi79",
    9: "CW98",
    10: "Lu00",
    11: "MM02",
    12: "Mi02",
    13: "Mi06",
    14: "Mi10",
    15: "Wa13",
    16: "Su20",
    17: "SB21",
    18: "Pa18",
}

bh_best = 28995  # J/mol
ch_best = 18.2420
bh_best_pCO2 = 28963  # J/mol
ch_best_pCO2 = 18.2321
bh_theory = 25288  # J/mol
ch_theory_best = 16.709
ch_theory_best_pCO2 = 16.713

thinspace = " "
f = "$ƒ$" + thinspace


def get_eta_l(bl):
    """Calculate linear υ with manuscript equation (8)."""
    return bl * 1


def get_eta_q(aq_bq, t):
    """Calculate quadratic υ with manuscript equation (9)."""
    aq, bq = aq_bq
    return 2 * aq * t + bq


def get_eta_h(bh, t):
    """Calculate van 't Hoff υ with manuscript equation (10)."""
    return bh / (Rgas * (t + tzero) ** 2)


def get_H_l(bl, t0, t1):
    """Calculate linear Υ with manuscript equation (11)."""
    return bl * (t1 - t0)


def get_H_q(aq_bq, t0, t1):
    """Calculate quadratic Υ with manuscript equation (12)."""
    aq, bq = aq_bq
    return aq * (t1**2 - t0**2) + bq * (t1 - t0)


def get_H_h(bh, t0, t1):
    """Calculate van 't Hoff Υ with manuscript equation (13)."""
    return (1 / (t0 + tzero) - 1 / (t1 + tzero)) * bh / Rgas


def get_var_eta_l(var_bl):
    """Calculate variance in linear υ with manuscript equation (14)."""
    return var_bl * 1


def get_var_H_l(var_bl, t0, t1):
    """Calculate variance in linear Υ with manuscript equation (15)."""
    return var_bl * (t1 - t0) ** 2


def get_var_eta_h(var_bh, t):
    """Calculate variance in van 't Hoff υ with manuscript equation (16)."""
    return var_bh / (Rgas * (t + tzero) ** 2) ** 2


def get_var_H_h(var_bh, t0, t1):
    """Calculate variance in van 't Hoff Υ with manuscript equation (17)."""
    return (1 / (t0 + tzero) - 1 / (t1 + tzero)) ** 2 * var_bh / Rgas**2


def get_var_eta_q(cov_q_ab, t):
    """Calculate variance in quadratic υ with manuscript equation (22)."""
    var_aq = cov_q_ab[0, 0]
    var_bq = cov_q_ab[1, 1]
    cov_ab = cov_q_ab[0, 1]
    assert cov_ab == cov_q_ab[1, 0]
    return 4 * t**2 * var_aq + var_bq + 4 * t * cov_ab


def get_var_H_q(cov_q_ab, t0, t1):
    """Calculate variance in quadratic Υ with manuscript equation (23)."""
    var_aq = cov_q_ab[0, 0]
    var_bq = cov_q_ab[1, 1]
    cov_ab = cov_q_ab[0, 1]
    assert cov_ab == cov_q_ab[1, 0]
    return (
        (t1**2 - t0**2) ** 2 * var_aq
        + (t1 - t0) ** 2 * var_bq
        + 2 * (t1**2 - t0**2) * (t1 - t0) * cov_ab
    )


def get_grad_eta_l(bl):
    return grad(get_eta_l)(bl)


def get_jac_eta_q(aq_bq, t):
    return jacobian(get_eta_q)(aq_bq, t)


def get_grad_eta_h(bh, t):
    return grad(get_eta_h)(bh, t)


def get_grad_H_l(bl, t0, t1):
    return grad(get_H_l)(bl, t0, t1)


def get_jac_H_q(aq_bq, t0, t1):
    return jacobian(get_H_q)(aq_bq, t0, t1)


def get_grad_H_h(bh, t0, t1):
    return grad(get_H_h)(bh, t0, t1)


def get_lnfCO2_vh(coeffs, temperature):
    b, c = coeffs
    return c - b / (Rgas * (temperature + tzero))


def _lsqfun_get_lnfCO2_vh(coeffs, temperature, ln_fCO2):
    return np.exp(get_lnfCO2_vh(coeffs, temperature)) - np.exp(ln_fCO2)
    # return get_lnfCO2_vh(coeffs, temperature) - ln_fCO2


def fit_fCO2_vh(temperature, ln_fCO2):
    return least_squares(
        _lsqfun_get_lnfCO2_vh, [25288, 20], args=(temperature, ln_fCO2)
    )


def _curfun_fit_vh(temperature, bh, ch):
    return np.exp(get_lnfCO2_vh((bh, ch), temperature))


def fit_vh_curve(temperature, fCO2):
    return curve_fit(_curfun_fit_vh, temperature, fCO2, p0=(25288, 20))


def get_lnfCO2_vht(c, temperature):
    return c - bh_theory / (Rgas * (temperature + tzero))


def _lsqfun_get_fCO2_vht(c, temperature, ln_fCO2):
    return np.exp(get_lnfCO2_vht(c, temperature)) - np.exp(ln_fCO2)


def fit_fCO2_vht(temperature, ln_fCO2):
    return least_squares(_lsqfun_get_fCO2_vht, [20], args=(temperature, ln_fCO2))
