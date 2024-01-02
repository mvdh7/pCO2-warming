from autograd import grad, jacobian

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


def get_eta_l(bl):
    """Calculate linear η with manuscript equation (7)."""
    return bl * 1


def get_eta_q(aq_bq, t):
    """Calculate quadratic η with manuscript equation (8)."""
    aq, bq = aq_bq
    return 2 * aq * t + bq


def get_H_l(bl, t0, t1):
    """Calculate linear H with manuscript equation (9)."""
    return bl * (t1 - t0)


def get_H_q(aq_bq, t0, t1):
    """Calculate quadratic H with manuscript equation (10)."""
    aq, bq = aq_bq
    return aq * (t1**2 - t0**2) + bq * (t1 - t0)


def get_var_eta_l(var_bl):
    """Calculate variance in linear η with manuscript equation (11)."""
    return var_bl * 1


def get_var_eta_q(cov_q_ab, t):
    """Calculate variance in quadratic η with manuscript equation (17)."""
    var_aq = cov_q_ab[0, 0]
    var_bq = cov_q_ab[1, 1]
    cov_ab = cov_q_ab[0, 1]
    assert cov_ab == cov_q_ab[1, 0]
    return 4 * t**2 * var_aq + var_bq + 4 * t * cov_ab


def get_var_H_l(var_bl, t0, t1):
    """Calculate variance in linear H with manuscript equation (12)."""
    return var_bl * (t1 - t0) ** 2


def get_var_H_q(cov_q_ab, t0, t1):
    """Calculate variance in quadratic H with manuscript equation (18)."""
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


def get_grad_H_l(bl, t0, t1):
    return grad(get_H_l)(bl, t0, t1)


def get_jac_H_q(aq_bq, t0, t1):
    return jacobian(get_H_q)(aq_bq, t0, t1)
