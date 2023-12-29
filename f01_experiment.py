from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from matplotlib import pyplot as plt
from takahashi93 import get_alkalinity, dic, tak93, pCO2, temperature

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

for opt_k_carbonic in range(1, 19):
    opt_total_borate = 1
    alkalinity, alkalinity_std = get_alkalinity(opt_k_carbonic, opt_total_borate)

    # Calculate pCO2 variation with temperature with different approaches
    v_temperature = np.linspace(-1.8, 30, num=100)
    # - with PyCO2SYS (autograd approach)
    v_results = pyco2.sys(
        par1=alkalinity,
        par1_type=1,
        par2=dic,
        par2_type=2,
        temperature=v_temperature,
        opt_k_carbonic=opt_k_carbonic,
        **tak93,
    )
    v_pCO2 = v_results["pCO2"]
    v_lnpCO2_PyCO2SYS = np.log(v_pCO2)
    # - with the linear temperature sensitivity of T93
    zero_lnpCO2 = np.log(pCO2) - 0.0423 * temperature
    v_lnpCO2_linear = np.mean(zero_lnpCO2) + v_temperature * 0.0423
    # - with the polynomial temperature sensitivity of T93
    zero_lnpCO2_poly = np.log(pCO2) - 0.0433 * temperature + 8.7e-5 * temperature**2
    v_lnpCO2_poly = (
        np.mean(zero_lnpCO2_poly) + v_temperature * 0.0433 - 8.7e-5 * v_temperature**2
    )

    # Calculate 1-degree warming correction (i.e., if seawater had warmed up by 1 °C
    # before being measured)
    dewarmed_PyCO2SYS = pyco2.sys(
        par1=alkalinity,
        par1_type=1,
        par2=dic,
        par2_type=2,
        temperature=v_temperature - 1,
        opt_k_carbonic=opt_k_carbonic,
        **tak93,
    )["pCO2"]
    dewarming_cx_PyCO2SYS = dewarmed_PyCO2SYS - v_pCO2
    dewarmed_linear = v_pCO2 * np.exp(0.0423 * ((v_temperature - 1) - v_temperature))
    dewarming_cx_linear = dewarmed_linear - v_pCO2
    dewarmed_poly = v_pCO2 * np.exp(
        0.0433 * ((v_temperature - 1) - v_temperature)
        - 4.35e-5 * ((v_temperature - 1) ** 2 - v_temperature**2)
    )
    dewarming_cx_poly = dewarmed_poly - v_pCO2

    # Visualise similar to Takahashi et al. (1993) Figure A1
    style_linear = dict(c="k", label="Ta93 linear", alpha=0.9, lw=2.5)
    style_poly = dict(label="Ta93 quadratic", alpha=0.9, lw=2.5, ls="--")
    style_pyco2 = dict(
        c="xkcd:coral",
        label="Theory ({})".format(okc_codes[opt_k_carbonic]),
        alpha=0.9,
        lw=2.5,
        ls=":",
    )
    fig, axs = plt.subplots(dpi=300, ncols=2, figsize=(17.4 / 2.54, 8 / 2.54))
    ax = axs[0]
    # ax.set_title("opt_k_carbonic = {}".format(tak93['opt_k_carbonic']))
    ax.text(0, 1.05, "(a)", transform=ax.transAxes)
    ax.scatter(
        temperature,
        # np.log(pCO2),
        pCO2,
        c="k",
        label="Ta93 measured",
        s=90,
        zorder=0,
        alpha=0.9,
        edgecolor="none",
    )
    ax.plot(v_temperature, np.exp(v_lnpCO2_linear), **style_linear)
    ax.plot(v_temperature, np.exp(v_lnpCO2_poly), **style_poly)
    ax.plot(v_temperature, np.exp(v_lnpCO2_PyCO2SYS), **style_pyco2)
    ax.legend()
    ax.set_xlabel("Equilibration temperature / °C")
    ax.set_ylabel("$p$CO$_2$ / µatm")
    ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))

    ax = axs[1]
    ax.text(0, 1.05, "(b)", transform=ax.transAxes)
    ax.axhline(1e3 * 0.0423, **style_linear)
    ax.plot(
        v_temperature,
        1e3 * (0.0433 - 8.7e-5 * v_temperature),
        c="xkcd:azure",
        **style_poly,
    )
    ax.plot(v_temperature, 1e3 * v_results["dlnpCO2_dT"], **style_pyco2)
    ax.set_xlabel("Equilibration temperature / °C")
    ax.set_ylabel("$η$ / kK$^{–1}$")
    ax.set_xlim((np.min(v_temperature), np.max(v_temperature)))

    # ax = axs[2]
    # ax.text(0, 1.03, "(c)", transform=ax.transAxes)
    # ax.plot(v_temperature, dewarming_cx_linear - dewarming_cx_PyCO2SYS, **style_linear)
    # ax.plot(v_temperature, dewarming_cx_poly - dewarming_cx_PyCO2SYS, **style_poly)
    # ax.axhline(0, c="k", lw=0.8)
    # ax.set_xlabel("Equilibration temperature / °C")
    # ax.set_ylabel("Warming corr. error / µatm")

    fig.tight_layout()
    fig.savefig("figures/f01/f01_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()
