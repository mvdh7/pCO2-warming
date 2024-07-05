from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np
from scipy.optimize import least_squares

# Data from Takahashi et al. (1993) Table A1
dic = 2074
pCO2 = np.array([571.2, 376.1, 267, 322.1, 461.6, 688.2, 571, 526.3])
temperature = np.array([20, 10.05, 2.1, 6.33, 15.01, 24.5, 20, 18])
tak93 = dict(
    salinity=35.38,
    total_silicate=0,
    total_phosphate=0,
)
quadratic = np.array([-0.435e-4, 4.33e-2])

# Get RMSDs for fits
fit_linear = np.polyfit(temperature, np.log(pCO2), 1)
fit_quadratic = np.polyfit(temperature, np.log(pCO2), 2)
rmsd_linear = np.sqrt(
    np.mean((np.exp(np.polyval(fit_linear, temperature)) - pCO2) ** 2)
)
rmsd_quadratic = np.sqrt(
    np.mean((np.exp(np.polyval(fit_quadratic, temperature)) - pCO2) ** 2)
)


def get_alkalinity_old(opt_k_carbonic, opt_total_borate):
    """Determine alkalinity in the experiment as the mean alkalinity calculated from DIC
    and pCO2 across all measurement points.
    """
    r = pyco2.sys(
        par1=dic,
        par1_type=2,
        par2=pCO2,
        par2_type=4,
        temperature=temperature,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        **tak93,
    )
    alkalinity = np.mean(r["alkalinity"])
    alkalinity_std = np.std(r["alkalinity"])
    return alkalinity, alkalinity_std


def get_alkalinity(opt_k_carbonic, opt_total_borate):
    """Determine alkalinity in the experiment as the best-fitting alkalinity to match
    all the experimental pCO2 points.
    """

    def pCO2_from_alkalinity(alkalinity):
        return pyco2.sys(
            par1=dic,
            par1_type=2,
            par2=alkalinity,
            par2_type=1,
            temperature=temperature,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
            **tak93,
        )["pCO2"]

    def _lsqfun_pCO2_from_alkalinity(alkalinity, pCO2):
        return pCO2_from_alkalinity(alkalinity) - pCO2

    opt_result = least_squares(_lsqfun_pCO2_from_alkalinity, [2300], args=(pCO2,))
    return opt_result["x"][0], np.sqrt(np.mean(opt_result.fun**2))


def get_fCO2(opt_k_carbonic, opt_total_borate):
    """Convert pCO2 to fCO2 for the given `opt_k_carbonic` and `opt_total_borate`."""
    fCO2 = pyco2.sys(
        par1=dic,
        par1_type=2,
        par2=pCO2,
        par2_type=4,
        temperature=temperature,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        **tak93,
    )["fCO2"]
    return fCO2
