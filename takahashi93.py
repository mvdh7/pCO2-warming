from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import numpy as np

# Data from Takahashi et al. (1993) Table A1
dic = 2074
pCO2 = np.array([571.2, 376.1, 267, 322.1, 461.6, 688.2, 571, 526.3])
temperature = np.array([20, 10.05, 2.1, 6.33, 15.01, 24.5, 20, 18])
tak93 = dict(
    salinity=35.38,
    total_silicate=0,
    total_phosphate=0,
)


def get_alkalinity(opt_k_carbonic, opt_total_borate):
    # Determine alkalinity in the experiment as the mean alkalinity calculated from DIC
    # and pCO2 across all measurement points
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
