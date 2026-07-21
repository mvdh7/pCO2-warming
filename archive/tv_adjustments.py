from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import takahashi93 as tak93

t_in = 25
t_out = 35

alkalinity, fCO2_std = tak93.get_alkalinity(10, 1)
results = pyco2.sys(
    par1=tak93.dic,
    par2=alkalinity,
    par1_type=2,
    par2_type=1,
    temperature=t_in,
    **tak93.tak93
)
adjust = pyco2.sys(
    par1=results["fCO2"],
    par1_type=5,
    temperature=t_in,
    temperature_out=t_out,
    opt_adjust_temperature=[5, 6, 2],
    **tak93.tak93
)
fCO2a = adjust["fCO2_out"]
print(
    "Adjustment from {} to {} °C with quadratic is {:.1f}".format(
        t_in, t_out, fCO2a[0] - fCO2a[1]
    )
    + " µatm lower than with linear."
)
print(
    "Adjustment from {} to {} °C with van 't Hoff is {:.1f}".format(
        t_in, t_out, fCO2a[0] - fCO2a[2]
    )
    + " µatm lower than with linear."
)
print("RMSD of Lu00 fit to Ta93 dataset is {:.1f} µatm.".format(fCO2_std))
