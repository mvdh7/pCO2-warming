from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2

# Define uncertainties
u_alkalinity = {"par1": 8.1}
u_total = pyco2.uncertainty_OEDG18.copy()
u_total.update(u_alkalinity)

# Input parameters are medians from soda_monthly
results = pyco2.sys(
    par1=2290,
    par2=350,
    par1_type=1,
    par2_type=5,
    temperature=15,
    temperature_out=25,
    salinity=34.3,
    uncertainty_from=u_total,
    uncertainty_into=["fCO2_out"],
)

# %%
print("Total uncertainty in fCO2_out = {:.2f} µatm,".format(results["u_fCO2_out"]))
print(
    "of which alkalinity contributes {:.2f} µatm,".format(results["u_fCO2_out__par1"])
)
print(
    "which is a factor of {:.0f} smaller.".format(
        results["u_fCO2_out"] / results["u_fCO2_out__par1"]
    )
)
