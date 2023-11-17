from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
from takahashi93 import get_alkalinity, dic, tak93

opt_k_carbonic = 10
alkalinity, alkalinity_std = get_alkalinity(opt_k_carbonic)

# Calculate components of dlnpCO2/dT using forward finite difference derivatives
grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
    "k_bisulfate",
    "k_fluoride",
    "fugacity_factor",
    "k_silicate",
    "k_phosphoric_1",
    "k_phosphoric_2",
    "k_phosphoric_3",
]
rr = pyco2.sys(
    par1=alkalinity,
    par1_type=1,
    par2=dic,
    par2_type=2,
    temperature=15,
    **tak93,
    grads_of=["pCO2", *grads],
    grads_wrt=["temperature", *grads],
)
pCO2_wf_autograd = rr["dlnpCO2_dT"]
pCO2_wf = rr["d_pCO2__d_temperature"] / rr["pCO2"]
pCO2_wf_components = {}
pCO2_wf_percent = {}
pCO2_wf_sum = 0
pCO2_wf_percent_sum = 0
for k in grads:
    k_comp = (rr["d_" + k + "__d_temperature"] * rr["d_pCO2__d_" + k]) / rr["pCO2"]
    pCO2_wf_components[k] = k_comp
    pCO2_wf_percent[k] = 100 * k_comp / pCO2_wf
    pCO2_wf_sum += k_comp
    pCO2_wf_percent_sum += pCO2_wf_percent[k]

# Visualise
fig, ax = plt.subplots(dpi=300, figsize=(10 / 2.54, 7 / 2.54))
bvars = {
    "k_CO2": r"$K_{\mathrm{CO}_2}^*$",
    "k_carbonic_2": "$K_2^*$",
    "k_carbonic_1": "$K_1^*$",
    "k_borate": r"$K_\mathrm{B}^*$",
    "k_water": "$K_w^*$",
}
bvals = [pCO2_wf_percent[k] for k in bvars]
# bvars.append('Mismatch')
# bvals.append(100 - np.sum(bvals))
ax.bar(list(bvars.values()), bvals, facecolor="xkcd:midnight")
ax.axhline(0, c="k", lw=0.8)
ax.set_ylabel("Contribution to ∂(ln $p$CO$_2$)/∂$T$ / %")
ax.grid(alpha=0.2, axis="y")
ax.set_ylim((-60, 80))
fig.tight_layout()
