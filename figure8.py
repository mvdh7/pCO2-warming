from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate
import pwtools

use_quickload = True
dts = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
opt_at = [5, 6, 1, 2]

if not use_quickload:
    soda_monthly = xr.open_dataset("quickload/soda_monthly.zarr", engine="zarr")

    for i, dt in enumerate(dts):
        print(i + 1, "/", len(dts))
        # Convert with PyCO2SYS from TA and DIC
        shape3 = soda_monthly.temperature.data.shape
        results_calc12 = pyco2.sys(
            par1=soda_monthly.talk.data,
            par1_type=1,
            par2=soda_monthly.dic.data,
            par2_type=2,
            salinity=soda_monthly.salinity.data,
            temperature=soda_monthly.temperature.data,
            temperature_out=soda_monthly.temperature.data + dt,
            opt_k_carbonic=10,
            opt_total_borate=1,
            opt_buffers_mode=0,
        )
        soda_monthly["fCO2_dt_calc12_{}".format(dt)] = (
            ("month", "lat", "lon"),
            results_calc12["fCO2_out"]
            - results_calc12["fCO2"]
            + soda_monthly.fCO2.data,
        )

    # %% Convert with the various upsilon options
    for i, dt in enumerate(dts):
        print(i + 1, "/", len(dts))
        results_dt = pyco2.sys(
            par1=soda_monthly.fCO2.data,
            par1_type=5,
            salinity=soda_monthly.salinity.data,
            temperature=soda_monthly.temperature.data,
            temperature_out=soda_monthly.temperature.data + dt,
            opt_k_carbonic=10,
            opt_total_borate=1,
            opt_adjust_temperature=np.reshape(
                np.array([[opt_at]]), [len(opt_at), 1, 1, 1]
            ),
            opt_which_fCO2_insitu=1,
        )
        for i, at in enumerate(opt_at):
            soda_monthly["fCO2_dt_{}_{}".format(dt, at)] = (
                ("month", "lat", "lon"),
                results_dt["fCO2_out"][i],
            )
            soda_monthly["fCO2_dt_{}_{}_diff".format(dt, at)] = (
                soda_monthly["fCO2_dt_{}_{}".format(dt, at)]
                - soda_monthly["fCO2_dt_calc12_{}".format(dt)]
            )

    # %% Calculate statistics
    rmsd = np.full((len(opt_at), len(dts)), np.nan)
    for i, at in enumerate(opt_at):
        for j, dt in enumerate(dts):
            dtnp = soda_monthly["fCO2_dt_{}_{}_diff".format(dt, at)].to_numpy()
            dtnp = dtnp[~np.isnan(dtnp)]
            rmsd[i, j] = np.sqrt(np.mean(dtnp**2))
    np.save("quickload/figure8_rmsd.npy", rmsd)

else:
    rmsd = np.load("quickload/figure8_rmsd.npy")

# %% Visualise
fstyle = {
    1: dict(
        label="$υ_p$ (van 't Hoff, $b_h$ parameterised)",
        c=pwtools.blue,
        lw=1.5,
    ),
    2: dict(
        label="$υ_h$ (van 't Hoff, $b_h$ fitted)",
        c=pwtools.blue,
        lw=1.5,
        ls=(0, (3, 1)),
    ),
    5: dict(
        label="$υ_l$ (Ta93, linear)",
        c=pwtools.dark,
        lw=1.5,
        zorder=10,
    ),
    6: dict(
        label="$υ_q$ (Ta93, quadratic)",
        c=pwtools.dark,
        lw=1.5,
        ls=(0, (6, 2)),
        zorder=10,
    ),
}
fx = np.linspace(dts.min(), dts.max(), num=101)
fig, ax = plt.subplots(dpi=300, figsize=(9 / 2.54, 10 / 2.54))
for i, at in enumerate(opt_at):
    fy = pchip_interpolate(dts, rmsd[i], fx)
    ax.plot(fx, fy, **fstyle[at])
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.25), edgecolor="k", ncol=1, fontsize=9
)
ax.set_xlabel("∆$t$ / °C")
ax.set_ylabel(r"RMSD vs. $υ_\mathrm{Lu00}$ / µatm")
ax.set_xlim((dts.min(), dts.max()))
ax.set_ylim((0, 13))
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("figures_final/figure8.png")
