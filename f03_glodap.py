from sys import path

pyco2path = "/Users/matthew/github/PyCO2SYS"
if pyco2path not in path:
    path.append(pyco2path)

import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import xarray as xr
import numpy as np
from scipy.stats import linregress
from scipy.optimize import least_squares
from takahashi93 import temperature as ex_temperature

# opt_k_carbonic = 10
opt_total_borate = 1

# Import OceanSODA
soda = xr.open_dataset(
    "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
    + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
)
gvars = ["temperature", "salinity", "talk", "dic"]

# # Import the GLODAP gridded dataset (Lauvset et al., 2016)
# gpath = "/Users/matthew/Documents/data/GLODAP/GLODAPv2.2016b_MappedClimatologies/"
# gvars = ["temperature", "salinity", "silicate", "PO4", "TCO2", "TAlk"]
# glodap = []
# for gvar in gvars:
#     gfile = "GLODAPv2.2016b.{}.nc".format(gvar)
#     glodap.append(xr.open_dataset(gpath + gfile)[gvar])
# glodap = xr.merge(glodap)
# glodap_i = glodap.copy()
# glodap_i["silicate"] = glodap.silicate.interpolate_na("lat").interpolate_na("lon")
# glodap_i["PO4"] = glodap.PO4.interpolate_na("lat").interpolate_na("lon")

grads = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_borate",
    "k_water",
    # "k_silicate",
]


for opt_k_carbonic in range(1, 18):
    print(opt_k_carbonic)
    # Calculate surface field of dlnpCO2/dT
    results = pyco2.sys(
        # par1=glodap.isel(depth_surface=0).TAlk.data,
        # par2=glodap.isel(depth_surface=0).TCO2.data,
        par1=soda.talk.mean("time").data,
        par2=soda.dic.mean("time").data,
        par1_type=1,
        par2_type=2,
        # temperature=glodap.isel(depth_surface=0).temperature.data,
        # salinity=glodap.isel(depth_surface=0).salinity.data,
        temperature=soda.temperature.mean("time").data,
        salinity=soda.salinity.mean("time").data,
        # total_silicate=glodap_i.isel(depth_surface=0).silicate.data,
        # total_phosphate=glodap_i.isel(depth_surface=0).PO4.data,
        opt_k_carbonic=opt_k_carbonic,
        opt_total_borate=opt_total_borate,
        grads_of=["pCO2", *grads],
        grads_wrt=["temperature", *grads],
    )
    # glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)] = (
    #     ("lat", "lon"),
    #     results["dlnpCO2_dT"] * 100,
    # )
    soda["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        results["dlnpCO2_dT"] * 100,
    )
    pCO2_wf = results["d_pCO2__d_temperature"] / results["pCO2"]
    pCO2_wf_components = {}
    pCO2_wf_percent = {}
    pCO2_wf_sum = 0
    pCO2_wf_percent_sum = 0
    for k in grads:
        k_comp = (
            results["d_" + k + "__d_temperature"] * results["d_pCO2__d_" + k]
        ) / results["pCO2"]
        pCO2_wf_components[k] = k_comp
        # glodap[k] = (("lat", "lon"), k_comp)
        soda[k] = (("lat", "lon"), k_comp)
        pCO2_wf_percent[k] = 100 * k_comp / pCO2_wf
        pCO2_wf_sum += k_comp
        pCO2_wf_percent_sum += pCO2_wf_percent[k]

    # Calculate mean surface ocean conditions
    # mean_surface = {gvar: glodap[gvar].isel(depth_surface=0).mean() for gvar in gvars}
    mean_surface = {gvar: soda[gvar].mean() for gvar in gvars}
    mean_k_percent = {k: np.nanmean(v) for k, v in pCO2_wf_percent.items()}
    std_k_percent = {k: np.nanstd(v) for k, v in pCO2_wf_percent.items()}
    p95_k_percent = {
        k: [np.percentile(v[~np.isnan(v)], 0.01), np.percentile(v[~np.isnan(v)], 0.99)]
        for k, v in pCO2_wf_percent.items()
    }

    # TODO half-way through converting this from GLODAP to OceanSODA - but better to put
    # it in a different script so I have both options!!!

    # %% Simulate T93 experiment across the globe
    fits_linear = np.full_like(
        glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    fits_poly_const = np.full_like(
        glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    fits_poly_Tcoeff = np.full_like(
        glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data, np.nan
    )
    ex_pCO2 = np.full(
        (
            *glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].shape,
            ex_temperature.size,
        ),
        np.nan,
    )
    for i, t in enumerate(ex_temperature):
        print(i)
        ex_pCO2[:, :, i] = pyco2.sys(
            par1=glodap.isel(depth_surface=0).TAlk.data,
            par2=glodap.isel(depth_surface=0).TCO2.data,
            par1_type=1,
            par2_type=2,
            temperature=t,
            salinity=glodap.isel(depth_surface=0).salinity.data,
            total_silicate=glodap_i.isel(depth_surface=0).silicate.data,
            total_phosphate=glodap_i.isel(depth_surface=0).PO4.data,
            opt_k_carbonic=opt_k_carbonic,
            opt_total_borate=opt_total_borate,
        )["pCO2"]
    for i in range(180):
        print(i)
        for j in range(360):
            if ~np.isnan(ex_pCO2[i, j, 0]):
                ex_lnpCO2 = np.log(ex_pCO2[i, j, :])
                ex_linear = linregress(ex_temperature, ex_lnpCO2)
                fits_linear[i, j] = ex_linear.slope
                ex_poly = np.polyfit(ex_temperature, ex_lnpCO2, 2)
                fits_poly_const[i, j], fits_poly_Tcoeff[i, j] = ex_poly[:2]
    glodap["fits_linear_{:02.0f}".format(opt_k_carbonic)] = (
        ("lat", "lon"),
        fits_linear * 100,
    )

    # Get axis limits
    pt = glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    pt = pt[~np.isnan(pt)]
    fl = glodap["fits_linear_{:02.0f}".format(opt_k_carbonic)].to_numpy().ravel()
    fl = fl[~np.isnan(fl)]
    xlims = (
        min(np.quantile(fl, 0.005), np.quantile(pt, 0.005)),
        max(np.quantile(fl, 0.995), np.quantile(pt, 0.995)),
    )

    # %% Visualise - map
    for fvar in [
        "dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic),
        "fits_linear_{:02.0f}".format(opt_k_carbonic),
    ]:
        fig, ax = plt.subplots(
            dpi=300, subplot_kw={"projection": ccrs.Robinson(central_longitude=205)}
        )
        fm = glodap[fvar].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            # vmin=3.95, vmax=4.7
            vmin=xlims[0],
            vmax=xlims[1],
        )
        plt.colorbar(
            fm,
            location="bottom",
            label="100 × ∂(ln $p$CO$_2$)/∂$T$ / °C$^{–1}$",
            pad=0.05,
            aspect=20,
            fraction=0.05,
            extend="both",
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "land", "50m"),
            facecolor=0.1 * np.array([1, 1, 1]),
        )
        fig.tight_layout()
        fig.savefig("figures/f03/f03_map_{}_{:02.0f}.png".format(fvar, opt_k_carbonic))
        plt.show()
        plt.close()

    # %% Visualise - histogram
    fig, ax = plt.subplots(dpi=300, figsize=(12 / 2.54, 8 / 2.54))
    ax.hist(
        glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data.ravel(),
        bins=np.arange(3.5, 5, 0.01),
        facecolor="xkcd:midnight",
        label="PyCO2SYS",
    )
    ax.hist(
        glodap["fits_linear_{:02.0f}".format(opt_k_carbonic)].data.ravel(),
        bins=np.arange(3.5, 5, 0.01),
        facecolor="xkcd:ocean blue",
        alpha=0.85,
        label="T93 experiment",
    )
    ax.legend()
    ax.set_xlim(xlims)
    # ax.set_xlim((3.8, 4.8))
    ax.set_ylabel("Frequency")
    ax.set_xlabel("100 × ∂(ln $p$CO$_2$)/∂$T$ / °C$^{–1}$")
    fig.tight_layout()
    fig.savefig("figures/f03/f03_histogram_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()

    # %% Visualise - violins
    fig, ax = plt.subplots(dpi=300, figsize=(8 / 2.54, 12 / 2.54))
    ifac = 1  # bigger number pushes the violins together
    fvars = grads.copy()
    maxfreq = []
    vdiff = []
    for i, var in enumerate(fvars):
        fvar = pCO2_wf_percent[var]
        hist = np.histogram(fvar[~np.isnan(fvar)], bins=100)
        maxfreq.append(np.max(hist[0]))
        vdiff.append(np.mean(np.diff(hist[1])))
    maxfreq = np.array(maxfreq) / max(maxfreq)
    for i, var in enumerate(fvars):
        fvar = pCO2_wf_percent[var]
        parts = ax.violinplot(
            fvar[~np.isnan(fvar)],
            [i / ifac],
            showextrema=False,
            points=100,
            widths=1.6 * maxfreq[i] / vdiff[i],
        )
        parts["bodies"][0].set_facecolor("xkcd:midnight")
        parts["bodies"][0].set_alpha(0.8)
        # ax.plot(
        #     [i / ifac, i / ifac],
        #     [np.nanmin(fvar), np.nanmax(fvar)],
        #     c="xkcd:midnight",
        #     lw=1,
        # )
    ax.grid(alpha=0.2, axis="y")
    ax.set_xlim((-1, 6))
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(0, len(fvars) / ifac, 1 / ifac))
    ax.set_yticks(np.arange(-100, 120, 20))
    ax.set_ylim([-80, 100])
    flabels = {
        "k_CO2": r"$K_{\mathrm{CO}_2}^*$",
        "k_carbonic_2": "$K_2^*$",
        "k_carbonic_1": "$K_1^*$",
        "k_borate": r"$K_\mathrm{B}^*$",
        "k_water": "$K_w^*$",
        "k_silicate": r"$K_\mathrm{Si}^*$",
    }
    ax.set_xticklabels([flabels[k] for k in fvars])
    ax.set_ylabel("Contribution to ∂(ln $p$CO$_2$)/∂$T$ / %")
    ax.tick_params(top=True, labeltop=True)
    fig.tight_layout()
    fig.savefig("figures/f03/f03_violins_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()

    # %% Temperature relationship
    fx = glodap.isel(depth_surface=0).temperature.data.ravel()
    fy = glodap["dlnpCO2_dT_{:02.0f}".format(opt_k_carbonic)].data.ravel()
    L = ~np.isnan(fx) & ~np.isnan(fy)
    fx, fy = fx[L], fy[L]

    tfit = np.polyfit(fx, fy, 3)
    # tfit = array([-3.27613909e-06,  4.90600047e-04, -3.29258551e-02,  4.58750033e+00])
    fvx = np.linspace(np.min(fx), np.max(fx), num=500)
    fvy = np.polyval(tfit, fvx)

    def sensitivity(coeffs, temperature):
        a, b, c, d = coeffs
        return a + b * np.exp(c * (temperature + d))

    def _lsqfun_sensitivity(coeffs, temperature, dlnpCO2_dT):
        return sensitivity(coeffs, temperature) - dlnpCO2_dT

    opt_result = least_squares(_lsqfun_sensitivity, [0, 1, -1, 0], args=(fx, fy))
    # opt_result['x'] = array([ 3.57413027,  1.32214733,  0.03284989, -8.09607568])
    fvy_exp = sensitivity(opt_result["x"], fvx)

    # Visualise
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(
        fx,
        fy,
        c="xkcd:midnight",
        s=2,
        edgecolor="none",
        alpha=0.2,
    )
    ax.plot(fvx, fvy, c="xkcd:strawberry")
    # ax.plot(fvx, fvy_exp, c="xkcd:strawberry")
    ax.set_xlabel("Temperature / °C")
    ax.set_ylabel("∂(ln $p$CO$_2$)/∂$T$ / °C$^{–1}$")
    # ax.scatter(
    #     glodap.isel(depth_surface=0).temperature.data.ravel(),
    #     glodap["fits_linear_{:02.0f}".format(opt_k_carbonic)].data.ravel(),
    # )
    fig.tight_layout()
    fig.savefig("figures/f03/f03_temperature_fit_{:02.0f}.png".format(opt_k_carbonic))
    plt.show()
    plt.close()

# Save GLODAP dataset with calculations in
glodap.to_netcdf("quickload/f03_glodap.nc")
