import xarray as xr

# soda = xr.open_dataset(
#     "/Users/matthew/Documents/data/OceanSODA/0220059/5.5/data/0-data/"
#     + "OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
# )

soda = xr.open_dataset(
    "/Users/matthew/github/pCO2-warming/quickload/soda_monthly.zarr",
    engine="zarr",
)
