
'''
todo:

- rh: either rh500 or dew point temp or rh700




'''









import xarray as xr
import numpy as np  
import pygrib
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import tarfile
from datetime import timedelta


# ============================================================
# Settings
# ============================================================

# Unfortunately chooosing a time range is difficult since the time is encoded in the filename of the archives
# which name patterns change over time. So rn the only way to limit the time range is to adjust the filename patterns below

base_folder = Path(r"/mnt/cosmo")                               # set to the real folder for full run

output_path_nc = Path("StatHintereisModelArchive2.nc")           # names
output_path_csv = Path("StatHintereisModelArchive2.csv")

target_lat = 46.798896
target_lon = 10.760373

min_hour = 2                                                    # depending on if the model is run daily or at different intervals adjust
max_hour = 25                                                  # in some of the datasets the radiation is not available for the first 2 timesteps

tmp_base = Path("/mnt/data/workspace/tmp_extract")
tmp_base.mkdir(parents=True, exist_ok=True)

qc = False                                                      # quality control (needs adjustment in the ICON and cosmo dicts)


# ==================================================
# Conversion functions
# ==================================================

def identity(x):
    return x

def Pa_to_hPa(x):
    return x / 100

def percent_to_fraction(x):
    return x / 100


def acc_rain_to_mm(x):                                          #only works for hourly data
    x_out = np.empty_like(x)
    x_out[0] = x[0]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] >=0:
            x_out[i] = x[i] - x[i - 1]
        else:
            x_out[i] = x[i]
    return x_out

ICON = {
    "raw": {
        "PRES": {
            "name": "Air pressure",
            "GRIB_key_number": 7,
            "units_out": "hPa",
            "convert": Pa_to_hPa,
            "valid_range": (890, 1080),
        },

        "T2": {
            "name": "Air temperature",
            "GRIB_key_number": 1,
            "units_out": "K",
            "convert": identity,
            "valid_range": (240, 295),
        },

        "Td2": {
            "name": "Dew point temperature",
            "GRIB_key_number": 2,
            "units_out": "K",
            "convert": identity,
            "valid_range": (230, 295),
        },

        "N": {
            "name": "Cloud cover",
            "GRIB_key_number": 16,
            "units_out": "-",
            "convert": percent_to_fraction,
            "valid_range": (-0.1, 11),
        },

        "G": {
            "name": "Solar radiation",
            "GRIB_key_number": 21,
            "units_out": "W m^-2",
            "convert": identity,
            "valid_range": (-500, 1700),
        },

        "LWin": {
            "name": "Incoming longwave radiation",
            "GRIB_key_number": 24,
            "units_out": "W m^-2",
            "convert": identity,
            "valid_range": (-5000, 1600),
        },

        "RRR": {
            "name": "Rain rate",
            "GRIB_key_number": 8,
            "units_out": "mm h^-1",
            "convert": identity,
            "valid_range": (0, 2000),
        },

        "SNOWDEPTH": {
            "name": "Snow depth swe",
            "GRIB_key_number": 3,
            "units_out": "mm",
            "convert": identity,
            "valid_range": (0, 50000),
        },

        "SNOWFALL_acc": {
            "name": "Snowfall SWE acc",
            "GRIB_key_number": 9,
            "units_out": "mm",
            "convert": identity,
            "valid_range": (0, 5000),
        },

        "GRAUPEL_acc": {
            "name": "Graupel swe acc",
            "GRIB_key_number": 10,
            "units_out": "kg m**-2",
            "convert": identity,
            "valid_range": (0, 5000),
        },

        "U10m": {
            "name": "10m U wind component",
            "GRIB_key_number": 4,
            "units_out": "m s^-1",
            "convert": identity,
            "valid_range": (-70, 70),
        },

        "V10m": {
            "name": "10m V wind component",
            "GRIB_key_number": 5,
            "units_out": "m s^-1",
            "convert": identity,
            "valid_range": (-70, 70),
        },
  
        "RH_500": {
            "name": "Relative humidity 500hPa",                 #alarm
            "GRIB_key_number": 36,
            "units_out": "%",
            "convert": identity,
            "valid_range": (-0.01, 101),
        },
        "RH_700": {
            "name": "Relative humidity 700hPa",
            "GRIB_key_number": 37,
            "units_out": "%",
            "convert": identity,
            "valid_range": (0, 101),
        },
    },
    "derived": {
        
        "U10": {
            "name": "Wind speed",
            "units_out": "m s^-1",
            "valid_range": (0, 100),

            "formula": lambda ds: np.sqrt(
                ds["U10m"]**2 + ds["V10m"]**2
            ),
        },
        "RRR_h": {
            "name": "Hourly rainfall",
            "description": "Hourly rainfall derived from accumulated rainfall",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["RRR"].values),
                dims=ds["RRR"].dims,
                coords=ds["RRR"].coords,
            ),
        },

        "SNOWFALL_h": {
            "name": "Hourly snowfall",
            "description": "Hourly snowfall water equivalent derived from accumulated snowfall",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["SNOWFALL_acc"].values),
                dims=ds["SNOWFALL_acc"].dims,
                coords=ds["SNOWFALL_acc"].coords,
            ),
        },

        "GRAUPEL_h": {
            "name": "Hourly graupel precipitation",
            "description": "Hourly graupel water equivalent derived from accumulated graupel",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["GRAUPEL_acc"].values),
                dims=ds["GRAUPEL_acc"].dims,
                coords=ds["GRAUPEL_acc"].coords,
            ),
        },

        "TOTAL_PRCEP": {
            "name": "Hourly total precipitation",
            "description": (
                "Hourly total precipitation derived from "
                "RRR_h, SNOWFALL_h, and GRAUPEL_h"
            ),
            "units_out": "mm h^-1",
            "formula": lambda ds: (
                ds["RRR_h"]
                + ds["SNOWFALL_h"]
                + ds["GRAUPEL_h"]
            ),
        },
    }

            

        
}
COSMO = {
    "raw": {
        "T2": {
            "name": "Air temperature",
            "GRIB_key_number": 1,
            "units_out": "K",
            "convert": identity,
            "valid_range": (200, 330),
        },
        "Td2": {
            "name": "Dew point temperature",
            "GRIB_key_number": 2,
            "units_out": "K",
            "convert": identity,
            "valid_range": (180, 330),
        },
        "SNOWDEPTH": {
            "name": "Snow depth swe",
            "GRIB_key_number": 3,
            "units_out": "kg m**-2",
            "convert": identity,
            "valid_range": (0, 50000),
        },
        "U10m": {
            "name": "10 metre U wind component",
            "GRIB_key_number": 4,
            "units_out": "m s^-1",
            "convert": identity,
            "valid_range": (-100, 100),
        },
        "V10m": {
            "name": "10 metre V wind component",
            "GRIB_key_number": 5,
            "units_out": "m s^-1",
            "convert": identity,
            "valid_range": (-100, 100),
        },
        "PRES": {
            "name": "Mean sea level pressure",          #iwie noch rausfinden wie hoch der gridpunkt ist und umrechnen
            "GRIB_key_number": 7,
            "units_out": "hPa",
            "convert": Pa_to_hPa,
            "valid_range": (400, 1080),
        },
        "RRR": {
            "name": "surface precipitation amount, rain, grid scale",
            "GRIB_key_number": 8,
            "units_out": "kg m**-2",
            "convert": identity,
            "valid_range": (0, 500),
        },
        "SNOWFALL_acc": {
            "name": "Snowfall SWE acc",
            "GRIB_key_number": 9,
            "units_out": "mm",
            "convert": identity,
            "valid_range": (0, 500),
        },
                "GRAUPEL_acc": {
            "name": "Graupel swe acc",
            "GRIB_key_number": 10,
            "units_out": "kg m**-2",
            "convert": identity,
            "valid_range": (0, 500),
        },
        "N": {
            "name": "Total Cloud Cover",
            "GRIB_key_number": 16,
            "units_out": "-",
            "convert": percent_to_fraction,
            "valid_range": (-0.001, 1.01),
        },
        "G": {
            "name": "Downward shortwave radiation flux at surface (time average)",          #checken ob net shortwave sinn macht u was man dagegen tun kann evtl gribkey 21
            "GRIB_key_number": 21,
            "units_out": "W m^-2",
            "convert": identity,
            "valid_range": (-5, 1700),
        },
        "LWin": {
            "name": "Average downward longwave radiation at the surface based on T_G",
            "GRIB_key_number": 24,
            "units_out": "W m^-2",
            "convert": identity,
            "valid_range": (-1500, 1600),
        },
        "RH_500": {
            "name": "Relative humidity 500hPa",
            "GRIB_key_number": 36,
            "units_out": "%",
            "convert": identity,
            "valid_range": (0, 101),
        },
        
        "RH_700": {
            "name": "Relative humidity 700hPa",
            "GRIB_key_number": 37,
            "units_out": "%",
            "convert": identity,
            "valid_range": (0, 101),
        },
    },

    "derived": {
        
        "U10": {
            "name": "Wind speed",
            "units_out": "m s^-1",
            "formula": lambda ds: np.sqrt(ds["U10m"]**2 + ds["V10m"]**2),
        },
        "RRR_h": {
            "name": "Hourly rainfall",
            "description": "Hourly rainfall derived from accumulated rainfall",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["RRR"].values),
                dims=ds["RRR"].dims,
                coords=ds["RRR"].coords,
            ),
        },

        "SNOWFALL_h": {
            "name": "Hourly snowfall",
            "description": "Hourly snowfall water equivalent derived from accumulated snowfall",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["SNOWFALL_acc"].values),
                dims=ds["SNOWFALL_acc"].dims,
                coords=ds["SNOWFALL_acc"].coords,
            ),
        },

        "GRAUPEL_h": {
            "name": "Hourly graupel precipitation",
            "description": "Hourly graupel water equivalent derived from accumulated graupel",
            "units_out": "mm h^-1",
            "formula": lambda ds: xr.DataArray(
                acc_rain_to_mm(ds["GRAUPEL_acc"].values),
                dims=ds["GRAUPEL_acc"].dims,
                coords=ds["GRAUPEL_acc"].coords,
            ),
        },

        "TOTAL_PRCEP": {
            "name": "Hourly total precipitation",
            "description": (
                "Hourly total precipitation derived from "
                "RRR_h, SNOWFALL_h, and GRAUPEL_h"
            ),
            "units_out": "mm h^-1",
            "formula": lambda ds: (
                ds["RRR_h"]
                + ds["SNOWFALL_h"]
                + ds["GRAUPEL_h"]
            ),
        },
    }
        
    
}





# ============================================================
# Different dataset names
# ============================================================
icon_archives = sorted(
    base_folder.glob("20*_*_*_03_icon-ch1-eps_uibk_acinn.zip")              # only way to adjust time range is to adjust these name patterns
)                                                                           # example filename: 2023_09_03_icon-ch1-eps_uibk_acinn.zip

cosmo_archives_vnrz = sorted(
    base_folder.glob("20*/VNRZ06.20*.tgz")                                  # example filename: 2023/VNRZ06.202309050300.tgz
)


cosmo_archives_100 = sorted(
    base_folder.glob("20*/*_100.tar.gz")                                    # example filename 2016/16012500_100.tar.gz                        
)

cosmo_archives_570 = sorted(
    base_folder.glob("20*/*_570.tar.gz")                                    # example filename 2014/14012500_570.tar.gz
)

cosmo_archives = sorted(
    cosmo_archives_vnrz
    + cosmo_archives_100
    + cosmo_archives_570
)

print("ICON archives found:", len(icon_archives))
print("COSMO archives found:", len(cosmo_archives))


# ============================================================
# extraction
# ============================================================

def extract_archive(archive_path, out_dir):
    archive_path = Path(archive_path)

    try:
        if archive_path.suffix == ".zip":
            if not zipfile.is_zipfile(archive_path):
                print(f"Skipping invalid ZIP: {archive_path}")
                return False

            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(out_dir)

        elif archive_path.suffix == ".tgz" or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as t:
                t.extractall(out_dir)

        else:
            print(f"Skipping unsupported archive type: {archive_path} (not .zip or.tgz)")
            return False

    except Exception as e:
        print(f"Skipping archive due to error: {archive_path}")
        print(e)
        return False

    return True


def get_forecast_hour(file, model_name):
    file = Path(file)
    if model_name == "ICON":
        x = file.stem.split("_h_")[1].split("_")[0]             # example filename: icon-ch1-eps_uibk_acinn_h_01_m_000.grb2
        return int(x)

    if model_name == "COSMO":                                   #example filename: imgi_cosmo1_00.grb1
        x = file.stem.split("_")[-1]
        return int(x)

    raise ValueError(f"Unknown model/datset name: {model_name}")


def process_archives(file_archives, model_dict, model_name, grib_pattern):

    variables = list(model_dict["raw"].keys())                         # variable list from the dataset dicts (raw variables)

    times = []
    data = {var: [] for var in variables}
    grid_lats = []
    grid_lons = []
    source_model = []
    archive_type = []
    source_archive = []

    for archive in file_archives:

        print(f"Processing:", archive)

        if "VNRZ06" in archive.name:
            this_archive_type = "VNRZ"
        elif "_100" in archive.name:
            this_archive_type = "COSMO_100"
        elif "_570" in archive.name:
            this_archive_type = "COSMO_570"
        elif "icon" in archive.name.lower():
            this_archive_type = "ICON"
        else:
            this_archive_type = "unknown"

        with tempfile.TemporaryDirectory(dir=tmp_base) as tmpdir:

            tmpdir = Path(tmpdir)

            ok = extract_archive(archive, tmpdir)

            if not ok:
                continue


            grib_files = []

            for pattern in grib_pattern:
                for file in tmpdir.rglob(pattern):
                    hour = get_forecast_hour(file, model_name)
                    if this_archive_type == "COSMO_570" and hour > 0:       # _570 archive only has 24 hours
                        grib_files.append(file)
                    elif min_hour <= hour <= max_hour:
                        grib_files.append(file)

            
            grib_files = sorted(
                set(grib_files),
                key=lambda f: get_forecast_hour(f, model_name)
            )


            print(archive.name, ": found ", len(grib_files), "GRIB files")

            for file in grib_files:

                if model_name == "ICON":
                    this_source_model = "ICON"

                elif model_name == "COSMO":
                    fname = file.name.lower()

                    if "cosmo1e_" in fname:
                        this_source_model = "COSMO1E"
                    elif "cosmo1_" in fname:
                        this_source_model = "COSMO1"
                    else:
                        this_source_model = "COSMO_unknown"

                else:
                    this_source_model = model_name

                with pygrib.open(str(file)) as grbs:

                    values_by_var = {var: np.nan for var in variables}          # dict to safe data of 1 timestep and check validity of the values

                    lats = None
                    lons = None
                    idx = None
                    valid_date = None

                    for var_name, meta in model_dict["raw"].items():

                        try:
                            grb = grbs.message(meta["GRIB_key_number"])
                        except RuntimeError as e:
                            print(
                                f"Missing GRIB message: {model_name} {var_name}, "
                                f"message={meta['GRIB_key_number']}, "
                                f"file={file.name}, error={e}"
                            )
                            continue

                        if lats is None:
                            lats, lons = grb.latlons()

                            dist = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
                            idx = np.unravel_index(np.argmin(dist),dist.shape)

                        if valid_date is None:
                            valid_date = grb.validDate

                            if model_name == "COSMO":
                                hour_offset = int(file.stem.split("_")[-1])
                                valid_date = valid_date + timedelta(hours=hour_offset) 

                        raw_value = float(grb.values[idx])

                        value = meta["convert"](raw_value)
                        #value = raw_value

                        vmin, vmax = meta["valid_range"]

                        
                        if qc:

                            if vmin <= value <= vmax:
                                values_by_var[var_name] = value
                            else:
                                values_by_var[var_name] = np.nan
                                print(
                                    f"QC failed: {model_name} {var_name}, "
                                    f"value={value:.3f}, "
                                    f"allowed=[{vmin}, {vmax}], "
                                    f"file={file.name}, "
                                    f"time={grb.validDate}"
                                )
                        else:
                            values_by_var[var_name] = value

                if valid_date is None:
                    continue

                times.append(pd.to_datetime(valid_date))
                grid_lats.append(float(lats[idx]))
                grid_lons.append(float(lons[idx]))
                source_model.append(this_source_model)
                archive_type.append(this_archive_type)
                source_archive.append(archive.name)

                for var in variables:
                    data[var].append(values_by_var[var])

    if len(times) == 0:
        return None

    ds = xr.Dataset(
        data_vars={
            **{
                var: (
                    ("time",),
                    np.array(data[var], dtype=float),
                    {
                        "long_name": model_dict["raw"][var]["name"],
                        "units": model_dict["raw"][var]["units_out"],
                    },
                )
                for var in variables
            },
            "source_model": (
                ("time",),
                np.array(source_model, dtype=str),
                {
                    "long_name": "Source model",
                    "description": "Model used for this timestep: COSMO or ICON",
                },
            ),

            "archive_type": (
                ("time",),
                np.array(archive_type, dtype=str),
                {
                    "long_name": "Archive type",
                    "description": "Archive family: VNRZ, COSMO_100, COSMO_570, ICON",
                },
            ),

            "source_archive": (
                ("time",),
                np.array(source_archive, dtype=str),
                {
                    "long_name": "Source archive",
                    "description": "Original archive filename",
                },
            ),
        },
        coords={
            "time": pd.to_datetime(times),

            
            "target_lat": target_lat,
            "target_lon": target_lon,

            
            "grid_lat": ("time", np.array(grid_lats, dtype=float)),
            "grid_lon": ("time", np.array(grid_lons, dtype=float)),
        },
        attrs={
            "description": "Point time series extracted from ACINN server",
        },
    )

    ds = ds.sortby("time")


  

    for var_name, meta in model_dict["derived"].items():

        ds[var_name] = meta["formula"](ds)

        ds[var_name].attrs = {
            "long_name": meta["name"],
            "description": meta.get("description", meta["name"]),
            "units": meta["units_out"],
        }

    return ds

# ============================================================
# main
# ============================================================

#cosmo_archives = cosmo_archives[:1]
#icon_archives = icon_archives[:1]

ds_cosmo = process_archives(
    file_archives=cosmo_archives,
    model_dict=COSMO,
    model_name="COSMO",
    grib_pattern=["grib/imgi_cosmo1_[0-9][0-9].grb1", "grib/imgi_cosmo1e_[0-9][0-9].grb1"],
)

ds_icon = process_archives(
    file_archives=icon_archives,
    model_dict=ICON,
    model_name="ICON",
    grib_pattern=["icon-ch1-eps_uibk_acinn_h_[0-9][0-9]_m_000.grb2"],
)


datasets = []

if ds_cosmo is not None:
    datasets.append(ds_cosmo)

if ds_icon is not None:
    datasets.append(ds_icon)

ds_all = xr.concat(datasets, dim="time")
ds_all = ds_all.sortby("time")


keep = ~pd.Index(ds_all.time.values).duplicated(keep="last")

ds_final = ds_all.isel(time=keep)

# Complete hourly time axis
full_time = pd.date_range(
    ds_final.time.values[0],
    ds_final.time.values[-1],
    freq="1h",
)

missing = full_time.difference(
    pd.DatetimeIndex(ds_final.time.values)
)
print("Missing timesteps:", missing)

# Insert missing timestamps as NaN
ds_final = ds_final.reindex(time=full_time)

# Fill each missing timestep with previous hour
for t in missing:
    previous_t = t - pd.Timedelta(hours=1)

    ds_final.loc[dict(time=t)] = ds_final.sel(time=previous_t)
    print(f"Filled missing timestep {t} with previous timestep {previous_t}")


ds_final = ds_final.assign_coords(
    target_lat=target_lat,
    target_lon=target_lon,
)

ds_final.attrs["description"] = (
    "Combined COSMO/ICON point time series; ICON preferred where overlapping"
)


ds_final.to_netcdf(output_path_nc, mode="w", format="NETCDF4")
ds_final.to_dataframe().to_csv(output_path_csv, index=True)

print("Done! Dataset saved to:", output_path_nc)