'''
------------------------------------------------------------
Module to download ICON-CH1 data from the OGD API and select a specific lat/lon point.


Example usage:

target_lat = 46.798896
target_lon = 10.760373
ds = download_ICON.download_ICON(target_lat, target_lon, ['QV'], ref_run="12H")


To Do:
- cdo installation so machen dass es nicht local in meinem dir liegt
- add forecast
- get rid of warning: "------------------------------------
WARNING: definitions.edzw version 2.47.0 is NOT compatible with ecCodes library version 2.47.3!
Please check environment settings and/or use ECCODES_DEFINITION_PATH environment variable for including compatible definitions.edzw version 2.47.3.
------------------------------------------------------"
------------------------------------------------------------
'''


from meteodatalab import ogd_api
from earthkit.data import config
from datetime import datetime, timezone
import xarray as xr
import numpy as np
from meteodatalab import grib_decoder, data_source
from pathlib import Path
from cdo import Cdo
import tempfile


cdo = Cdo()         # since cdo isnt really a python package it needs some sort of wrapper object

#CDO_TMP = Path("/mnt/data/workspace/simon/tmp_cache/cdo_tmp")       # had the problem that the default dir where this wrapper object is created was to small. this fixed it
#CDO_TMP.mkdir(parents=True, exist_ok=True)                          
#cdo = Cdo(tempdir=str(CDO_TMP))

valid_variables = [
    "ALB_DIF",
    "ALB_RAD",
    "ALHFL_S",
    "ASHFL_S",
    "ASOB_S",
    "ASOB_S_OS",
    "ASOD_S",
    "ASWDIFD_S",
    "ASWDIFU_S",
    "ASWDIFU_S_OS",
    "ASWDIR_S",
    "ASWDIR_S_OS",
    "ATHB_S",
    "ATHD_S",
    "ATHU_S",
    "AUMFL_S",
    "AVMFL_S",
    "CAPE_3KM",
    "CAPE_ML",
    "CAPE_MU",
    "CDCT",
    "CEILING",
    "CIN_ML",
    "CIN_MU",
    "CLAT",
    "CLC",
    "CLCH",
    "CLCL",
    "CLCM",
    "CLCT",
    "CLON",
    "C_T_LK",
    "DBZ",
    "DBZ_850",
    "DBZ_CMAX",
    "DEPTH_LK",
    "DURSUN",
    "DURSUN_M",
    "FOR_D",
    "FRESHSNW",
    "FR_ICE",
    "FR_LAKE",
    "FR_LAND",
    "GRAU_GSP",
    "HBAS_SC",
    "HHL",
    "HSURF",
    "HTOP_SC",
    "HZEROCL",
    "H_ML_LK",
    "H_SNOW",
    "LAI",
    "LCL_ML",
    "LFC_ML",
    "LPI",
    "P",
    "PLCOV",
    "PMSL",
    "PS",
    "QC",
    "QG",
    "QI",
    "QR",
    "QS",
    "QV",
    "RAIN_GSP",
    "RHO_SNOW",
    "ROOTDP",
    "RUNOFF_G",
    "RUNOFF_S",
    "SDI_2",
    "SI",
    "SKC",
    "SLI",
    "SMI",
    "SNOWC",
    "SNOWLMT",
    "SNOW_GSP",
    "SOILTYP",
    "SSO_GAMMA",
    "SSO_SIGMA",
    "SSO_STDH",
    "SSO_THETA",
    "SWISS12",
    "T",
    "TD_2M",
    "TKE",
    "TMAX_2M",
    "TMIN_2M",
    "TOT_PR",
    "TOT_PREC",
    "TQC",
    "TQG",
    "TQI",
    "TQR",
    "TQS",
    "TQV",
    "TWATER",
    "T_2M",
    "T_BOT_LK",
    "T_G",
    "T_MNW_LK",
    "T_SEA",
    "T_SNOW",
    "T_SO",
    "T_WML_LK",
    "U",
    "U_10M",
    "V",
    "VIS",
    "VMAX_10M",
    "V_10M",
    "W",
    "W_I",
    "W_SNOW",
    "W_SO",
    "W_SO_ICE",
    "Z0",
]


def download_func(vars, ref_time, target_dir, add_forecast=False):

    '''
    Function that handels the actual download, without any lat/lon selection.

    Params:
        vars (list): list of variables to download from ICON-CH1
        ref_time (str): reference run of the ICON-CH1 model to download. Can be "latest" or a specific datetime in the format "YYYY-MM-DDTHH:MM:
        target_dir (str): directory to save the downloaded files relative to current working directory
        add_forecast (bool): whether to include forecast data

    returns:
    '''


    lead_times = ["P0DT0H", "P0DT1H", "P0DT2H", "P0DT3H", "P0DT4H", "P0DT5H"]
    reqlist = []
    target_dir = Path.cwd() / target_dir
    
    for var in vars:
        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch1",
            variable=var,
            ref_time=ref_time,
            perturbed=False,
            lead_time=lead_times,
        )
        reqlist.append((req))

    for req in reqlist:
        ogd_api.download_from_ogd(req, target_dir)

    print("\n Downloaded files:")
    for file in sorted(target_dir.iterdir()):
        print(f" - {file.name}")
        




def sel_latlon(target_lat, target_lon, input_grib):
    '''
    Takes in a target lat and lon, and selects the required datapoints from the downloaded files of download_func().
    Adds the model altitude at the selected gridpoint to the xarray.

    Params:
        target_lat (float): lat of interest in degrees
        target_lon (float): lon of interest in degrees
        input_grib (str): name of the grib file
        ouput_nc (str): name of the output netcdf file

    returns: 
        ds: xarray containing the variables found in the grib files at the target latlon

    '''
    ds = get_hor_grid()

    alt, grid_lat, grid_lon, idx = get_alt(ds, target_lat, target_lon)

    input_grib = Path(input_grib)
    #output_nc = Path(output_nc)

    idx_cdo = idx + 1               # CDO uses 1 based indexing python 0 so add 1 on idx

    ds =cdo.selgridcell(
        str(idx_cdo),
        input=str(input_grib),
        options="-s -O -f nc",
        returnXDataset=True,
    )
    ds["model_altitude"] = xr.DataArray(
        alt,
        attrs={
            "units": "m",
            "long_name": "ICON-CH1 model surface altitude at selected grid cell",
        },
    )
    return ds




def sel_level(ds):

    return ds.sel(height=80)





def get_hor_grid():
    '''
    Loads the horizontal constant fields of the ICON-ch1 model. And extracts the model topography.

    returns: ds_ch1_hor: xarray containing the horizontal constant fields ["CLON", "CLAT", "HSURF"] of the ICON-ch1 model
    '''

    url_ch1_hor = ogd_api.get_collection_asset_url(
        collection_id="ch.meteoschweiz.ogd-forecasting-icon-ch1",
        asset_id="horizontal_constants_icon-ch1-eps.grib2"
    )

    ds_ch1_hor = grib_decoder.load(
        source=data_source.URLDataSource(urls=[url_ch1_hor]),
        request={"param": ["CLON", "CLAT", "HSURF"]},
        geo_coords=lambda uuid: {}
    )

    return ds_ch1_hor




def get_alt(ds, target_lat, target_lon):
    """
    Get altitude of an ICON gridpoint.

    Parameters:
        ds (dict): xarrays: ds["HSURF"], ds["CLAT"], ds["CLON"] (use get_hor_grid() function)
        target_lat (float): latitude of interest in degrees
        target_lon (float): longitude of interest in degrees

    returns:
        alt (float): model altitude at the nearest model gridpoint to target lat/lon
        grid_lat, grid_lon (float): lat/lon of the nearest model gridpoint
    """

    hsurf = np.asarray(ds["HSURF"]).squeeze()
    clat = np.asarray(ds["CLAT"]).squeeze()
    clon = np.asarray(ds["CLON"]).squeeze()

    dist = (clat - target_lat) ** 2 + (clon - target_lon) ** 2
    idx = np.nanargmin(dist)

    alt = float(hsurf[idx])
    grid_lat = float(clat[idx])
    grid_lon = float(clon[idx])

    return alt, grid_lat, grid_lon, idx         # CDO uses 1 based indexing. python 0 so add 1 on idx




def download_ICON(target_lat, target_lon, vars, ref_time="latest", add_forecast=False):
    '''
    Main function from the module. Downloads the ICON-CH1 data from the OGD API, 
    selects the nearest gridpoint to the target lat/lon and
    combines the different variables to one xarray dataset.

    Params:
        target_lat (float): lat of interest in degrees
        target_lon (float): lon of interest in degrees
        vars (list): list of variables to download from ICON-CH1
        ref_time (str): "Initialization time of the forecast in UTC, provided as either:
                        - The string "latest" to select the newest forecast run (ref_time) that contains all requested lead times. All assets returned by a single request therefore share the same ref_time. Be cautious: separate requests (for example, for different variables) resolve "latest" independently and may return different ref_time values while MeteoSwiss is uploading a new forecast run.
                        - datetime.datetime object (e.g.,
                        datetime.datetime(2025, 5, 22, 9, 0, 0, tzinfo=datetime.timezone.utc))
                        - ISO 8601 date string (e.g., "2025-05-22T09:00:00Z")"from meteodatalab documentation
        add_forecast (bool): whether to include forecast data

    returns:
        ds_final: xarray containing the variables found in the grib files closest to the target lat/lon  
    '''

    temp_dir = Path.cwd() / "tmp"                   # creating a dir to temporarily store the downloaded grib files in current working dir since the default home dir is to small
    temp_dir.mkdir(parents=True, exist_ok=True)

    datasets = []

    for var in vars:
        if var not in valid_variables:
            raise ValueError(
                f"Variable '{var}' is not included in the ICON-CH1 output. "
                f"Check spelling and choose from: {valid_variables}"
            )

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        tmp = Path(tmp)

        input_dir = tmp / "temporary_ICON"
        output_dir = tmp / "temporary_ICON_point"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        download_func(vars, ref_time, input_dir)

        #output_files = []

        for input_grib in sorted(input_dir.glob("*.grib2")):

            if input_grib.name.startswith(("horizontal_constants", "vertical_constants")):
                continue

            #output_nc = output_dir / f"{input_grib.stem}_point.nc"
            
            print("Processing:", input_grib.name)
            ds = sel_latlon(target_lat=target_lat, target_lon=target_lon, input_grib=input_grib)


            if "height" in ds.dims and ds.sizes["height"] == 80:    # if a variable with multiple vert lvls is processed the lowest/ground lvl is extracted
                ds = ds.isel(height=-1)

            for dim in list(ds.dims):
                if dim != "time" and ds.sizes[dim] == 1:    # at this point all dims should be 1 and except for time can be removed
                    ds = ds.squeeze(dim=dim, drop=True)

            ds = ds.load()

            datasets.append(ds)

        ds_final = xr.combine_by_coords(
            datasets,
            combine_attrs="override",
            compat="no_conflicts",
        )

        ds_final = ds_final.load()

    return ds_final