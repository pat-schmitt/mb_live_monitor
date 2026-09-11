import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path



from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_mass_balance_hef(
    stake_file_path,
    ds,
    variable_name="MB",
    start_year=None,
    end_year=None,
    balance_codes=("annual", "winter"),
    plot=True,
):
    """
    Compare Hintereisferner point mass-balance observations against COSIPY.

    The observation file is expected to contain:
        year, begin_date, end_date, latitude, longitude,
        elevation, balance, balance_code

    COSIPY output should contain:
        variable_name, usually "MB"
        time, lat, lon
    """

    # ---------------------------------------------------------
    # 1. Read observations
    # ---------------------------------------------------------
    stake_file_path = Path(stake_file_path)

    obs = pd.read_csv(
        stake_file_path,
        parse_dates=["begin_date", "end_date"],
    )

    required = [
        "year",
        "begin_date",
        "end_date",
        "latitude",
        "longitude",
        "elevation",
        "balance",
        "balance_code",
    ]

    missing = [c for c in required if c not in obs.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {stake_file_path}: {missing}"
        )

    # ---------------------------------------------------------
    # 2. Select years / balance types
    # ---------------------------------------------------------
    if start_year is not None:
        obs = obs[obs["year"] >= start_year]

    if end_year is not None:
        obs = obs[obs["year"] <= end_year]

    obs = obs[
        obs["balance_code"].isin(balance_codes)
    ].copy()

    # ---------------------------------------------------------
    # 3. COSIPY time coverage
    # ---------------------------------------------------------
    model_start = pd.Timestamp(ds.time.min().values)
    model_end = pd.Timestamp(ds.time.max().values)

    print("COSIPY period:")
    print(model_start, "to", model_end)

    # ---------------------------------------------------------
    # 4. Compare every stake measurement
    # ---------------------------------------------------------
    rows = []

    for _, r in obs.iterrows():

        t1 = pd.Timestamp(r["begin_date"])
        t2 = pd.Timestamp(r["end_date"])

        # end_date is a date, so include the whole final day
        t2_full = (
            t2
            + pd.Timedelta(days=1)
            - pd.Timedelta(nanoseconds=1)
        )

        # Skip observations whose complete measurement
        # period is not covered by COSIPY
        if t1 < model_start or t2_full > model_end:
            continue

        stake_lat = float(r["latitude"])
        stake_lon = float(r["longitude"])

        da = ds[variable_name]

        # -----------------------------------------------------
        # Select nearest COSIPY grid cell
        # -----------------------------------------------------
        if (
            "lat" in da.dims
            and "lon" in da.dims
            and ds.lat.ndim == 1
            and ds.lon.ndim == 1
        ):
            cell = da.sel(
                lat=stake_lat,
                lon=stake_lon,
                method="nearest",
            )

            model_lat = float(cell.lat.values)
            model_lon = float(cell.lon.values)

            if "HGT" in ds:
                model_hgt = float(
                    ds["HGT"]
                    .sel(
                        lat=stake_lat,
                        lon=stake_lon,
                        method="nearest",
                    )
                    .values
                )
            else:
                model_hgt = np.nan

        else:
            raise ValueError(
                "This function currently expects 1-D lat/lon "
                "coordinates in the COSIPY output."
            )

        # -----------------------------------------------------
        # Select measurement period
        # -----------------------------------------------------
        period = cell.sel(
            time=slice(t1, t2_full)
        )

        if period.time.size == 0:
            continue

        # Sum COSIPY mass balance over period
        model_balance = float(
            period.sum(skipna=False).values
        )

        rows.append(
            {
                "year": int(r["year"]),
                "stake_id": r.get("original_id", np.nan),
                "balance_code": r["balance_code"],

                "begin_date": t1,
                "end_date": t2,

                "latitude": stake_lat,
                "longitude": stake_lon,
                "stake_elevation": float(r["elevation"]),

                "model_lat": model_lat,
                "model_lon": model_lon,
                "model_elevation": model_hgt,

                "measured": float(r["balance"]),
                "modeled": model_balance,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        print("No complete overlapping observation periods.")
        return None

    # ---------------------------------------------------------
    # 5. Errors
    # ---------------------------------------------------------
    result["error"] = (
        result["modeled"] - result["measured"]
    )

    result["abs_error"] = np.abs(result["error"])

    # ---------------------------------------------------------
    # 6. Statistics
    # ---------------------------------------------------------
    print("\nMASS BALANCE COMPARISON")

    for code in balance_codes:

        sub = result[
            result["balance_code"] == code
        ]

        if sub.empty:
            continue

        rmse = np.sqrt(
            np.mean(
                (sub["modeled"] - sub["measured"]) ** 2
            )
        )

        mae = np.mean(
            np.abs(
                sub["modeled"] - sub["measured"]
            )
        )

        bias = np.mean(
            sub["modeled"] - sub["measured"]
        )

        if len(sub) >= 2:
            corr = np.corrcoef(
                sub["measured"],
                sub["modeled"],
            )[0, 1]
        else:
            corr = np.nan

        print(f"\n{code.upper()}")
        print(f"N:    {len(sub)}")
        print(f"RMSE: {rmse:.3f} m w.e.")
        print(f"MAE:  {mae:.3f} m w.e.")
        print(f"Bias: {bias:.3f} m w.e.")
        print(f"Corr: {corr:.3f}")

    # ---------------------------------------------------------
    # 7. Plot
    # ---------------------------------------------------------
    if plot:

        fig, ax = plt.subplots(figsize=(7, 7))

        for code in balance_codes:

            sub = result[
                result["balance_code"] == code
            ]

            if len(sub) == 0:
                continue

            ax.scatter(
                sub["measured"],
                sub["modeled"],
                label=code,
                alpha=0.7,
            )

        values = np.concatenate(
            [
                result["measured"].values,
                result["modeled"].values,
            ]
        )

        lo = np.nanmin(values)
        hi = np.nanmax(values)

        ax.plot(
            [lo, hi],
            [lo, hi],
            "--k",
            label="1:1",
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_xlabel("Measured mass balance (m w.e.)")
        ax.set_ylabel("COSIPY mass balance (m w.e.)")

        ax.set_title(
            "Hintereisferner: COSIPY vs observations"
        )

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    return result




def select_nearest_point(cosipy_out, lat=None, lon=None):
    """
    Select the nearest grid point in COSIPY output dataset based on latitude and longitude.

    Parameters:
    cosipy_out (xr.Dataset): COSIPY output.
    lat (float, optional): Latitude to select nearest grid point.
    lon (float, optional): Longitude to select nearest grid point.

    Returns:
    dict: Dictionary with selected lat, lon, and elevation (HGT).
    dict: Selection dictionary for xarray operations.
    """
    if lat is not None and lon is not None:
        sel_dict = dict(lat=lat, lon=lon, method="nearest")
    else:
        sel_dict = dict(lat=cosipy_out.lat[0], lon=cosipy_out.lon[0])

    selected_lat = float(cosipy_out.lat.sel(lat=sel_dict['lat'], method="nearest").values)
    selected_lon = float(cosipy_out.lon.sel(lon=sel_dict['lon'], method="nearest").values)
    hgt = float(cosipy_out["HGT"].sel(**sel_dict).values)

    return {"lat": selected_lat, "lon": selected_lon, "HGT": hgt}, sel_dict


def comp_aws_2_cosipy_output(aws, cosipy_out, readable_name, start_year=None, end_year=None, lat=None, lon=None):
    """
    Plot comparison of AWS data and COSIPY output with optional per-year alignment (for height vars).
    Computes RMSE strictly over the *displayed time window*.
    Inputs are never modified.

    Parameters
    ----------
    aws : pandas.DataFrame
        Must contain TIMESTAMP and the mapped AWS column below.
    cosipy_out : xarray.Dataset
        COSIPY output dataset.
    readable_name : str
        One of: TOTALHEIGHT, SNOWHEIGHT, ALBEDO, TS, MB, RRR, RAIN, SNOWFALL, LWin, LWout
    start_year, end_year : int or None
        Start/end years for filtering/plot window. If only start_year given, plots that single year.
    lat, lon : float or None
        Location for selecting the nearest COSIPY grid point.
    """

 
    variable_dict = {
        "TOTALHEIGHT": {"aws_var": "HS_sel",         "cosipy_var": "TOTALHEIGHT"},
        "SNOWHEIGHT":  {"aws_var": "HS_sel",         "cosipy_var": "SNOWHEIGHT"},
        "ALBEDO":      {"aws_var": "Albedo_acc",     "cosipy_var": "ALBEDO"},
        "TS":          {"aws_var": "Ts",             "cosipy_var": "TS"},
        "MB":          {"aws_var": "nan",            "cosipy_var": "MB"},
        "RRR":         {"aws_var": "TotalPrecipmm",  "cosipy_var": "RRR"},   # both in mm
        "RAIN":        {"aws_var": "Rainfallmweq",   "cosipy_var": "RAIN"},  # AWS m w.e., COSIPY mm
        "SNOWFALL":    {"aws_var": "Snowfallmweq",   "cosipy_var": "SNOWFALL"},
        "LWin":        {"aws_var": "lw_in",          "cosipy_var": "LWin"},
        "LWout":       {"aws_var": "lw_out",         "cosipy_var": "LWout"},
    }
    if readable_name not in variable_dict:
        print(f"Variable '{readable_name}' not found in the mapping.")
        return

    aws_var_name    = variable_dict[readable_name]["aws_var"]
    cosipy_var_name = variable_dict[readable_name]["cosipy_var"]

    
    aws_plot = aws.copy()
    if "nan" not in aws_plot.columns:
        aws_plot["nan"] = np.nan

    location_info, sel_dict = select_nearest_point(cosipy_out, lat, lon)
    selected_lat = location_info["lat"]
    selected_lon = location_info["lon"]
    hgt          = location_info["HGT"]
    cosipy_data  = cosipy_out[cosipy_var_name].sel(**sel_dict).to_dataframe().reset_index()

    
    overall_rmse = None
    yearly_rmse  = []

    if readable_name in ["SNOWHEIGHT", "TOTALHEIGHT"]:
        if "TIMESTAMP" not in aws_plot or aws_var_name not in aws_plot:
            print("Missing TIMESTAMP or required AWS column.")
            return
        aws_plot["Year"] = aws_plot["TIMESTAMP"].dt.year

        # years list
        if start_year is not None and end_year is not None:
            years = range(start_year, end_year + 1)
        elif start_year is not None:
            years = [start_year]
        else:
            years = aws_plot["Year"].dropna().unique()

        aligned = []
        for year in years:
            g = aws_plot[aws_plot["Year"] == year].copy()
            summer_g = g.dropna(subset=[aws_var_name])
            if summer_g.empty:
                continue
            first_date = summer_g["TIMESTAMP"].iloc[0]
            c0 = cosipy_data.loc[cosipy_data["time"] == first_date, cosipy_var_name].values
            if len(c0) > 0:
                adj = c0[0] - summer_g[aws_var_name].iloc[0]
                g.loc[:, aws_var_name] = g[aws_var_name] + adj

                tmp = (
                    pd.merge(g, cosipy_data, left_on="TIMESTAMP", right_on="time", how="inner")
                      .dropna(subset=[aws_var_name, cosipy_var_name])
                )
                if not tmp.empty:
                    rmse_y = float(np.sqrt(mean_squared_error(tmp[aws_var_name], tmp[cosipy_var_name])))
                    yearly_rmse.append((year, rmse_y))
            aligned.append(g)

        aws_aligned = pd.concat(aligned) if aligned else aws_plot.copy()

    elif readable_name == "MB":
        aws_aligned = aws_plot.copy()

    else:
        aws_aligned = aws_plot.copy()

    
    # converting the rain in the aws from mweq to mm
    if readable_name == "RAIN" and aws_var_name in aws_aligned.columns:
        aws_aligned = aws_aligned.copy()
        aws_aligned[aws_var_name] = aws_aligned[aws_var_name] * 1000.0

    
    if start_year is not None and end_year is not None:
        start_date = pd.Timestamp(f"{start_year}-01-01"); end_date = pd.Timestamp(f"{end_year}-12-31")
    elif start_year is not None:
        start_date = pd.Timestamp(f"{start_year}-01-01"); end_date = pd.Timestamp(f"{start_year}-12-31")
    else:
        start_date = end_date = None

    aws_win    = aws_aligned
    cosipy_win = cosipy_data
    if start_date is not None and end_date is not None:
        aws_win    = aws_aligned[(aws_aligned["TIMESTAMP"] >= start_date) & (aws_aligned["TIMESTAMP"] <= end_date)]
        cosipy_win = cosipy_data[(cosipy_data["time"]      >= start_date) & (cosipy_data["time"]      <= end_date)]

    
    if aws_var_name in aws_win.columns:
        merged = (
            pd.merge(aws_win, cosipy_win, left_on="TIMESTAMP", right_on="time", how="inner")
              .dropna(subset=[aws_var_name, cosipy_var_name])
        )
        if not merged.empty:
            overall_rmse = float(np.sqrt(mean_squared_error(merged[aws_var_name], merged[cosipy_var_name])))

   
    legend_loc = f"lat={selected_lat:.3f}, lon={selected_lon:.3f}, HGT={hgt:.1f} m"

    plt.figure(figsize=(12, 6))
    aws_series = aws_win[aws_var_name] if aws_var_name in aws_win.columns else pd.Series(index=aws_win.index, dtype=float)
    if readable_name == "LWout":
        aws_series = -aws_series  # sign convention for plotting only

    plt.plot(aws_win["TIMESTAMP"], aws_series, label=f"{aws_var_name} (AWS)", linewidth=1, alpha=0.7)
    plt.plot(cosipy_win["time"], cosipy_win[cosipy_var_name], label=f"{cosipy_var_name} (COSIPY)", linewidth=1, alpha=0.7)

    
    all_vals = pd.concat([aws_series, cosipy_win[cosipy_var_name]], ignore_index=True)
    if np.isfinite(all_vals.to_numpy()).any():
        plt.ylim(all_vals.min(), all_vals.max())

    
    units = "mm" if readable_name == "RAIN" else cosipy_out[cosipy_var_name].attrs.get("units", "")
    plt.xlabel("Time")
    plt.ylabel(f"{readable_name} ({units})")
    plt.title(f"Comparison AWS input and COSIPY output at {hgt:.1f} m\n(Location: lat={selected_lat:.3f}, lon={selected_lon:.3f})")
    plt.legend(title=legend_loc)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # print metrics
    if readable_name in ["SNOWHEIGHT", "TOTALHEIGHT"]:
        for y, v in yearly_rmse:
            print(f"{y}: RMSE = {v:.4f}")
    if overall_rmse is not None:
        print(f"Overall RMSE of {readable_name}: {overall_rmse:.4f}")

        
def _default_latlon(ds):
        lat = float(np.atleast_1d(ds.lat.values)[0])
        lon = float(np.atleast_1d(ds.lon.values)[0])
        return lat, lon

def _parse_latlon(latlon, ds):
        if latlon is None:
            return _default_latlon(ds)
        if isinstance(latlon, dict):
            if "lat" in latlon and "lon" in latlon:
                return float(latlon["lat"]), float(latlon["lon"])
            raise ValueError("latlon dict must have keys 'lat' and 'lon'.")
        if isinstance(latlon, (list, tuple)) and len(latlon) == 2:
            return float(latlon[0]), float(latlon[1])  # [lat, lon]
        raise ValueError("latlon must be {'lat':..., 'lon':...} or [lat, lon].")

def compare_outputs(dataset1, dataset2=None, dataset3=None, dataset4=None,
                    variable_name=None,
                    latlon1=None, latlon2=None, latlon3=None, latlon4=None,
                    labels=None, start_year=None, end_year=None):
    """
    Compare a specific variable from up to 4 datasets and plot them.

    - If only start_year is given, plot that single calendar year.
    - If start_year and end_year are given, plot the inclusive window.
    - If neither is given, plot the full range.

    """

    if not variable_name:
        raise ValueError("Please provide variable_name")

    vn = variable_name.strip()
    lower = vn.lower()
    use_cum = False
    use_mean = False
    while lower.startswith("cum") or lower.startswith("mean") or lower.startswith("_"):
        if lower.startswith("_"):
            vn = vn[1:]; lower = lower[1:]
            continue
        if lower.startswith("cum"):
            use_cum = True
            vn = vn[3:]; lower = lower[3:]
            if lower.startswith("_"):
                vn = vn[1:]; lower = lower[1:]
            continue
        if lower.startswith("mean"):
            use_mean = True
            vn = vn[4:]; lower = lower[4:]
            if lower.startswith("_"):
                vn = vn[1:]; lower = lower[1:]
            continue
    base_var = vn
    if not base_var:
        raise ValueError("After removing prefixes, variable name is empty.")

    ds_list  = [dataset1, dataset2, dataset3, dataset4]
    ll_list  = [latlon1,  latlon2,  latlon3,  latlon4]

    for i in range(4):
        if ds_list[i] is None and (ll_list[i] is not None):
            if dataset1 is None:
                raise ValueError("No dataset provided to reuse for extra locations.")
            ds_list[i] = dataset1

    # Labels
    default_labels = [f"Dataset {i+1}" for i in range(4)]
    label_list = default_labels if labels is None else (list(labels) + default_labels)[:4]

   
    if start_year is not None:
        sy = int(start_year)
        ey = int(end_year) if end_year is not None else sy
        start_date = pd.Timestamp(f"{sy}-01-01")
        end_date   = pd.Timestamp(f"{ey}-12-31")
    else:
        start_date = end_date = None

    prepared = []
    for i in range(4):
        ds = ds_list[i]
        if ds is None or base_var not in ds:
            continue

        lat, lon = _parse_latlon(ll_list[i], ds)  # helper must exist in your environment
        sel = dict(lat=lat, lon=lon, method="nearest")

        try:
            lat_sel = float(ds.lat.sel(lat=sel["lat"], method="nearest").values)
            lon_sel = float(ds.lon.sel(lon=sel["lon"], method="nearest").values)
        except Exception:
            continue

        hgt = None
        if "HGT" in ds:
            try:
                hgt = float(ds["HGT"].sel(**sel).values)
            except Exception:
                hgt = None

        df = ds[base_var].sel(**sel).to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])

        
        if start_date is not None:
            df = df[df["time"] >= start_date]
        if end_date is not None:
            df = df[df["time"] <= end_date]

        if df.empty:
            continue

        if use_cum:
            df[base_var] = df[base_var].cumsum()

        prepared.append({"df": df, "lat": lat_sel, "lon": lon_sel, "hgt": hgt, "idx": i})

    if not prepared:
        print("No valid series to plot.")
        return

    
    units = ""
    for ds in ds_list:
        if ds is not None and base_var in ds and hasattr(ds[base_var], "attrs"):
            units = ds[base_var].attrs.get("units", "")
            if units:
                break

    colors = ["blue", "orange", "violet", "green"]

    plt.figure(figsize=(12, 6))
    for item in prepared:
        idx = item["idx"]
        base_label = label_list[idx]
        if item["hgt"] is not None:
            lbl = f"{base_label} ({item['hgt']:.0f}m, {item['lat']:.3f}, {item['lon']:.3f})"
        else:
            lbl = f"{base_label} ({item['lat']:.3f}, {item['lon']:.3f})"
        plt.plot(item["df"]["time"], item["df"][base_var],
                 label=lbl, color=colors[idx], linewidth=.9, alpha=0.8)

    if use_mean:
        for item in prepared:
            idx = item["idx"]
            m = item["df"][base_var].mean()
            plt.axhline(m, color=colors[idx], linestyle='--',
                        label=f"Mean {label_list[idx]}: {m:.2f}")

    
    if start_year is not None:
        window_txt = f" ({int(start_year)}" + (f"–{int(end_year)}" if end_year is not None else "") + ")"
    else:
        window_txt = ""

    title_var = base_var
    if use_cum and use_mean:
        title_var = f"cum(mean({base_var}))"
    elif use_cum:
        title_var = f"cum({base_var})"
    elif use_mean:
        title_var = f"mean({base_var})"

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{base_var}" + (f" ({units})" if units else ""), fontsize=12)
    plt.title(f"Comparison of {title_var}{window_txt}", fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def compare_variables(dataset1, variable_name1, variable_name2, year=None, lat=None, lon=None):
    """
    Compare two variables from a dataset and plot them with dual y-axes, with an option to filter by year and location.

    Parameters:
    dataset1 (xr.Dataset): Dataset containing the variables.
    variable_name1 (str): First variable name to compare.
    variable_name2 (str): Second variable name to compare.
    year (int, optional): Year to filter the data. If None, the full range is used.
    lat (float, optional): Latitude to select nearest grid point.
    lon (float, optional): Longitude to select nearest grid point.

    Returns:
    None
    """
   
    if lat is not None and lon is not None:
        sel_dict = dict(lat=lat, lon=lon, method="nearest")
        
    else:
        sel_dict = dict(lat=dataset1.lat[0], lon=dataset1.lon[0])

    
    selected_lat = float(dataset1.lat.sel(lat=sel_dict['lat'], method="nearest").values)
    selected_lon = float(dataset1.lon.sel(lon=sel_dict['lon'], method="nearest").values)

    
    hgt = float(dataset1["HGT"].sel(**sel_dict).values)

    if variable_name1 in dataset1:
        data1 = dataset1[variable_name1].sel(**sel_dict).to_dataframe().reset_index()
    else:
        print(f"Variable '{variable_name1}' not found in the dataset.")
        return

    if variable_name2 in dataset1:
        data2 = dataset1[variable_name2].sel(**sel_dict).to_dataframe().reset_index()
    else:
        print(f"Variable '{variable_name2}' not found in the dataset.")
        return

    
    if year is not None:
        data1["time"] = pd.to_datetime(data1["time"])
        data2["time"] = pd.to_datetime(data2["time"])
        data1 = data1[data1["time"].dt.year == year]
        data2 = data2[data2["time"].dt.year == year]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = "blue"
    color2 = "orange"

    ax1.set_xlabel("Time", fontsize=12)
    ax1.grid(True, axis="both", linestyle="--", alpha=0.5)
    ax1.set_ylabel(f"{variable_name1} ({dataset1[variable_name1].attrs.get('units', '')})", color=color1, fontsize=12)
    ax1.plot(data1["time"], data1[variable_name1], label=f"{variable_name1}", color=color1, linewidth=0.9, alpha=0.8)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{variable_name2} ({dataset1[variable_name2].attrs.get('units', '')})", color=color2, fontsize=12)
    ax2.plot(data2["time"], data2[variable_name2], label=f"{variable_name2}", color=color2, linewidth=0.9, alpha=0.8)
    ax2.tick_params(axis='y')

    # Final plot title with variable names, elevation, and location
    plt.title(f'Comparison of {variable_name1} and {variable_name2} at {hgt:.1f} m\n(Location: lat={selected_lat:.3f}, lon={selected_lon:.3f})', fontsize=14)
    fig.tight_layout()
    plt.show()

