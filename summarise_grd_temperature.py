import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- SETTINGS ----------
TEMP_DIR = r"C:\p1\raw\Tempreture raw data"
BOUNDARY_FILE = r"C:\p1\raw\gadm41_IND_2.json"
OUT_DIR = r"C:\p1\processed"
START_YEAR, END_YEAR = 1951, 2025

NLAT, NLON = 31, 31
LAT = np.arange(7.5, 7.5 + NLAT, 1.0)     # 7.5 to 37.5
LON = np.arange(67.5, 67.5 + NLON, 1.0)   # 67.5 to 97.5
FILL_VALUES = (99.9, -999.0)              # IMD missing-value codes
VALID_MIN, VALID_MAX = -50.0, 60.0        # plausible range (deg C); winter minimums can go below -5
# ------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def var_from_name(path):
    # Check the file name first, then the folder name
    for text in (os.path.basename(path).lower(), os.path.dirname(path).lower()):
        if "max" in text:
            return "tmax"
        if "min" in text:
            return "tmin"
    return None 

def year_from_name(path):
    m = re.search(r"(19|20)\d{2}", os.path.basename(path))
    return int(m.group(0)) if m else None


def read_grd(path):
    raw = np.fromfile(path, dtype="<f4")
    cells = NLAT * NLON
    if raw.size % cells != 0:
        raise ValueError(f"{os.path.basename(path)}: {raw.size} values do not fit a "
                         f"{NLAT}x{NLON} grid")
    days = raw.size // cells
    arr = raw.reshape(days, NLAT, NLON).astype("float64")
    for fv in FILL_VALUES:
        arr[np.isclose(arr, fv, atol=0.05)] = np.nan
    arr[(arr < VALID_MIN) | (arr > VALID_MAX)] = np.nan
    return arr, days


# ---- 1. Find .grd files (including subfolders) ----
files = sorted(glob.glob(os.path.join(TEMP_DIR, "**", "*.grd"), recursive=True))
files += sorted(glob.glob(os.path.join(TEMP_DIR, "**", "*.GRD"), recursive=True))
files = sorted(set(files))
print("GRD files found:", len(files))
if not files:
    raise SystemExit("No .grd files found. Check TEMP_DIR.")

# ---- 2. Read each year into daily arrays ----
daily = {"tmax": {}, "tmin": {}}
for f in files:
    var, yr = var_from_name(f), year_from_name(f)
    if var is None or yr is None:
        print("  skipped (name has no max/min or year):", os.path.basename(f))
        continue
    if not (START_YEAR <= yr <= END_YEAR):
        continue
    if yr in daily[var]:
        print("  duplicate file skipped:", os.path.basename(f))
        continue
    arr, days = read_grd(f)
    leap = (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0))
    expected = 366 if leap else 365
    if days != expected:
        print(f"  WARNING {os.path.basename(f)}: {days} days, expected {expected}")
    t = pd.date_range(f"{yr}-01-01", periods=days, freq="D")
    daily[var][yr] = xr.DataArray(arr, coords={"time": t, "lat": LAT, "lon": LON},
                                  dims=("time", "lat", "lon"))

for v in daily:
    years = sorted(daily[v])
    print(f"{v}: {len(years)} years loaded")
    if years:
        missing = [y for y in range(START_YEAR, END_YEAR + 1) if y not in years]
        print(f"  missing years: {missing if missing else 'none'}")

# ---- 3. Join years and convert daily to monthly means ----
data_vars = {v: xr.concat([daily[v][y] for y in sorted(daily[v])], dim="time")
             for v in daily if daily[v]}
ds = xr.Dataset(data_vars)
monthly = ds.resample(time="MS").mean(skipna=True)
print("Monthly range:", str(monthly.time.values[0])[:7], "to",
      str(monthly.time.values[-1])[:7])

# ---- 4. Grid CSV (one row per grid cell per month) ----
grid = monthly.to_dataframe().reset_index()
grid = grid.dropna(how="all", subset=list(monthly.data_vars)).round(2)
grid_path = os.path.join(OUT_DIR, "temperature_grid_monthly_1951_2025.csv")
grid.to_csv(grid_path, index=False)
print("Saved grid CSV:", grid_path, "rows:", len(grid))

# ---- 5. Find which grid cells lie inside each district ----
districts = gpd.read_file(BOUNDARY_FILE).reset_index(drop=True)
n_dist = len(districts)

rr, cc = np.meshgrid(np.arange(NLAT), np.arange(NLON), indexing="ij")
cells = gpd.GeoDataFrame(
    {"j": rr.ravel(), "k": cc.ravel()},
    geometry=[Point(LON[c], LAT[r]) for r, c in zip(rr.ravel(), cc.ravel())],
    crs="EPSG:4326")
joined = gpd.sjoin(cells, districts[["geometry"]], how="inner", predicate="within")
print("Grid cells matched to a district:", len(joined))
points = districts.geometry.representative_point()

# ---- 6. District monthly averages (keyed by GID_2) ----
times = pd.to_datetime(monthly.time.values)
arrays = {v: monthly[v].values for v in monthly.data_vars}   # (time, lat, lon)

frames = []
no_cells = 0
for i, row in districts.iterrows():
    sel = joined[joined["index_right"] == i]
    if len(sel) > 0:
        js, ks = sel["j"].values, sel["k"].values
        method = "grid_mean"
    else:
        no_cells += 1
        method = "nearest_cell"
        p = points.iloc[i]
        js = np.array([int(np.argmin((LAT - p.y) ** 2))])
        ks = np.array([int(np.argmin((LON - p.x) ** 2))])

    df = pd.DataFrame({
        "GID_2": row["GID_2"],
        "state": row["NAME_1"],
        "district": row["NAME_2"],
        "year": times.year,
        "month": times.month,
        "method": method,
    })
    for v, arr in arrays.items():
        vals = arr[:, js, ks]
        with np.errstate(all="ignore"):
            df[f"{v}_mean"] = np.round(np.nanmean(vals, axis=1), 2)
    frames.append(df)

    if i % 50 == 0:
        print("  district", i, "of", n_dist)

district_df = pd.concat(frames, ignore_index=True)
dist_path = os.path.join(OUT_DIR, "temperature_district_monthly_1951_2025.csv")
district_df.to_csv(dist_path, index=False)
print("Districts using nearest cell:", no_cells, "of", n_dist)
print("Saved district CSV:", dist_path, "rows:", len(district_df))
print("Done.")