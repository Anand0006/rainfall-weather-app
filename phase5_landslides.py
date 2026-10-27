import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", message=".*geographic CRS.*")

BASE = Path(r"C:\p1")
RAW = BASE / "raw"
PROC = BASE / "processed"
LANDSLIDE_FILE = RAW / "Global_Landslide_Catalog_Export_rows.csv"
BOUNDARY_FILE = RAW / "gadm41_IND_2.json"
SOURCE = "NASA GLC (1988-2017)"
MAX_NEAREST_M = 5000   # 5 km tolerance for events just outside a boundary
# Albers equal-area projection centred on India, in metres
ALBERS = "+proj=aea +lat_1=12 +lat_2=32 +lat_0=22 +lon_0=82 +datum=WGS84 +units=m +no_defs"

PROC.mkdir(exist_ok=True)

# 1. Load and parse
df = pd.read_csv(LANDSLIDE_FILE)
print("Rows in file:", len(df))
df["event_date"] = pd.to_datetime(df["event_date"],
                                  format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
print("Unparsed dates:", int(df.event_date.isna().sum()))
df = df.dropna(subset=["latitude", "longitude", "event_date"]).reset_index(drop=True)

# 2. Rough India box (includes neighbouring countries; boundaries remove them)
ind = df[df.latitude.between(6, 38) & df.longitude.between(68, 98)].copy()
print("Events inside India box:", len(ind))

pts = gpd.GeoDataFrame(ind, geometry=gpd.points_from_xy(ind.longitude, ind.latitude),
                       crs="EPSG:4326")
dist = gpd.read_file(BOUNDARY_FILE)[["GID_2", "NAME_1", "NAME_2", "geometry"]]

# 3. Events inside a district polygon
inside = gpd.sjoin(pts, dist, how="inner", predicate="within")
inside = inside[~inside.index.duplicated(keep="first")]
inside["match"] = "inside"

# 4. Events outside every polygon: nearest district within 5 km, measured in metres
outside = pts.drop(index=inside.index)
near = gpd.sjoin_nearest(outside.to_crs(ALBERS), dist.to_crs(ALBERS),
                         how="inner", max_distance=MAX_NEAREST_M)
near = near[~near.index.duplicated(keep="first")].to_crs("EPSG:4326")
near["match"] = "nearest"

events = pd.concat([inside, near])
print("Matched inside a district:", int((events.match == "inside").sum()))
print("Matched to nearest district (within 5 km):", int((events.match == "nearest").sum()))
print("Outside all Indian districts (other countries or offshore):", len(pts) - len(events))

# 5. Event-level file
events = events.rename(columns={"NAME_1": "state", "NAME_2": "district"})
events["year"] = events.event_date.dt.year
events["month"] = events.event_date.dt.month
events["source"] = SOURCE
event_cols = ["GID_2", "state", "district", "event_date", "year", "month",
              "landslide_category", "latitude", "longitude", "match", "source"]
events[event_cols].sort_values("event_date").to_csv(
    PROC / "landslide_events_india.csv", index=False)

# 6. Monthly counts: every district x every month in the record (zeros included)
start = events.event_date.min().to_period("M")
end = events.event_date.max().to_period("M")
months = pd.period_range(start, end, freq="M")
ym = pd.DataFrame({"year": months.year, "month": months.month})
gids = pd.DataFrame({"GID_2": dist.GID_2.unique()})
grid = gids.merge(ym, how="cross")

counts = (events.groupby(["GID_2", "year", "month"]).size()
                .rename("events").reset_index())
monthly = grid.merge(counts, on=["GID_2", "year", "month"], how="left")
monthly["events"] = monthly["events"].fillna(0).astype(int)
names = dist[["GID_2", "NAME_1", "NAME_2"]].rename(
    columns={"NAME_1": "state", "NAME_2": "district"})
monthly = monthly.merge(names, on="GID_2", how="left")
monthly["source"] = SOURCE
monthly = monthly[["GID_2", "state", "district", "year", "month", "events", "source"]]
monthly.to_csv(PROC / "landslide_district_monthly_1988_2017.csv", index=False)
print("Monthly rows:", len(monthly), "| expected:", len(gids) * len(ym))

# 7. Yearly totals per district
yearly = (monthly.groupby(["GID_2", "state", "district", "year"], as_index=False)
                 ["events"].sum())
yearly.to_csv(PROC / "landslide_district_yearly_1988_2017.csv", index=False)

# 8. Checks
print("\nEvents per year (last 5 years):")
print(events.groupby("year").size().tail(5).to_string())
print("\nTop 10 districts by total events:")
print(events.groupby(["state", "district"]).size()
            .sort_values(ascending=False).head(10).to_string())
print("\nDone. Files written to", PROC)