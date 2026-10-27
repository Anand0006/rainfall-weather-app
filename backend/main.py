import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from shapely.geometry import mapping, shape

BASE = Path(__file__).resolve().parents[1]   # C:\p1
RAW = BASE / "raw"
PROC = BASE / "processed"
HERE = Path(__file__).resolve().parent

app = FastAPI(title="Weather App API", version="0.4")


def make_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def clean(df):
    """Convert to JSON-safe records (NaN becomes null)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


@lru_cache(maxsize=1)
def quality():
    return pd.read_csv(PROC / "district_data_quality.csv")

@lru_cache(maxsize=1)
def annual():
    return pd.read_csv(PROC / "rainfall_annual.csv")

@lru_cache(maxsize=1)
def climatology():
    return pd.read_csv(PROC / "rainfall_climatology_1991_2020.csv")

@lru_cache(maxsize=1)
def avg5():
    return pd.read_csv(PROC / "rainfall_5yr_average.csv")

@lru_cache(maxsize=1)
def dry():
    return pd.read_csv(PROC / "dry_region_metrics.csv")

@lru_cache(maxsize=1)
def temp():
    return pd.read_csv(PROC / "temperature_district_monthly_1951_2025.csv")

@lru_cache(maxsize=1)
def monthly():
    df = pd.read_csv(RAW / "district_monthly_rainfall_1901_2025.csv")
    df["match_key"] = df["state"].map(make_key) + "|" + df["district"].map(make_key)
    return df[["match_key", "year", "month", "rain_mm"]]

@lru_cache(maxsize=1)
def boundaries():
    """District polygons, simplified so the map loads quickly."""
    with open(RAW / "gadm41_IND_2.json", encoding="utf-8") as f:
        g = json.load(f)
    feats = []
    for ft in g["features"]:
        p = ft["properties"]
        geom = shape(ft["geometry"]).simplify(0.02, preserve_topology=True)
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "GID_2": p["GID_2"],
                "state": p["NAME_1"],
                "district": p["NAME_2"],
                "match_key": f"{make_key(p['NAME_1'])}|{make_key(p['NAME_2'])}",
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def get_district(key):
    q = quality()
    row = q[q.match_key == key]
    if row.empty:
        raise HTTPException(404, f"District not found: {key}")
    return row.iloc[0]


def annual_value(key, year):
    a = annual()
    r = a[(a.match_key == key) & (a.year == year)]
    if r.empty or pd.isna(r.rain_mm.iloc[0]):
        return None
    return float(r.rain_mm.iloc[0])


# ---------- Web page ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "index.html")


# ---------- Data endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/boundaries")
def get_boundaries():
    return boundaries()


@app.get("/districts")
def districts(include_no_data: bool = False):
    q = quality()
    if not include_no_data:
        q = q[q.status != "no_data"]
    out = q.rename(columns={"NAME_1": "state", "NAME_2": "district"})
    return clean(out[["match_key", "state", "district", "status", "valid_years"]])


@app.get("/rainfall/{key}")
def rainfall(key: str, year: int | None = None):
    get_district(key)
    if year is None:
        a = annual()
        a = a[(a.match_key == key) & a.rain_mm.notna()]
        return {"match_key": key, "years": clean(a[["year", "rain_mm"]])}
    total = annual_value(key, year)
    if total is None:
        raise HTTPException(404, f"No valid data for {key} in {year}")
    m = monthly()
    mm = m[(m.match_key == key) & (m.year == year)]
    return {"match_key": key, "year": year, "annual_mm": round(total, 1),
            "monthly": clean(mm[["month", "rain_mm"]])}


@app.get("/compare/{key}")
def compare(key: str, year: int = Query(..., ge=1901, le=2025)):
    get_district(key)
    this = annual_value(key, year)
    if this is None:
        raise HTTPException(404, f"No valid data for {key} in {year}")
    last = annual_value(key, year - 1)

    c = climatology()
    normal = float(c[c.match_key == key].rain_mean.sum())

    a5 = avg5()
    r5 = a5[(a5.match_key == key) & (a5.year == year)]
    avg_5yr = float(r5.avg_5yr_prior.iloc[0]) if not r5.empty else None

    dep = (this - normal) / normal * 100 if normal > 0 else None
    if dep is None:
        label = "unknown"
    elif dep > 10:
        label = "above normal"
    elif dep < -10:
        label = "below normal"
    else:
        label = "near normal"

    return {
        "match_key": key,
        "year": year,
        "this_year_mm": round(this, 1),
        "last_year_mm": round(last, 1) if last is not None else None,
        "avg_5yr_prior_mm": round(avg_5yr, 1) if avg_5yr is not None else None,
        "normal_1991_2020_mm": round(normal, 1),
        "departure_from_normal_pct": round(dep, 1) if dep is not None else None,
        "status": label,
    }


@app.get("/dry-regions")
def dry_regions(top: int = Query(20, ge=1, le=100)):
    names = quality()[["match_key", "NAME_1", "NAME_2"]]
    d = dry().merge(names, on="match_key", how="left")
    d = d.sort_values("dryness_score", ascending=False).head(top)
    d = d.rename(columns={"NAME_1": "state", "NAME_2": "district"})
    return clean(d[["match_key", "state", "district", "dryness_score",
                    "dry_chance_all_years", "mean_rain_train"]])


@app.get("/map/rainfall")
def map_rainfall(year: int = Query(..., ge=1901, le=2025)):
    a = annual()
    a = a[(a.year == year) & a.rain_mm.notna()]
    return clean(a[["match_key", "rain_mm"]].round(1))


@app.get("/map/dryness")
def map_dryness():
    return clean(dry()[["match_key", "dryness_score", "dry_chance_all_years"]])


@app.get("/map/temperature")
def map_temperature(year: int = Query(..., ge=1951, le=2025),
                    month: int | None = Query(None, ge=1, le=12)):
    t = temp()
    t = t[t.year == year]
    if month is not None:
        t = t[t.month == month]
        out = t[["GID_2", "tmax_mean"]]
    else:
        out = t.groupby("GID_2", as_index=False)["tmax_mean"].mean().round(2)
    return clean(out)


@app.get("/temperature/{gid}")
def temperature(gid: str, year: int | None = None):
    t = temp()
    t = t[t.GID_2 == gid]
    if t.empty:
        raise HTTPException(404, f"No temperature data for {gid}")
    name = t.iloc[0]
    if year is not None:
        t = t[t.year == year]
    return {"GID_2": gid, "state": name["state"], "district": name["district"],
            "rows": clean(t[["year", "month", "tmax_mean"]])}