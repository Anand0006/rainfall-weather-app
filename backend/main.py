import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

BASE = Path(__file__).resolve().parents[1]   # C:\p1
RAW = BASE / "raw"
PROC = BASE / "processed"

app = FastAPI(title="Weather App API", version="0.3")


def make_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


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
def monthly():
    df = pd.read_csv(RAW / "district_monthly_rainfall_1901_2025.csv")
    df["match_key"] = df["state"].map(make_key) + "|" + df["district"].map(make_key)
    return df[["match_key", "year", "month", "rain_mm"]]


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/districts")
def districts(include_no_data: bool = False):
    q = quality()
    if not include_no_data:
        q = q[q.status != "no_data"]
    out = q.rename(columns={"NAME_1": "state", "NAME_2": "district"})
    return out[["match_key", "state", "district", "status", "valid_years"]].to_dict("records")


@app.get("/rainfall/{key}")
def rainfall(key: str, year: int | None = None):
    get_district(key)
    if year is None:
        a = annual()
        a = a[(a.match_key == key) & a.rain_mm.notna()]
        return {"match_key": key, "years": a[["year", "rain_mm"]].to_dict("records")}
    total = annual_value(key, year)
    if total is None:
        raise HTTPException(404, f"No valid data for {key} in {year}")
    m = monthly()
    mm = m[(m.match_key == key) & (m.year == year)]
    return {"match_key": key, "year": year, "annual_mm": round(total, 1),
            "monthly": mm[["month", "rain_mm"]].to_dict("records")}


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
    return d[["match_key", "state", "district", "dryness_score",
              "dry_chance_all_years", "mean_rain_train"]].to_dict("records")


@app.get("/map/rainfall")
def map_rainfall(year: int = Query(..., ge=1901, le=2025)):
    a = annual()
    a = a[(a.year == year) & a.rain_mm.notna()]
    return a[["match_key", "rain_mm"]].round(1).to_dict("records")