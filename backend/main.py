import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from shapely.geometry import mapping, shape
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

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

# ---------- Live forecast (Phase 7) ----------

LIVE_CACHE_SECONDS = 30 * 60
_live_cache = {}

WEATHER_LABELS = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow",
    75: "Heavy snow", 80: "Light showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


@lru_cache(maxsize=1)
def centres():
    """One point inside each district, used for the live forecast."""
    with open(RAW / "gadm41_IND_2.json", encoding="utf-8") as f:
        g = json.load(f)
    out = {}
    for ft in g["features"]:
        pt = shape(ft["geometry"]).representative_point()
        out[ft["properties"]["GID_2"]] = (round(pt.y, 4), round(pt.x, 4))
    return out


@app.get("/live/{gid}")
def live(gid: str):
    now = time.time()
    cached = _live_cache.get(gid)
    if cached and now - cached[0] < LIVE_CACHE_SECONDS:
        return cached[1]

    if gid not in centres():
        raise HTTPException(404, f"Unknown district: {gid}")
    lat, lon = centres()[gid]

    params = urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urlopen(url, timeout=10) as r:
            data = json.load(r)
    except Exception:
        raise HTTPException(503, "Forecast service unavailable. Try again later.")

    cur = data.get("current", {})
    daily = data.get("daily", {})
    days = []
    for i, day in enumerate(daily.get("time", [])):
        code = daily["weather_code"][i]
        days.append({
            "date": day,
            "max_c": daily["temperature_2m_max"][i],
            "min_c": daily["temperature_2m_min"][i],
            "rain_mm": daily["precipitation_sum"][i],
            "condition": WEATHER_LABELS.get(code, "Unknown"),
        })

    code = cur.get("weather_code")
    result = {
        "GID_2": gid,
        "latitude": lat,
        "longitude": lon,
        "current": {
            "time": cur.get("time"),
            "temp_c": cur.get("temperature_2m"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "rain_mm": cur.get("precipitation"),
            "condition": WEATHER_LABELS.get(code, "Unknown"),
        },
        "daily": days,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cache_minutes": LIVE_CACHE_SECONDS // 60,
        "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)",
    }
    _live_cache[gid] = (now, result)
    return result



# ---------- Landslides (Phase 5) ----------

def _ls_file(kind):
    matches = sorted(PROC.glob(f"landslide_{kind}_*.csv"))
    if not matches:
        raise HTTPException(503, f"Landslide file '{kind}' not found in processed folder")
    return matches[0]


@lru_cache(maxsize=1)
def ls_yearly():
    return pd.read_csv(_ls_file("district_yearly"))


@lru_cache(maxsize=1)
def ls_events():
    return pd.read_csv(PROC / "landslide_events_india.csv", parse_dates=["event_date"])


@app.get("/map/landslides")
def map_landslides(year_from: int = Query(2007, ge=2007, le=2017),
                   year_to: int = Query(2017, ge=2007, le=2017)):
    y = ls_yearly()
    y = y[(y.year >= year_from) & (y.year <= year_to)]
    out = y.groupby("GID_2", as_index=False)["events"].sum()
    return clean(out)


@app.get("/landslides/{gid}")
def landslides(gid: str, year_from: int = Query(2007, ge=2007, le=2017),
               year_to: int = Query(2017, ge=2007, le=2017)):
    y = ls_yearly()
    y = y[(y.GID_2 == gid) & (y.year >= year_from) & (y.year <= year_to)]
    if y.empty:
        raise HTTPException(404, f"No landslide record for {gid}")
    ev = ls_events()
    ev = ev[(ev.GID_2 == gid) & (ev.year >= year_from) & (ev.year <= year_to)]
    recent = ev.sort_values("event_date", ascending=False).head(15).copy()
    recent = ev.sort_values("event_date", ascending=False).head(15).copy()
    recent["event_date"] = recent["event_date"].dt.strftime("%Y-%m-%d")
    recent["landslide_category"] = recent["landslide_category"].fillna("unknown")
    return {
        "GID_2": gid,
        "state": y.state.iloc[0],
        "district": y.district.iloc[0],
        "total_events": int(y.events.sum()),
        "by_year": clean(y[["year", "events"]]),
        "recent_events": clean(recent[["event_date", "landslide_category"]]),
    }