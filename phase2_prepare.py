"""Phase 2: clean and prepare rainfall data (district level, India)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\p1")
RAW = BASE / "raw"
OUT = BASE / "processed"
OUT.mkdir(exist_ok=True)


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# 2.1 Load and check
df = pd.read_csv(RAW / "district_monthly_rainfall_1901_2025.csv")
assert len(df) == 973500, len(df)
assert df.isna().sum().sum() == 0

# 2.2 Standardise names and match to boundaries
df["match_key"] = df["state"].map(key) + "|" + df["district"].map(key)
with open(RAW / "gadm41_IND_2.json", encoding="utf-8") as f:
    g = json.load(f)
geo = pd.DataFrame([{"GID_2": ft["properties"]["GID_2"],
                     "NAME_1": ft["properties"]["NAME_1"],
                     "NAME_2": ft["properties"]["NAME_2"]} for ft in g["features"]])
geo["match_key"] = geo["NAME_1"].map(key) + "|" + geo["NAME_2"].map(key)
unmatched = geo[~geo.match_key.isin(set(df.match_key))].copy()
unmatched["status"] = "no rainfall rows"
unmatched.to_csv(OUT / "district_name_map_unmatched.csv", index=False)

# 2.2b Zero rain in all 12 months of a year = missing data, not a dry year
zero_year = df.groupby(["match_key", "year"]).rain_mm.transform("sum") == 0
df.loc[zero_year, ["rain_mm", "dry_days", "days"]] = np.nan

# 2.3 Dry fraction (dry_days definition still to confirm with IMD)
df["dry_fraction"] = df["dry_days"] / df["days"]

# 2.6 Annual totals (a year is kept only if all 12 months are valid)
annual = (df.groupby(["match_key", "state", "district", "year"])
            .agg(rain_mm=("rain_mm", lambda s: s.sum(min_count=12)),
                 dry_days=("dry_days", lambda s: s.sum(min_count=12)),
                 days=("days", lambda s: s.sum(min_count=12)))
            .reset_index())
annual.to_csv(OUT / "rainfall_annual.csv", index=False)

# 2.7 Seasonal totals
df["season"] = np.select(
    [df.month.between(3, 5), df.month.between(6, 9), df.month.between(10, 11)],
    ["pre_monsoon", "monsoon", "post_monsoon"], default="winter")
df["season_year"] = np.where(df.month == 12, df.year + 1, df.year)
seasonal = (df[df.season_year.between(1901, 2025)]
            .groupby(["match_key", "season_year", "season"])
            .agg(rain_mm=("rain_mm", lambda s: s.sum(min_count=1)))
            .reset_index().rename(columns={"season_year": "year"}))
seasonal.to_csv(OUT / "rainfall_seasonal.csv", index=False)

# 2.8 Climatology 1991-2020 by district and calendar month
ref = df[df.year.between(1991, 2020)]
clim = (ref.groupby(["match_key", "month"])
           .agg(rain_mean=("rain_mm", "mean"), rain_sd=("rain_mm", "std"),
                dry_days_mean=("dry_days", "mean"))
           .reset_index())
clim.to_csv(OUT / "rainfall_climatology_1991_2020.csv", index=False)

# 2.9 Average of the previous five valid years
piv = annual.pivot(index="year", columns="match_key", values="rain_mm")
total = sum(piv.shift(k) for k in range(1, 6))
count = sum(piv.shift(k).notna().astype(int) for k in range(1, 6))
avg5 = (total / 5).where(count == 5).stack().rename("avg_5yr_prior").reset_index()
avg5.columns = ["year", "match_key", "avg_5yr_prior"]
avg5.round(2).to_csv(OUT / "rainfall_5yr_average.csv", index=False)

# Data quality per district
valid = annual.dropna(subset=["rain_mm"])
q = (valid.groupby("match_key")
          .agg(valid_years=("year", "count"), first_valid=("year", "min"),
               last_valid=("year", "max")).reset_index())
q = geo[["match_key", "NAME_1", "NAME_2"]].drop_duplicates("match_key").merge(q, on="match_key", how="left")
q["valid_years"] = q["valid_years"].fillna(0).astype(int)
q["status"] = np.select([q.valid_years == 0, q.valid_years < 30],
                        ["no_data", "short_record_lt_30y"], default="ok")
q.to_csv(OUT / "district_data_quality.csv", index=False)

# 2.10 Dry-region metrics (training years 1901-2015)
train = annual[annual.year.between(1901, 2015)].dropna(subset=["rain_mm"])
p25 = train.groupby("match_key")["rain_mm"].quantile(0.25).rename("p25_rain_mm")
avg_dry = train.groupby("match_key")["dry_days"].mean().rename("avg_dry_days_train")
mean_rain = train.groupby("match_key")["rain_mm"].mean().rename("mean_rain_train")
dry_chance = (annual.dropna(subset=["rain_mm"]).merge(p25, on="match_key")
                     .assign(below=lambda x: x.rain_mm < x.p25_rain_mm)
                     .groupby("match_key")["below"].mean().rename("dry_chance_all_years"))
met = pd.concat([avg_dry, mean_rain, p25, dry_chance], axis=1)
met = met.dropna(subset=["avg_dry_days_train"]).reset_index()
norm = lambda s: (s - s.min()) / (s.max() - s.min())
met["dry_days_norm"] = norm(met.avg_dry_days_train)
met["inv_rain_norm"] = 1 - norm(met.mean_rain_train)
met["dryness_score"] = (0.5 * met.dry_days_norm + 0.5 * met.inv_rain_norm).round(4)
met.round(4).to_csv(OUT / "dry_region_metrics.csv", index=False)

# Checks
print("districts:", df.match_key.nunique(), "| unmatched boundaries:", len(unmatched))
print("quality status:", q.status.value_counts().to_dict())
print("annual rows with data:", int(annual.rain_mm.notna().sum()), "of", len(annual))
print("dry_chance mean (expect ~0.25):", round(met.dry_chance_all_years.mean(), 3))
print("Done. Files written to", OUT)