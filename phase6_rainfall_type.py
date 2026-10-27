import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(r"C:\p1")
RAW = BASE / "raw"
PROC = BASE / "processed"
SEED = 42
K_RANGE = range(3, 7)
PILOT_STATE = "UttarPradesh"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# 1. Load rainfall and keep only districts with a complete record
rain = pd.read_csv(RAW / "district_monthly_rainfall_1901_2025.csv")
rain["match_key"] = rain.state.map(key) + "|" + rain.district.map(key)

q = pd.read_csv(PROC / "district_data_quality.csv")
q = q[q.status == "ok"][["match_key", "NAME_1", "NAME_2"]]

# Add GID_2 from the boundary file (the quality file does not have it)
import json
with open(RAW / "gadm41_IND_2.json", encoding="utf-8") as f:
    g = json.load(f)
gid = pd.DataFrame([{"GID_2": ft["properties"]["GID_2"],
                     "match_key": key(ft["properties"]["NAME_1"]) + "|" + key(ft["properties"]["NAME_2"])}
                    for ft in g["features"]]).drop_duplicates("match_key")
q = q.merge(gid, on="match_key", how="left")

rain = rain.merge(q, on="match_key", how="inner")

# 2. Annual totals; drop all-zero years (missing data, as in Phase 2)
ann = rain.groupby(["GID_2", "NAME_1", "NAME_2", "year"], as_index=False).rain_mm.sum()
ann = ann.rename(columns={"rain_mm": "annual_mm"})
ann = ann[ann.annual_mm > 0]

# ---------- OPTION 1: rainfall type (statistical grouping) ----------

def monthly_profile(y0, y1):
    """Each district's average share of annual rain falling in each month."""
    s = rain[rain.year.between(y0, y1)]
    m = s.groupby(["GID_2", "month"]).rain_mm.mean().unstack()
    m = m.dropna()
    share = m.div(m.sum(axis=1), axis=0)
    share.columns = [f"m{c:02d}" for c in share.columns]
    return share


def cluster(share, k=None):
    X = StandardScaler().fit_transform(share.values)
    if k is not None:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        return km.labels_, None
    best = None
    for kk in K_RANGE:
        km = KMeans(n_clusters=kk, n_init=20, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_)
        print(f"  k={kk}: silhouette={sil:.3f}")
        if best is None or sil > best[1]:
            best = (kk, sil, km.labels_)
    return best[2], best[0]


# Main grouping: 1951-2020, all districts
share_main = monthly_profile(1951, 2020)
labels, best_k = cluster(share_main)
print("\nChosen number of rainfall types (highest silhouette):", best_k)

types = share_main.reset_index()[["GID_2"]].copy()
types["rainfall_type"] = labels + 1
types = types.merge(q[["GID_2", "NAME_1", "NAME_2"]], on="GID_2")
types = types.rename(columns={"NAME_1": "state", "NAME_2": "district"})
types.to_csv(PROC / "rainfall_type_districts.csv", index=False)

# Describe each type so it can be named from its profile
prof = share_main.copy()
prof["rainfall_type"] = labels + 1
summary = prof.groupby("rainfall_type").mean()
summary["districts"] = prof.groupby("rainfall_type").size()
summary["peak_month"] = summary[[f"m{i:02d}" for i in range(1, 13)]].idxmax(axis=1).str[1:].astype(int).map(lambda i: MONTHS[i - 1])
summary["share_jun_sep_pct"] = (summary[["m06", "m07", "m08", "m09"]].sum(axis=1) * 100).round(1)
summary.round(3).to_csv(PROC / "rainfall_type_profile.csv")
print("\nType profile:")
print(summary[["districts", "peak_month", "share_jun_sep_pct"]].to_string())

# Stability: cluster two halves separately and compare labels
half_a = monthly_profile(1951, 1985)
half_b = monthly_profile(1986, 2020)
la, _ = cluster(half_a, k=best_k)
lb, _ = cluster(half_b, k=best_k)
common = half_a.index.intersection(half_b.index)
la_s = pd.Series(la, index=half_a.index).loc[common]
lb_s = pd.Series(lb, index=half_b.index).loc[common]
ari = adjusted_rand_score(la_s, lb_s)
print(f"\nStability, 1951-1985 vs 1986-2020 (adjusted Rand index, 1 = identical): {ari:.3f}")

# Pilot slice: Uttar Pradesh
pilot = types[types.state.map(key) == key(PILOT_STATE)]
print(f"\nPilot, {PILOT_STATE}: {len(pilot)} districts")
print(pilot.rainfall_type.value_counts().sort_index().to_string())

# ---------- OPTION 2: unusually wet or dry years ----------

ref = ann[ann.year.between(1991, 2020)]
bands = ref.groupby("GID_2").annual_mm.quantile([0.10, 0.90]).unstack()
bands.columns = ["p10_mm", "p90_mm"]
flags = ann.merge(bands, left_on="GID_2", right_index=True, how="inner")
flags["flag"] = np.select(
    [flags.annual_mm > flags.p90_mm, flags.annual_mm < flags.p10_mm],
    ["unusually wet", "unusually dry"],
    default="within normal range")
flags = flags.round(1)
flags.to_csv(PROC / "rainfall_unusual_years.csv", index=False)

# Check: about 10% of 1991-2020 years should fall outside each band by construction
ref_flags = flags[flags.year.between(1991, 2020)].flag.value_counts(normalize=True).round(3)
print("\nShare of 1991-2020 years by flag (expect about 0.10 wet, 0.10 dry):")
print(ref_flags.to_string())

print("\nSaved:")
print(" ", PROC / "rainfall_type_districts.csv")
print(" ", PROC / "rainfall_type_profile.csv")
print(" ", PROC / "rainfall_unusual_years.csv")
print("Done.")