"""E2 (Week 8): build the wider feature table needed for the rolling-origin backtest.

outputs/03_features_v3.csv only covers 2025-04..2026-05, so earlier cutoffs (test month 2025-11
needs train 2024-11..2025-10) have no data. This rebuilds the SAME pipeline over a wider window:
  mp.load_sold -> mp.valid_rows -> geo.assign_school_districts -> ft.add_engineered_features
No shared module is modified and no existing artifact is overwritten.

ROW ORDER: mp.load_sold finishes with sort_values('CloseDate'), which uses pandas' default
*unstable* quicksort, so loading a wider window permutes rows that share a CloseDate. Row order is
not cosmetic here - RandomForest bootstrap draws and booster subsampling consume rows in order, so
it changes fitted models even at seed 42. The overlap block is therefore emitted in exactly
03_features_v3.csv's order, after asserting the rebuilt content is identical row-set-wise.

HARD GUARDS (any failure exits 1):
  G1 overlap row count == v3 row count
  G2 rebuilt overlap content == v3 content on every feature column (canonical sort on both sides)
  G3 emitted overlap block is byte-identical to v3 (ClosePrice + SchoolDistrict sequence)
  G4 no missing month between MIN_MONTH and MAX_MONTH
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import features as ft, geo, model_pipeline as mp

MIN_MONTH, MAX_MONTH = "2024-11", "2026-05"   # 2024-11 = first train month of the earliest cutoff
OVERLAP_MIN = "2025-04"                        # first month present in 03_features_v3.csv
OUT = Path("outputs/08_features_backtest.csv")

KEY = ["CloseDate", "ClosePrice", "LivingArea", "Bedrooms", "Bathrooms", "LotSize", "YearBuilt",
       "GarageSpaces", "ParkingTotal", "Stories", "PoolPrivateYN", "PostalCode", "Latitude", "Longitude"]
DERIVED = ["SchoolDistrict", "SchoolDistrictName", "SchoolDistrictType", "HighSchoolDistrict",
           "coord_valid", "BedBathRatio", "PropertyAge", "LivingArea_per_Bedroom",
           "Lot_to_Living_ratio", "SaleMonth_sin", "SaleMonth_cos"]

t0 = time.perf_counter()
df = mp.load_sold(min_month=MIN_MONTH, max_month=MAX_MONTH)
print(f"Loaded {len(df):,} single-family rows, {df['Month'].min()} to {df['Month'].max()} "
      f"({time.perf_counter()-t0:.1f}s)")

valid = mp.valid_rows(df)
print(f"Feature-validity rule (LivingArea missing or < {mp.MIN_LIVING_AREA} sqft): "
      f"{int((~valid).sum()):,} rows excluded ({100*(~valid).mean():.3f}%), {int(valid.sum()):,} kept")
df = df.loc[valid].reset_index(drop=True)

districts = geo.load_districts()
print(f"{len(districts)} district polygons ({districts.attrs['n_invalid_repaired']} invalid geometries repaired)")
joined = geo.assign_school_districts(df, districts)
assert len(joined) == len(df), "spatial join changed the row count"
print(f"Spatial join: rows {len(df):,} -> {len(joined):,} (asserted equal)")
print(geo.match_report(joined).to_string(float_format=lambda v: f"{v:,.3f}"))

data = ft.add_engineered_features(joined)
data["Month"] = data["Month"].astype(str)

# County carried from the district boundary file (identity attribute only; the file's
# enrollment/ethnicity/poverty columns stay dropped as in Week 6).
county_map = districts.drop_duplicates("CDSCode").set_index("CDSCode")["CountyName"]
def add_county(d):
    d = d.copy()
    d["County"] = d["SchoolDistrict"].map(county_map).fillna(geo.UNKNOWN).astype(object)
    return d
data = add_county(data)

# ------------------------------------------------------------------ GUARDS vs v3
v3 = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
ov = data.loc[data["Month"] >= OVERLAP_MIN].reset_index(drop=True)
print(f"\nG1 overlap {OVERLAP_MIN}..{MAX_MONTH}: rebuilt {len(ov):,} rows vs v3 {len(v3):,} rows")
if len(ov) != len(v3):
    cmp = pd.concat([ov.groupby("Month").size().rename("rebuilt"),
                     v3.groupby("Month").size().rename("v3")], axis=1).fillna(0).astype(int)
    cmp["diff"] = cmp["rebuilt"] - cmp["v3"]
    print(cmp.to_string()); sys.exit("G1 FAILED: overlap row count differs")
print("G1 PASS")

def canon(d):
    x = d[KEY + DERIVED].copy()
    x["CloseDate"] = pd.to_datetime(x["CloseDate"]).astype("int64")
    return x.sort_values(KEY, kind="mergesort", na_position="last").reset_index(drop=True)

a, b = canon(ov), canon(v3)
mismatch = {}
for c in KEY + DERIVED:
    x, y = a[c], b[c]
    if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
        d = ~np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
    else:
        d = x.astype(str).to_numpy() != y.astype(str).to_numpy()
    if d.sum():
        mismatch[c] = int(d.sum())
if mismatch:
    print(f"G2 FAILED: columns differing after canonical sort: {mismatch}")
    c = next(iter(mismatch))
    i = np.flatnonzero(a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy())[:10]
    print(pd.DataFrame({"row": i, "rebuilt": a[c].to_numpy()[i], "v3": b[c].to_numpy()[i]}).to_string(index=False))
    sys.exit(1)
print(f"G2 PASS: all {len(KEY+DERIVED)} feature columns identical across {len(a):,} overlap rows "
      "(row sets equal; the two loads differ only by a permutation of rows sharing a CloseDate)")

# ------------------------------------------------------------------ emit in v3 order for the overlap
earlier = data.loc[data["Month"] < OVERLAP_MIN].copy()
overlap = add_county(v3.drop(columns=["split", "val_split"]))
out = pd.concat([earlier[overlap.columns], overlap], ignore_index=True)

ov_block = out.loc[out["Month"] >= OVERLAP_MIN].reset_index(drop=True)
assert np.array_equal(ov_block["ClosePrice"].to_numpy(float), v3["ClosePrice"].to_numpy(float)), "G3 ClosePrice"
assert np.array_equal(ov_block["SchoolDistrict"].astype(str).to_numpy(),
                      v3["SchoolDistrict"].astype(str).to_numpy()), "G3 SchoolDistrict"
print(f"G3 PASS: emitted overlap block is byte-identical to 03_features_v3.csv "
      f"({len(ov_block):,} rows, ClosePrice + SchoolDistrict sequences exact)")

months = pd.period_range(MIN_MONTH, MAX_MONTH, freq="M").astype(str)
missing = sorted(set(months) - set(out["Month"]))
assert not missing, f"G4 FAILED: missing months {missing}"
print(f"G4 PASS: all {len(months)} months present, {MIN_MONTH}..{MAX_MONTH}")

print(f"\nCounty coverage: {100*out['County'].ne(geo.UNKNOWN).mean():.3f}% named, "
      f"{out['County'].nunique()} distinct")
print("\nRows per month:")
print(out.groupby("Month").size().rename("rows").to_string())

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"\nE2 PASSED. Wrote {OUT} ({len(out):,} rows x {out.shape[1]} cols, "
      f"{out['Month'].min()}..{out['Month'].max()}) in {time.perf_counter()-t0:.1f}s")
