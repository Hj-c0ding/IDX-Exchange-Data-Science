"""Diagnose the E2 guard failure: is the overlap a true content difference or only a row permutation?"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import features as ft, geo, model_pipeline as mp

v3 = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})

df = mp.load_sold(min_month="2024-11", max_month="2026-05")
df = df.loc[mp.valid_rows(df)].reset_index(drop=True)
joined = geo.assign_school_districts(df, geo.load_districts())
data = ft.add_engineered_features(joined)
data["Month"] = data["Month"].astype(str)
ov = data.loc[data["Month"] >= "2025-04"].reset_index(drop=True)

KEY = ["CloseDate", "ClosePrice", "LivingArea", "Bedrooms", "Bathrooms", "LotSize", "YearBuilt",
       "GarageSpaces", "ParkingTotal", "Stories", "PoolPrivateYN", "PostalCode", "Latitude", "Longitude"]
COMPARE = KEY + ["SchoolDistrict", "SchoolDistrictName", "SchoolDistrictType", "HighSchoolDistrict",
                 "coord_valid", "BedBathRatio", "PropertyAge", "LivingArea_per_Bedroom",
                 "Lot_to_Living_ratio", "SaleMonth_sin", "SaleMonth_cos"]

def canon(d):
    x = d[COMPARE].copy()
    x["CloseDate"] = pd.to_datetime(x["CloseDate"]).astype("int64")
    return x.sort_values(KEY, kind="mergesort", na_position="last").reset_index(drop=True)

a, b = canon(ov), canon(v3)
print(f"rebuilt overlap {len(a):,} rows | v3 {len(b):,} rows | identical shape: {a.shape == b.shape}")

# 1) Is every differing row only a reordering within the SAME CloseDate?
ne_raw = ov["ClosePrice"].to_numpy() != v3["ClosePrice"].to_numpy()
same_date = (pd.to_datetime(ov["CloseDate"]).to_numpy() == pd.to_datetime(v3["CloseDate"]).to_numpy())
print(f"rows differing in the ORIGINAL order: {int(ne_raw.sum()):,}; "
      f"of those, CloseDate still matches position-for-position: {int((ne_raw & same_date).sum()):,} "
      f"({'ALL' if (ne_raw & ~same_date).sum() == 0 else 'NOT all'})")
print(f"per-month row counts equal: {ov.groupby('Month').size().equals(v3.groupby('Month').size())}")

# 2) Content comparison after canonical sorting
mism = {}
for c in COMPARE:
    x, y = a[c], b[c]
    if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
        d = ~(np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True))
    else:
        d = (x.astype(str).to_numpy() != y.astype(str).to_numpy())
    if d.sum():
        mism[c] = int(d.sum())
print(f"\ncolumns differing after canonical sort: {mism if mism else 'NONE — row sets are identical'}")
if mism:
    c = next(iter(mism)); idx = np.flatnonzero(a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy())[:10]
    print(pd.DataFrame({"row": idx, "rebuilt": a[c].to_numpy()[idx], "v3": b[c].to_numpy()[idx]}).to_string(index=False))
