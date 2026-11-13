"""E1 (Week 8): HARD GATE. Reproduce outputs/week7_comparison.csv metrics from the saved
test predictions at full precision, then write outputs/metrics_summary.csv.

The 2026-05 test month is NOT re-scored: Week 7 scored each frozen config once and saved the
predictions to scratch/week7_test_predictions.csv. This script only recomputes metrics from them.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import features as ft

MODELS = ["LR", "DT", "RF", "LightGBM", "XGBoost"]
METRICS = ["R2", "MAE", "RMSE", "MAPE", "MdAPE"]
TOL = 1e-9  # relative; CSV round-trip of full repr should be exact to ~1e-15

preds = pd.read_csv("scratch/week7_test_predictions.csv")
comp = pd.read_csv("outputs/week7_comparison.csv").set_index("model")
v3 = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
test = v3.loc[v3["split"].eq("test")].reset_index(drop=True)

# --- Alignment check: saved predictions must line up row-for-row with the test split ---
assert len(preds) == len(test), f"row count mismatch: preds {len(preds):,} vs test {len(test):,}"
y_test = test["ClosePrice"].to_numpy(float)
max_price_diff = float(np.max(np.abs(preds["ClosePrice"].to_numpy(float) - y_test)))
assert max_price_diff == 0.0, f"ClosePrice sequence differs (max abs diff {max_price_diff})"
print(f"Alignment OK: {len(test):,} test rows (2026-05), ClosePrice sequence identical (max diff {max_price_diff}).")

# --- Recompute metrics and compare at full precision ---
rows, failures = [], []
for m in MODELS:
    got = ft.full_metrics(y_test, preds[m].to_numpy(float))
    exp = comp.loc[m]
    diffs = {}
    for k in METRICS:
        d = abs(got[k] - float(exp[k]))
        rel = d / max(abs(float(exp[k])), 1e-12)
        diffs[k] = (got[k], float(exp[k]), d, rel)
        if rel > TOL:
            failures.append((m, k, got[k], float(exp[k]), d, rel))
    worst = max(diffs.values(), key=lambda t: t[3])
    print(
        f"{m:9s} R2={got['R2']:.10f} MAE=${got['MAE']:,.4f} RMSE=${got['RMSE']:,.4f} "
        f"MAPE={got['MAPE']:.8%} MdAPE={got['MdAPE']:.8%} | max rel diff vs week7 = {worst[3]:.2e}"
    )
    rows.append({
        "model": m,
        "feature_set": exp["feature_set"],
        "categorical_handling": exp["categorical_handling"],
        "objective": exp["objective"],
        "n_estimators": int(exp["best_iteration"]),
        "n_features": int(exp["n_features"]),
        "test_month": "2026-05",
        "n_test": len(test),
        **{k: got[k] for k in METRICS},
    })

if failures:
    print("\nGATE FAILED — differences beyond full precision:")
    for f in failures:
        print(f"  {f[0]} {f[1]}: recomputed {f[2]!r} vs week7_comparison {f[3]!r} (abs {f[4]:.6g}, rel {f[5]:.3e})")
    sys.exit(1)

summary = pd.DataFrame(rows)
summary.to_csv("outputs/metrics_summary.csv", index=False)
print("\nGATE PASSED: all 5 models reproduce week7_comparison.csv within 1e-9 relative.")
print("Wrote outputs/metrics_summary.csv")
print(summary[["model", "feature_set", "n_test"] + METRICS].to_string(index=False))
