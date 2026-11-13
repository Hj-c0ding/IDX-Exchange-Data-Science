"""E4 (Week 8): rolling-origin backtest at 3 earlier cutoffs.

Protocol per cutoff: train on the 12 calendar months immediately BEFORE the cutoff month, score the
cutoff month. Train-only price bounds are refit inside each window; every learned step stays inside
the sklearn Pipeline; seed 42.

HYPERPARAMETERS ARE FROZEN, NOT RE-TUNED. They are read back from outputs/week6_validation.csv and
outputs/week7_validation.csv using the same selection rule as scratch/phase3_evaluate.py, i.e. they
were chosen on the SINGLE validation month 2026-04. This backtest therefore measures STABILITY OVER
TIME of a fixed configuration - it is NOT independent per-cutoff tuning, and the 2026-04 cutoff is
the tuning month itself (optimistic, reported separately).

ITERATION-SCALING RULE (boosters, same rule as Week 7):
    n_estimators(window) = round(best_iteration_val * n_train(window) / n_val_train)
with best_iteration_val the early-stopped iteration on 2026-04 and n_val_train = 127,734 the
price-filtered row count of the validation training window (2025-04..2026-03).

GUARDS: G5 the 2026-04 cutoff must reproduce week6_validation.csv exactly for LR/DT/RF (identical
rows in identical order); leakage guard: stop if any MdAPE < 5% or > 30% better than 7.38%.
"""
import json, os, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import boosting as bt, features as ft, model_pipeline as mp

SEED = 42
np.random.seed(SEED)
N_VAL_TRAIN = 127734          # price-filtered rows of the 2025-04..2026-03 validation training window
TRAIN_MONTHS = 12
MIN_FS_GAIN_MDAPE = 0.0010    # same rule as phase3_evaluate.py
BEST_PRIOR_MDAPE = 0.0737896551724137   # XGBoost F0, test 2026-05 (Week 7)
CUTOFFS = ["2025-11", "2026-02", "2026-04"]
TUNING_MONTH = "2026-04"

data = pd.read_csv("outputs/08_features_backtest.csv",
                   dtype={"PostalCode": "object", "SchoolDistrict": "object", "County": "object"})
print(f"Backtest table: {len(data):,} rows, {data['Month'].min()}..{data['Month'].max()}")

# ---- Frozen configs, read back from the Week 6/7 validation tables (never retyped) --------------
w6 = pd.read_csv("outputs/week6_validation.csv")
w6 = w6.loc[~w6["district_encoding"].str.startswith("onehot_mf")]
w7 = pd.read_csv("outputs/week7_validation.csv")

gbm_cfg = {}
for lib in ["LightGBM", "XGBoost"]:
    sub = w7.loc[w7["library"].eq(lib)]
    best_by_fs = sub.sort_values(["val_MdAPE", "val_MAE"]).groupby("feature_set").head(1).set_index("feature_set")
    b0 = best_by_fs.loc["F0"]
    alt = best_by_fs.loc[[s for s in best_by_fs.index if s != "F0"]].sort_values(["val_MdAPE", "val_MAE"]).iloc[0]
    adopt = (b0["val_MdAPE"] - alt["val_MdAPE"] >= MIN_FS_GAIN_MDAPE) and (alt["val_MAE"] < b0["val_MAE"])
    w = alt if adopt else b0
    fs = w.name if adopt else "F0"
    gbm_cfg[lib] = {"feature_set": fs, "categorical_handling": w["categorical_handling"],
                    "objective": w["objective"], "val_best_iteration": int(w["best_iteration"]),
                    "params": json.loads(w["params"])}
    print(f"Frozen {lib}: {fs}, {w['categorical_handling']}, {w['objective']}, "
          f"val_best_iteration={int(w['best_iteration'])}, params={gbm_cfg[lib]['params']}")

SPECS = [
    {"model": "LR", "feature_set": "F3", "kind": "lr",
     "build": lambda n: ft.make_model("LR", "F3", district_encoding="onehot"), "iters": lambda n: 1},
    {"model": "DT", "feature_set": "F0", "kind": "tree",
     "build": lambda n: ft.make_model("DT", "F0", max_depth=18, min_samples_leaf=5), "iters": lambda n: 1},
    {"model": "RF", "feature_set": "F0", "kind": "tree",
     "build": lambda n: ft.make_model("RF", "F0", min_samples_leaf=3), "iters": lambda n: 200},
]
for lib in ["LightGBM", "XGBoost"]:
    c = gbm_cfg[lib]
    SPECS.append({
        "model": lib, "feature_set": c["feature_set"], "kind": "tree",
        "iters": (lambda n, c=c: bt.scale_iterations(c["val_best_iteration"], n, N_VAL_TRAIN)),
        "build": (lambda n, l=lib, c=c: bt.make_boosting_model(
            l, c["feature_set"], categorical_handling=c["categorical_handling"],
            objective=c["objective"], n_estimators=bt.scale_iterations(c["val_best_iteration"], n, N_VAL_TRAIN),
            **c["params"])),
    })

rows = []
for cutoff in CUTOFFS:
    months = [str(p) for p in pd.period_range(end=pd.Period(cutoff, "M") - 1, periods=TRAIN_MONTHS, freq="M")]
    tr_all = data.loc[data["Month"].isin(months)]
    te = data.loc[data["Month"].eq(cutoff)]
    tr, bounds = mp.filter_train_prices(tr_all)
    y_tr, y_te = tr[mp.TARGET].to_numpy(float), te[mp.TARGET].to_numpy(float)
    print(f"\n=== cutoff {cutoff}{'  [TUNING MONTH - optimistic]' if cutoff == TUNING_MONTH else ''} ===")
    print(f"train {months[0]}..{months[-1]}: {len(tr_all):,} rows -> {len(tr):,} after train-only price "
          f"bounds ${bounds[0]:,.0f}-${bounds[1]:,.0f} ({len(tr_all)-len(tr):,} dropped) | test {cutoff}: {len(te):,}")

    for spec in SPECS:
        cols = ft.input_columns(spec["feature_set"], spec["kind"])
        n_est = spec["iters"](len(tr))
        est = spec["build"](len(tr))
        t0 = time.perf_counter()
        est.fit(tr[cols], y_tr)
        fit_s = time.perf_counter() - t0
        m = ft.full_metrics(y_te, est.predict(te[cols]))

        assert m["MdAPE"] >= 0.05, f"STOP leakage guard: {spec['model']} @ {cutoff} MdAPE {m['MdAPE']:.2%} < 5%"
        assert m["MdAPE"] >= BEST_PRIOR_MDAPE * 0.70, \
            f"STOP leakage guard: {spec['model']} @ {cutoff} MdAPE {m['MdAPE']:.2%} >30% better than 7.38%"

        rows.append({"cutoff": cutoff, "is_tuning_month": cutoff == TUNING_MONTH,
                     "train_start": months[0], "train_end": months[-1],
                     "n_train_raw": len(tr_all), "n_train": len(tr), "n_test": len(te),
                     "price_lower": float(bounds[0]), "price_upper": float(bounds[1]),
                     "model": spec["model"], "feature_set": spec["feature_set"], "n_estimators": n_est,
                     "scored_in_week": 8, **m, "fit_s": fit_s})
        print(f"  {spec['model']:9s} ({spec['feature_set']}, iter={n_est:5d}): R2={m['R2']:7.4f} "
              f"MAE=${m['MAE']:>9,.0f} RMSE=${m['RMSE']:>10,.0f} MAPE={m['MAPE']:6.2%} "
              f"MdAPE={m['MdAPE']:6.2%} ({fit_s:5.1f}s)")

bt_df = pd.DataFrame(rows)

# ---- G5: the 2026-04 cutoff must reproduce the Week 6 validation numbers for LR/DT/RF -----------
print("\nG5 (2026-04 cutoff vs outputs/week6_validation.csv, identical rows in identical order):")
fails = []
for model, fs in [("LR", "F3"), ("DT", "F0"), ("RF", "F0")]:
    got = bt_df[(bt_df.cutoff == "2026-04") & (bt_df.model == model)].iloc[0]
    ref = w6[(w6.model == model) & (w6.feature_set == fs)].iloc[0]
    dm, da = abs(got["MdAPE"] - ref["MdAPE"]), abs(got["MAE"] - ref["MAE"])
    ok = dm < 1e-12 and da < 1e-6
    fails += [] if ok else [f"{model} {fs}: MdAPE {got['MdAPE']!r} vs {ref['MdAPE']!r}, MAE {got['MAE']!r} vs {ref['MAE']!r}"]
    print(f"  {model:3s} {fs}: MdAPE {got['MdAPE']:.6%} vs {ref['MdAPE']:.6%} (d={dm:.2e}) | "
          f"MAE ${got['MAE']:,.2f} vs ${ref['MAE']:,.2f} (d={da:.2e}) -> {'PASS' if ok else 'FAIL'}")
if fails:
    print("G5 FAILED:"); [print("  -", f) for f in fails]; sys.exit(1)
print("G5 PASS (boosters are expected to differ: scaled iterations, not early stopping)")

# ---- Append the held-out test month as reference (NOT re-scored; Week 7 numbers) ----------------
# Window size and price bounds are recomputed from the Week 7 training split, never typed in.
v3 = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
w7_train, w7_bounds = mp.filter_train_prices(v3.loc[v3["split"].eq("train")])
ms = pd.read_csv("outputs/metrics_summary.csv")
for _, r in ms.iterrows():
    bt_df = pd.concat([bt_df, pd.DataFrame([{
        "cutoff": "2026-05", "is_tuning_month": False, "train_start": "2025-05", "train_end": "2026-04",
        "n_train": len(w7_train), "n_train_raw": np.nan, "n_test": int(r["n_test"]),
        "price_lower": float(w7_bounds[0]), "price_upper": float(w7_bounds[1]), "model": r["model"],
        "feature_set": r["feature_set"], "n_estimators": int(r["n_estimators"]), "scored_in_week": 7,
        "R2": r["R2"], "MAE": r["MAE"], "RMSE": r["RMSE"], "MAPE": r["MAPE"], "MdAPE": r["MdAPE"],
        "fit_s": np.nan}])], ignore_index=True)

bt_df.to_csv("outputs/week8_backtest.csv", index=False)
print("\nMdAPE by cutoff (2026-04 = tuning month, optimistic; 2026-05 = held-out test, Week 7 numbers):")
print((bt_df.pivot(index="cutoff", columns="model", values="MdAPE") * 100).round(2).to_string())
print("\nMAE by cutoff:")
print(bt_df.pivot(index="cutoff", columns="model", values="MAE").round(0).to_string())
print("\nWrote outputs/week8_backtest.csv")
