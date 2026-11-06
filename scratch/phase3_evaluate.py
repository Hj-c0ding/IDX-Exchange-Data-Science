"""Phase 3: Final test scoring (ONCE per final model), paired bootstrap, error breakdowns, and model persistence."""
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import boosting as bt
import features as ft
import model_pipeline as mp

SEED = 42
MIN_FS_GAIN_MDAPE = 0.0010

data = pd.read_csv("outputs/03_features_v3.csv", dtype={"PostalCode": "object", "SchoolDistrict": "object"})
train, train_bounds = mp.filter_train_prices(data.loc[data["split"].eq("train")])
test = data.loc[data["split"].eq("test")]
val_train, _ = mp.filter_train_prices(data.loc[data["val_split"].eq("train")])
val_rows = data.loc[data["val_split"].eq("val")]

n_train, n_val_train, n_test = len(train), len(val_train), len(test)
scale_factor = n_train / n_val_train
print(
    f"Train={n_train:,} (bounds ${train_bounds[0]:,.0f}-${train_bounds[1]:,.0f}) | "
    f"Val-train={n_val_train:,} | Scale={scale_factor:.6f} | Test={n_test:,}"
)

# Load Week 6 & Week 7 validation tables to pick final configs strictly from validation
w6_val = pd.read_csv("outputs/week6_validation.csv")
w6_val = w6_val.loc[~w6_val["district_encoding"].str.startswith("onehot_mf")]
w7_val = pd.read_csv("outputs/week7_validation.csv")

chosen_gbm = {}
for lib in ["LightGBM", "XGBoost"]:
    sub = w7_val.loc[w7_val["library"].eq(lib)]
    best_by_fs = sub.sort_values(["val_MdAPE", "val_MAE"]).groupby("feature_set").head(1).set_index("feature_set")
    b_f0 = best_by_fs.loc["F0"]
    alt_sets = [s for s in best_by_fs.index if s != "F0"]
    best_alt_fs = best_by_fs.loc[alt_sets].sort_values(["val_MdAPE", "val_MAE"]).index[0]
    b_alt = best_by_fs.loc[best_alt_fs]
    gain = b_f0["val_MdAPE"] - b_alt["val_MdAPE"]
    adopt = (gain >= MIN_FS_GAIN_MDAPE) and (b_alt["val_MAE"] < b_f0["val_MAE"])
    fs = best_alt_fs if adopt else "F0"
    winner = best_by_fs.loc[fs]
    scaled_iter = bt.scale_iterations(int(winner["best_iteration"]), n_train, n_val_train)
    chosen_gbm[lib] = {
        "feature_set": fs,
        "categorical_handling": winner["categorical_handling"],
        "objective": winner["objective"],
        "val_best_iteration": int(winner["best_iteration"]),
        "n_estimators": scaled_iter,
        "params": json.loads(winner["params"]),
        "val_MAE": float(winner["val_MAE"]),
        "val_MdAPE": float(winner["val_MdAPE"]),
    }
    print(f"Chosen {lib}: {chosen_gbm[lib]}")

# Define the 5 final models to fit on full train and score ONCE on test
final_specs = [
    {
        "model": "LR",
        "feature_set": "F3",
        "categorical_handling": "onehot",
        "objective": "squared_error",
        "val_best_iteration": 1,
        "n_estimators": 1,
        "val_MAE": float(w6_val.loc[w6_val["model"].eq("LR") & w6_val["feature_set"].eq("F3") & w6_val["district_encoding"].eq("onehot"), "MAE"].iloc[0]),
        "val_MdAPE": float(w6_val.loc[w6_val["model"].eq("LR") & w6_val["feature_set"].eq("F3") & w6_val["district_encoding"].eq("onehot"), "MdAPE"].iloc[0]),
        "builder": lambda: ft.make_model("LR", "F3", district_encoding="onehot"),
        "cols": ft.input_columns("F3", "lr"),
    },
    {
        "model": "DT",
        "feature_set": "F0",
        "categorical_handling": "onehot",
        "objective": "squared_error",
        "val_best_iteration": 1,
        "n_estimators": 1,
        "val_MAE": float(w6_val.loc[w6_val["model"].eq("DT") & w6_val["feature_set"].eq("F0"), "MAE"].iloc[0]),
        "val_MdAPE": float(w6_val.loc[w6_val["model"].eq("DT") & w6_val["feature_set"].eq("F0"), "MdAPE"].iloc[0]),
        "builder": lambda: ft.make_model("DT", "F0", max_depth=18, min_samples_leaf=5),
        "cols": ft.input_columns("F0", "tree"),
    },
    {
        "model": "RF",
        "feature_set": "F0",
        "categorical_handling": "onehot",
        "objective": "squared_error",
        "val_best_iteration": 200,
        "n_estimators": 200,
        "val_MAE": float(w6_val.loc[w6_val["model"].eq("RF") & w6_val["feature_set"].eq("F0"), "MAE"].iloc[0]),
        "val_MdAPE": float(w6_val.loc[w6_val["model"].eq("RF") & w6_val["feature_set"].eq("F0"), "MdAPE"].iloc[0]),
        "builder": lambda: ft.make_model("RF", "F0", min_samples_leaf=3),
        "cols": ft.input_columns("F0", "tree"),
    },
]

for lib in ["LightGBM", "XGBoost"]:
    cfg = chosen_gbm[lib]
    final_specs.append({
        "model": lib,
        "feature_set": cfg["feature_set"],
        "categorical_handling": cfg["categorical_handling"],
        "objective": cfg["objective"],
        "val_best_iteration": cfg["val_best_iteration"],
        "n_estimators": cfg["n_estimators"],
        "val_MAE": cfg["val_MAE"],
        "val_MdAPE": cfg["val_MdAPE"],
        "builder": lambda l=lib, c=cfg: bt.make_boosting_model(
            l,
            c["feature_set"],
            categorical_handling=c["categorical_handling"],
            objective=c["objective"],
            n_estimators=c["n_estimators"],
            **c["params"],
        ),
        "cols": ft.input_columns(cfg["feature_set"], "tree"),
    })

y_train = train[mp.TARGET].to_numpy(dtype=float)
y_test = test[mp.TARGET].to_numpy(dtype=float)

comp_rows = []
preds = {}
fitted_models = {}

for spec in final_specs:
    m = spec["model"]
    cols = spec["cols"]
    est = spec["builder"]()
    t0 = time.perf_counter()
    est.fit(train[cols], y_train)
    fit_s = time.perf_counter() - t0

    p_train = est.predict(train[cols])
    p_test = est.predict(test[cols])  # SINGLE TEST SCORING PER FINAL MODEL

    tm = ft.full_metrics(y_test, p_test)
    trm = ft.full_metrics(y_train, p_train)
    n_feat = len(est.regressor_.named_steps["prep"].get_feature_names_out())

    # Leakage sanity check on test
    if m in ("LightGBM", "XGBoost"):
        assert tm["MdAPE"] >= 0.05, f"STOP: test MdAPE {tm['MdAPE']:.2%} < 5%"
        assert tm["MdAPE"] >= 0.081899 * 0.70, f"STOP: test MdAPE {tm['MdAPE']:.2%} >30% better than RF"

    comp_rows.append({
        "model": m,
        "feature_set": spec["feature_set"],
        "categorical_handling": spec["categorical_handling"],
        "objective": spec["objective"],
        "val_best_iteration": spec["val_best_iteration"],
        "best_iteration": spec["n_estimators"],
        "n_features": n_feat,
        "R2": tm["R2"],
        "MAE": tm["MAE"],
        "RMSE": tm["RMSE"],
        "MAPE": tm["MAPE"],
        "MdAPE": tm["MdAPE"],
        "train_R2": trm["R2"],
        "fit_s": fit_s,
        "val_MAE": spec["val_MAE"],
        "val_MdAPE": spec["val_MdAPE"],
    })
    preds[m] = p_test
    fitted_models[m] = (est, cols)
    print(
        f"{m:8s} ({spec['feature_set']}): test R2={tm['R2']:.4f} | MAE=${tm['MAE']:,.0f} | "
        f"RMSE=${tm['RMSE']:,.0f} | MAPE={tm['MAPE']:.2%} | MdAPE={tm['MdAPE']:.2%} | "
        f"train_R2={trm['R2']:.4f} | n_feat={n_feat} | iter={spec['n_estimators']} | {fit_s:.1f}s"
    )

comp_df = pd.DataFrame(comp_rows)
rf_row = comp_df.loc[comp_df["model"].eq("RF")].iloc[0]
lr_row = comp_df.loc[comp_df["model"].eq("LR")].iloc[0]

# Regression checks vs Week 6
assert abs(lr_row["R2"] - 0.5024) < 5e-4 and abs(lr_row["MAE"] - 228644) < 5, "LR F3 drifted!"
assert abs(rf_row["R2"] - 0.4445) < 5e-4 and abs(rf_row["MAE"] - 220621) < 5 and abs(rf_row["MdAPE"] - 0.0819) < 5e-4, "RF F0 drifted!"

comp_df["dMAE_vs_RF"] = comp_df["MAE"] - rf_row["MAE"]
comp_df["dMdAPE_pp_vs_RF"] = (comp_df["MdAPE"] - rf_row["MdAPE"]) * 100
comp_df["dMAE_vs_LR"] = comp_df["MAE"] - lr_row["MAE"]
comp_df["dMdAPE_pp_vs_LR"] = (comp_df["MdAPE"] - lr_row["MdAPE"]) * 100
comp_df.to_csv("outputs/week7_comparison.csv", index=False)
print("\nSaved outputs/week7_comparison.csv")

# Save predictions for plotting in notebook without re-scoring test
pred_df = pd.DataFrame({"ClosePrice": y_test, **preds})
pred_df.to_csv("scratch/week7_test_predictions.csv", index=False)

# Paired bootstrap (2,000 resamples, seed 42)
gbm_sub = comp_df.loc[comp_df["model"].isin(["LightGBM", "XGBoost"])].sort_values(["MdAPE", "MAE"])
best_gbm_name = gbm_sub.iloc[0]["model"]
print(f"\nBest GBM on test by MdAPE: {best_gbm_name}")

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n_test, size=(2000, n_test))

boot_rows = []
for gbm_name in ["LightGBM", "XGBoost"]:
    for ref_name, ref_label in [("RF", "RF F0"), ("LR", "LR F3")]:
        ape_g = np.abs(preds[gbm_name] - y_test) / y_test
        ape_r = np.abs(preds[ref_name] - y_test) / y_test
        ae_g = np.abs(preds[gbm_name] - y_test)
        ae_r = np.abs(preds[ref_name] - y_test)

        d_md_boot = (np.median(ape_g[idx], axis=1) - np.median(ape_r[idx], axis=1)) * 100
        d_mae_boot = ae_g[idx].mean(axis=1) - ae_r[idx].mean(axis=1)

        d_md_pt = (np.median(ape_g) - np.median(ape_r)) * 100
        d_mae_pt = float(ae_g.mean() - ae_r.mean())
        md_lo, md_hi = float(np.percentile(d_md_boot, 2.5)), float(np.percentile(d_md_boot, 97.5))
        mae_lo, mae_hi = float(np.percentile(d_mae_boot, 2.5)), float(np.percentile(d_mae_boot, 97.5))

        boot_rows.append({
            "comparison": f"{gbm_name} vs {ref_label}",
            "gbm": gbm_name,
            "reference": ref_label,
            "dMdAPE_pp": d_md_pt,
            "dMdAPE_ci_lo": md_lo,
            "dMdAPE_ci_hi": md_hi,
            "mdape_sig": (md_hi < 0) or (md_lo > 0),
            "dMAE": d_mae_pt,
            "dMAE_ci_lo": mae_lo,
            "dMAE_ci_hi": mae_hi,
            "mae_sig": (mae_hi < 0) or (mae_lo > 0),
        })
        print(
            f"Bootstrap {gbm_name:8s} vs {ref_label}: "
            f"dMdAPE={d_md_pt:+.2f} pp [{md_lo:+.2f}, {md_hi:+.2f}] (excludes 0={md_hi < 0 or md_lo > 0}) | "
            f"dMAE=${d_mae_pt:+,.0f} [${mae_lo:+,.0f}, ${mae_hi:+,.0f}] (excludes 0={mae_hi < 0 or mae_lo > 0})"
        )

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv("outputs/week7_bootstrap.csv", index=False)

# Price quintile breakdown (edges from TRAIN)
edges = np.quantile(y_train, [0.2, 0.4, 0.6, 0.8])
q = np.digitize(y_test, edges)
qlabels = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
qtab = pd.DataFrame({"quintile": qlabels, "n_test_homes": [int((q == i).sum()) for i in range(5)]})
for m in ["LR", "DT", "RF", "LightGBM", "XGBoost"]:
    ape = np.abs(preds[m] - y_test) / y_test
    qtab[m] = [float(np.median(ape[q == i])) for i in range(5)]
qtab.to_csv("outputs/week7_mdape_by_quintile.csv", index=False)
print("\nMdAPE by train price quintile:")
print(qtab.to_string(index=False))

# High-end breakdown (>$3M, <=$8.9M train cap, >$8.9M train cap, >$10M)
hi_rows = []
slices = [
    ("All test homes", np.ones(n_test, dtype=bool)),
    ("At or below train cap (<=$8.9M)", y_test <= train_bounds[1]),
    ("High-end (>$3M)", y_test > 3_000_000),
    ("Above train cap (>$8.9M)", y_test > train_bounds[1]),
    ("Ultra-luxury (>$10M)", y_test > 10_000_000),
]
for s_name, mask in slices:
    for m in ["LR", "DT", "RF", "LightGBM", "XGBoost"]:
        yt_s = y_test[mask]
        yp_s = preds[m][mask]
        ape_s = np.abs(yp_s - yt_s) / yt_s
        hi_rows.append({
            "slice": s_name,
            "n_homes": int(mask.sum()),
            "model": m,
            "MdAPE": float(np.median(ape_s)),
            "MAE": float(np.mean(np.abs(yp_s - yt_s))),
            "R2": float(r2_score(yt_s, yp_s)),
            "max_pred": float(np.max(yp_s)),
            "max_actual": float(np.max(yt_s)),
        })
hi_df = pd.DataFrame(hi_rows)
hi_df.to_csv("outputs/week7_high_end_breakdown.csv", index=False)
print("\nHigh-end / train-cap breakdown:")
print(hi_df.pivot(index=["slice", "n_homes"], columns="model", values=["MdAPE", "R2", "max_pred"]).to_string())

# Save best overall model by test MdAPE and run reload smoke test
overall_best_model_name = comp_df.sort_values(["MdAPE", "MAE"]).iloc[0]["model"]
best_est, best_cols = fitted_models[overall_best_model_name]
best_row = comp_df.loc[comp_df["model"].eq(overall_best_model_name)].iloc[0]

models_dir = Path("outputs/models")
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "best_model.joblib"
feat_path = models_dir / "feature_list.json"

joblib.dump(best_est, model_path)
with open(feat_path, "w") as f:
    json.dump(
        {
            "model": overall_best_model_name,
            "feature_set": best_row["feature_set"],
            "categorical_handling": best_row["categorical_handling"],
            "objective": best_row["objective"],
            "best_iteration": int(best_row["best_iteration"]),
            "features": list(best_cols),
            "test_MdAPE": float(best_row["MdAPE"]),
            "test_MAE": float(best_row["MAE"]),
            "test_R2": float(best_row["R2"]),
        },
        f,
        indent=2,
    )

# Reload smoke test
reloaded_model = joblib.load(model_path)
reloaded_meta = json.load(open(feat_path))
reloaded_preds = reloaded_model.predict(test[reloaded_meta["features"]])
reloaded_metrics = ft.full_metrics(y_test, reloaded_preds)
assert abs(reloaded_metrics["MdAPE"] - best_row["MdAPE"]) < 1e-12, "Reloaded model MdAPE mismatch!"
assert abs(reloaded_metrics["MAE"] - best_row["MAE"]) < 1e-6, "Reloaded model MAE mismatch!"
print(
    f"\nSaved & verified outputs/models/best_model.joblib ({overall_best_model_name} {best_row['feature_set']}): "
    f"reloaded test MdAPE={reloaded_metrics['MdAPE']:.4%} == {best_row['MdAPE']:.4%} (PASS)"
)

