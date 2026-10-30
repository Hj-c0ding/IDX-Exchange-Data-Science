"""Week 6 feature engineering: engineered columns, feature-set definitions, and model factories.

Design rule: row-wise transforms here are deterministic and use no learned values (invalid -> NaN).
Everything that is *learned* (clip bounds, medians, missing indicators, scaling, category grouping,
target encoding) happens inside the sklearn Pipeline and is therefore fit on training rows only.
`model_pipeline.py` is reused as-is (so the v2 baseline stays reproducible).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

import model_pipeline as mp

RANDOM_STATE = 42
ENGINEERED = ['BedBathRatio', 'PropertyAge', 'LivingArea_per_Bedroom', 'Lot_to_Living_ratio',
              'SaleMonth_sin', 'SaleMonth_cos']
MAX_PROPERTY_AGE = 150

FEATURE_SET_DESCRIPTION = {
    'F0': 'v2 baseline (old)',
    'F1': 'F0 + engineered structural/temporal',
    'F2': 'F0 + school district',
    'F3': 'F1 + school district (full)',
}


def add_engineered_features(df):
    """Add the engineered columns. No learning here: invalid inputs become NaN (imputed in-pipeline)."""
    out = df.copy()
    close = pd.to_datetime(out['CloseDate'])
    year = close.dt.year
    baths = out['Bathrooms'].clip(lower=1.0)                      # guard divide-by-zero
    out['BedBathRatio'] = out['Bedrooms'] / baths
    yb = out['YearBuilt'].where((out['YearBuilt'] >= 1800) & (out['YearBuilt'] <= year + 1))
    out['PropertyAge'] = (year - yb).clip(0, MAX_PROPERTY_AGE)
    out['LivingArea_per_Bedroom'] = out['LivingArea'] / out['Bedrooms'].clip(lower=1.0)
    out['Lot_to_Living_ratio'] = out['LotSize'] / out['LivingArea'].clip(lower=mp.MIN_LIVING_AREA)
    angle = 2 * np.pi * close.dt.month / 12
    out['SaleMonth_sin'], out['SaleMonth_cos'] = np.sin(angle), np.cos(angle)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(feature_set, kind):
    """Return (numeric_cols, extra_numeric_cols, categorical_cols) for a feature set and model kind.

    kind 'lr' = ZIP location only, 'tree' = ZIP + Latitude/Longitude (as in the Week 5 trees).
    """
    use_eng = feature_set in ('F1', 'F3')
    use_district = feature_set in ('F2', 'F3')
    extra = list(mp.COORDINATE_FEATURES) if kind == 'tree' else []
    if use_eng:
        extra += ENGINEERED
    cats = list(mp.LOCATION_CATEGORICAL) + (['SchoolDistrict'] if use_district else [])
    return list(mp.NUMERIC_FEATURES), extra, cats


def input_columns(feature_set, kind):
    num, extra, cats = feature_columns(feature_set, kind)
    return num + extra + cats


def make_preprocessor(feature_set, kind, district_encoding='onehot', min_frequency=5, district_min_frequency=20):
    """ColumnTransformer: shared numeric pipe (clip -> impute+indicator -> scale) + categorical encoders.

    district_encoding: 'onehot' (rare districts grouped, fit on train) or 'target' (TargetEncoder with
    internal 5-fold cross-fitting, so a row's own price never feeds its own encoding).
    """
    num, extra, cats = feature_columns(feature_set, kind)
    transformers = [('num', mp._numeric_pipe(log=False), num + extra),
                    ('zip', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=min_frequency),
                     list(mp.LOCATION_CATEGORICAL))]
    if 'SchoolDistrict' in cats:
        if district_encoding == 'onehot':
            enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=district_min_frequency)
        elif district_encoding == 'target':
            enc = TargetEncoder(target_type='continuous',
                                cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        else:
            raise ValueError(district_encoding)
        transformers.append(('district', enc, ['SchoolDistrict']))
    return ColumnTransformer(transformers)


def make_model(name, feature_set, district_encoding='onehot', district_min_frequency=20, **params):
    """Log-target model (metrics are exponentiated back to dollars). name in {'LR','DT','RF'}."""
    kind = 'lr' if name == 'LR' else 'tree'
    prep = make_preprocessor(feature_set, kind, district_encoding, district_min_frequency=district_min_frequency)
    if name == 'LR':
        est = LinearRegression()
    elif name == 'DT':
        est = DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    elif name == 'RF':
        est = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **{'n_estimators': 200, **params})
    else:
        raise ValueError(name)
    return TransformedTargetRegressor(Pipeline([('prep', prep), ('model', est)]), func=np.log, inverse_func=np.exp)


def full_metrics(y_true, y_pred):
    """R2, MAE, RMSE, MAPE, MdAPE on the dollar scale (MdAPE/MAPE as fractions)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {'R2': r2_score(y_true, y_pred), 'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAPE': float(ape.mean()), 'MdAPE': float(np.median(ape))}
