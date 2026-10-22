"""Shared loading, row-validity rules, and sklearn preprocessing for the CRMLS price models.

Used by 02_preprocessing.ipynb, 03_baseline_model.ipynb, and (Week 5) 04_model_comparison.ipynb
so that every model sees exactly the same data, split, and preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_DIR = Path('Data and Python File for Internship')
DATA_PATTERN = 'CRMLSSold*.csv'
TARGET = 'ClosePrice'

# Raw source column -> analysis name. Only modeling fields are loaded (no identifiers/addresses).
SOURCE_COLUMNS = {
    'PropertyType': 'PropertyType', 'PropertySubType': 'PropertySubType',
    'CloseDate': 'CloseDate', 'ClosePrice': 'ClosePrice', 'LivingArea': 'LivingArea',
    'BedroomsTotal': 'Bedrooms', 'BathroomsTotalInteger': 'Bathrooms', 'LotSizeSquareFeet': 'LotSize',
    'PostalCode': 'PostalCode', 'YearBuilt': 'YearBuilt', 'GarageSpaces': 'GarageSpaces',
    'ParkingTotal': 'ParkingTotal', 'Stories': 'Stories', 'PoolPrivateYN': 'PoolPrivateYN',
    'Latitude': 'Latitude', 'Longitude': 'Longitude',
}
NUMERIC_FEATURES = ['LivingArea', 'Bedrooms', 'Bathrooms', 'LotSize', 'YearBuilt',
                    'GarageSpaces', 'ParkingTotal', 'Stories', 'PoolPrivateYN']
LOG_FEATURES = ['LivingArea', 'LotSize']          # log1p-transformed inside the pipeline
LOCATION_CATEGORICAL = ['PostalCode']             # linear model location signal
COORDINATE_FEATURES = ['Latitude', 'Longitude']   # kept in the CSV for tree models (Part B)

# Row-validity rule (feature validity, applied to every split incl. test):
MIN_LIVING_AREA = 200        # sqft; LivingArea missing or < 200 is treated as invalid
# Train-only price filter (percentiles computed on the training rows only; never applied to val/test):
PRICE_LOWER_PCT, PRICE_UPPER_PCT = 0.01, 0.995
# Numeric features are clipped to these train percentiles inside the pipeline:
CLIP_LOWER_PCT, CLIP_UPPER_PCT = 0.005, 0.995


def load_sold(data_dir=DATA_DIR, min_month=None, max_month=None):
    """Load sold CSVs, keep Residential/SingleFamilyResidence, drop rows without CloseDate/ClosePrice."""
    frames = []
    for csv_file in sorted(Path(data_dir).glob(DATA_PATTERN)):
        frames.append(pd.read_csv(csv_file, usecols=lambda c: c in SOURCE_COLUMNS,
                                  dtype={'PostalCode': 'string'}, low_memory=False))
    if not frames:
        raise FileNotFoundError(f'No sold CSV files found under {Path(data_dir).resolve()}.')
    df = pd.concat(frames, ignore_index=True, sort=False).rename(columns=SOURCE_COLUMNS)
    df['CloseDate'] = pd.to_datetime(df['CloseDate'], errors='coerce')
    df = df.loc[(df['PropertyType'] == 'Residential') & (df['PropertySubType'] == 'SingleFamilyResidence')]
    for column in [TARGET, 'LivingArea', 'Bedrooms', 'Bathrooms', 'LotSize', 'YearBuilt',
                   'GarageSpaces', 'ParkingTotal', 'Stories', 'Latitude', 'Longitude']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    pool = df['PoolPrivateYN'].astype('string').str.strip().str.lower()
    df['PoolPrivateYN'] = pool.map({'true': 1.0, 'false': 0.0, '1': 1.0, '0': 0.0}).astype('float64')
    zip5 = df['PostalCode'].astype('string').str.strip().str[:5]
    df['PostalCode'] = zip5.where(zip5.str.fullmatch(r'\d{5}'), 'Unknown').fillna('Unknown').astype(object)
    df = df.dropna(subset=['CloseDate', TARGET])
    df['Month'] = df['CloseDate'].dt.to_period('M')
    if min_month is not None:
        df = df.loc[df['Month'] >= pd.Period(min_month, 'M')]
    if max_month is not None:
        df = df.loc[df['Month'] <= pd.Period(max_month, 'M')]
    keep = ['CloseDate', 'Month', TARGET] + NUMERIC_FEATURES + LOCATION_CATEGORICAL + COORDINATE_FEATURES
    return df[keep].sort_values('CloseDate').reset_index(drop=True)


def valid_rows(df):
    """Feature-validity mask (no price information used)."""
    return df['LivingArea'].ge(MIN_LIVING_AREA)  # NaN compares False -> invalid


def filter_train_prices(train_df):
    """Drop training rows outside train-only ClosePrice percentile bounds. Returns (filtered, bounds)."""
    lower, upper = train_df[TARGET].quantile([PRICE_LOWER_PCT, PRICE_UPPER_PCT])
    return train_df.loc[train_df[TARGET].between(lower, upper)], (lower, upper)


def window_months(end_month, length):
    end = pd.Period(end_month, 'M')
    return pd.period_range(end - (length - 1), end, freq='M')


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to percentiles learned on the training data (NaNs pass through)."""

    def __init__(self, lower=CLIP_LOWER_PCT, upper=CLIP_UPPER_PCT):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _numeric_pipe(log=False):
    steps = [('clip', QuantileClipper())]
    if log:
        steps.append(('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')))
    steps += [('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]
    return Pipeline(steps)


def make_preprocessor(numeric=NUMERIC_FEATURES, log_features=LOG_FEATURES,
                      categorical=LOCATION_CATEGORICAL, min_frequency=10, extra_numeric=()):
    """ColumnTransformer: clip -> (log1p) -> median impute + missing indicators -> scale; ZIP one-hot."""
    log_cols = [c for c in numeric if c in log_features]
    plain_cols = [c for c in numeric if c not in log_features] + list(extra_numeric)
    transformers = []
    if log_cols:
        transformers.append(('num_log', _numeric_pipe(log=True), log_cols))
    if plain_cols:
        transformers.append(('num', _numeric_pipe(log=False), plain_cols))
    if categorical:
        transformers.append(('zip', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                  min_frequency=min_frequency), list(categorical)))
    return ColumnTransformer(transformers)


def make_linear_model(log_target=True, **preprocessor_kwargs):
    """Plain LinearRegression on the shared preprocessor; optional log(ClosePrice) target.

    With log_target=True predictions are exponentiated back, so all metrics are in dollars.
    """
    pipe = Pipeline([('prep', make_preprocessor(**preprocessor_kwargs)), ('model', LinearRegression())])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {'R2': r2_score(y_true, y_pred), 'MAE': mean_absolute_error(y_true, y_pred),
            'MdAPE': float(np.median(np.abs(y_true - y_pred) / y_true))}
