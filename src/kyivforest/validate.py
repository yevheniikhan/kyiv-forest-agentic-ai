"""Model training and validation with random and spatial block cross-validation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold

from . import config as C


def add_blocks(df, lon="lon", lat="lat", size=C.BLOCK_SIZE_M, epsg=C.UTM_EPSG):
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(df[lon].values, df[lat].values)
    df = df.copy()
    df["x"], df["y"] = x, y
    df["block"] = (np.floor(x / size).astype(int).astype(str) + "_" +
                   np.floor(y / size).astype(int).astype(str))
    return df


def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return {"n": int(obs.size), "r2": float(1 - ss_res / ss_tot), "rmse": rmse,
            "rrmse_pct": 100 * rmse / float(obs.mean()), "bias": float(np.mean(pred - obs)),
            "r": float(np.corrcoef(obs, pred)[0, 1])}


def make_model(kind="rf", seed=0):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    raise ValueError(kind)


def cross_validate(df, features, target="agbd", kind="rf", folds=5, scheme="spatial", seed=0):
    """Return out-of-fold predictions and metrics. scheme: 'random' or 'spatial' (block groups)."""
    X, y = df[features].values, df[target].values
    oof = np.full(len(df), np.nan)
    if scheme == "random":
        splits = KFold(folds, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(folds).split(X, y, groups=df["block"].values)
    for tr, te in splits:
        m = make_model(kind, seed).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof, metrics(y, oof)


def importance(df, features, target="agbd", seed=0):
    m = make_model("rf", seed).fit(df[features].values, df[target].values)
    return pd.Series(m.feature_importances_, index=features).sort_values(ascending=False)
