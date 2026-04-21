#!/usr/bin/env python3
"""Strong reproducible baseline for DACON CH2026 metrics prediction.

The script builds daily lifelog features from every sensor parquet file,
trains a per-target ensemble, tunes probability blending with subject-level
priors by out-of-fold log-loss, and writes a DACON-ready submission file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "lifelog_date"]
EPS = 1e-5
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--top-apps", type=int, default=40)
    parser.add_argument("--top-ambience", type=int, default=50)
    parser.add_argument("--recent-days", type=int, default=10)
    return parser.parse_args()


# Output hygiene for future agents:
# - Keep generated CSV/JSON/Parquet/log artifacts under the configured output_dir.
# - Do not scatter ad-hoc CSV files or extra experiment scripts in the repo root.
# - Submission files may accumulate, but always version them via
#   next_submission_path() as outputs/submission_vN.csv instead of overwriting.
# - Prefer stable diagnostic filenames under outputs/ when possible.
#
# Model-selection guidance for future agents:
# - README states the competition metric is Average Log-Loss; keep that primary.
# - Also report MAE between binary labels and predicted probabilities as a
#   secondary sanity metric, and macro-F1 as a thresholded reference metric.
# - Do not replace the main model from a single run or public score guess.
#   Use GroupKFold OOF results by target, then update the model only when the
#   candidate improves the objective metrics consistently.

def next_submission_path(output_dir: Path) -> Path:
    """Return outputs/submission_vN.csv with the next available version number."""
    versions = []
    for path in output_dir.glob("submission_v*.csv"):
        suffix = path.stem.removeprefix("submission_v")
        if suffix.isdigit():
            versions.append(int(suffix))
    next_version = max(versions, default=0) + 1
    return output_dir / f"submission_v{next_version}.csv"


def clip_proba(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_proba(y_pred)
    return float(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean())


def average_log_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return float(np.mean([binary_log_loss(y_true[t].values, y_pred[t].values) for t in TARGETS]))


def safe_token(value: object, max_len: int = 64) -> str:
    raw = str(value)
    ascii_part = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^0-9A-Za-z]+", "_", ascii_part).strip("_").lower()
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    if not token:
        token = f"x_{digest}"
    if token[0].isdigit():
        token = f"f_{token}"
    token = token[:max_len].strip("_")
    return f"{token}_{digest}"


def make_unique_columns(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        token = safe_token(col, max_len=120)
        count = seen.get(token, 0)
        seen[token] = count + 1
        out.append(token if count == 0 else f"{token}_{count}")
    return out


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    names = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x)]
            names.append(f"{prefix}_{'_'.join(parts)}")
        else:
            names.append(f"{prefix}_{col}")
    df.columns = names
    return df


def add_date_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lifelog_date"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d")
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    out["time_bin"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    return out


def daily_numeric_features(
    path: Path,
    value_cols: list[str],
    prefix: str,
    add_changes: bool = False,
    categorical_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", *value_cols])
    df = add_date_key(df)

    agg_map = {c: ["count", "mean", "std", "min", "max", "median", "sum"] for c in value_cols}
    daily = df.groupby(KEYS, observed=True).agg(agg_map)
    daily = flatten_columns(daily, prefix)

    pieces = [daily]
    for col in value_cols:
        pivot = df.pivot_table(
            index=KEYS,
            columns="time_bin",
            values=col,
            aggfunc=["mean", "sum"],
            observed=True,
        )
        pivot = flatten_columns(pivot, f"{prefix}_{col}_by_time")
        pieces.append(pivot)

    if add_changes:
        for col in value_cols:
            tmp = df.sort_values(["subject_id", "lifelog_date", "timestamp"])
            changed = tmp.groupby(KEYS, observed=True)[col].transform(lambda s: s.ne(s.shift()).astype(int))
            tmp[f"{col}_changes"] = changed
            chg = tmp.groupby(KEYS, observed=True)[f"{col}_changes"].sum().to_frame(f"{prefix}_{col}_changes")
            pieces.append(chg)

    if categorical_col:
        counts = (
            df.groupby([*KEYS, categorical_col], observed=True)
            .size()
            .unstack(categorical_col, fill_value=0)
        )
        counts.columns = [f"{prefix}_{categorical_col}_count_{safe_token(c, 24)}" for c in counts.columns]
        ratios = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        ratios.columns = [c.replace("_count_", "_ratio_") for c in ratios.columns]
        pieces.extend([counts, ratios])

    return pd.concat(pieces, axis=1).reset_index()


def summarize_numeric_list(items: object, numeric_keys: list[str]) -> dict[str, float]:
    result: dict[str, float] = {"n": 0.0}
    if items is None:
        return result
    try:
        seq = list(items)
    except TypeError:
        return result
    result["n"] = float(len(seq))
    for key in numeric_keys:
        vals = []
        for item in seq:
            if isinstance(item, dict) and item.get(key) is not None:
                try:
                    vals.append(float(item.get(key)))
                except (TypeError, ValueError):
                    pass
        if vals:
            arr = np.asarray(vals, dtype=float)
            result[f"{key}_mean"] = float(arr.mean())
            result[f"{key}_std"] = float(arr.std())
            result[f"{key}_min"] = float(arr.min())
            result[f"{key}_max"] = float(arr.max())
            result[f"{key}_sum"] = float(arr.sum())
    return result


def daily_nested_numeric_features(
    path: Path,
    list_col: str,
    numeric_keys: list[str],
    prefix: str,
    unique_key: str | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", list_col])
    df = add_date_key(df)

    rows = [summarize_numeric_list(items, numeric_keys) for items in df[list_col].to_numpy()]
    stats = pd.DataFrame(rows).add_prefix(f"{prefix}_row_")
    tmp = pd.concat([df[KEYS].reset_index(drop=True), stats], axis=1)

    agg_map = {}
    for col in stats.columns:
        if col.endswith("_n"):
            agg_map[col] = ["sum", "mean", "max"]
        elif col.endswith("_sum"):
            agg_map[col] = ["sum", "mean", "max"]
        else:
            agg_map[col] = ["mean", "std", "min", "max"]

    daily = tmp.groupby(KEYS, observed=True).agg(agg_map)
    daily = flatten_columns(daily, prefix)
    pieces = [daily]

    if unique_key:
        def unique_count(series: pd.Series) -> int:
            seen = set()
            for items in series:
                if items is None:
                    continue
                for item in list(items):
                    if isinstance(item, dict) and item.get(unique_key) is not None:
                        seen.add(item.get(unique_key))
            return len(seen)

        uniq = df.groupby(KEYS, observed=True)[list_col].apply(unique_count)
        pieces.append(uniq.to_frame(f"{prefix}_{unique_key}_nunique"))

    return pd.concat(pieces, axis=1).reset_index()


def daily_heart_rate_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "heart_rate"])
    df = add_date_key(df)

    records = []
    for items in df["heart_rate"].to_numpy():
        vals = []
        if items is not None:
            for value in list(items):
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    pass
        if vals:
            arr = np.asarray(vals, dtype=float)
            records.append(
                {
                    "hr_n": len(arr),
                    "hr_sum": float(arr.sum()),
                    "hr_mean": float(arr.mean()),
                    "hr_std": float(arr.std()),
                    "hr_min": float(arr.min()),
                    "hr_max": float(arr.max()),
                    "hr_median": float(np.median(arr)),
                }
            )
        else:
            records.append({"hr_n": 0, "hr_sum": 0.0})

    stats = pd.DataFrame(records)
    tmp = pd.concat([df[KEYS].reset_index(drop=True), stats], axis=1)
    daily = tmp.groupby(KEYS, observed=True).agg(
        {
            "hr_n": ["sum", "mean", "max"],
            "hr_sum": ["sum", "mean", "max"],
            "hr_mean": ["mean", "std", "min", "max"],
            "hr_std": ["mean", "max"],
            "hr_min": ["min", "mean"],
            "hr_max": ["max", "mean"],
            "hr_median": ["mean", "std"],
        }
    )
    daily = flatten_columns(daily, "w_hr")
    daily["w_hr_daily_mean"] = daily["w_hr_hr_sum_sum"] / daily["w_hr_hr_n_sum"].replace(0, np.nan)
    return daily.reset_index()


def daily_usage_features(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_usage_stats"])
    df = add_date_key(df)

    row_records = []
    app_records = []
    app_total = Counter()

    for sid, day, items in zip(df["subject_id"], df["lifelog_date"], df["m_usage_stats"]):
        seq = list(items) if items is not None else []
        total_time = 0.0
        max_time = 0.0
        apps = set()
        for item in seq:
            if not isinstance(item, dict):
                continue
            app = item.get("app_name")
            try:
                total = float(item.get("total_time", 0.0))
            except (TypeError, ValueError):
                total = 0.0
            if app is not None:
                apps.add(str(app))
                app_total[str(app)] += total
                app_records.append((sid, day, str(app), total))
            total_time += total
            max_time = max(max_time, total)
        row_records.append(
            {
                "subject_id": sid,
                "lifelog_date": day,
                "usage_events": len(seq),
                "usage_total_time": total_time,
                "usage_max_time": max_time,
                "usage_unique_apps": len(apps),
            }
        )

    row_df = pd.DataFrame(row_records)
    daily = row_df.groupby(KEYS, observed=True).agg(
        {
            "usage_events": ["sum", "mean", "max"],
            "usage_total_time": ["sum", "mean", "max"],
            "usage_max_time": ["mean", "max"],
            "usage_unique_apps": ["sum", "mean", "max"],
        }
    )
    daily = flatten_columns(daily, "m_usage")
    pieces = [daily]

    top_apps = {app for app, _ in app_total.most_common(top_n)}
    if app_records and top_apps:
        app_df = pd.DataFrame(app_records, columns=["subject_id", "lifelog_date", "app", "total_time"])
        app_df = app_df[app_df["app"].isin(top_apps)]
        pivot = app_df.pivot_table(index=KEYS, columns="app", values="total_time", aggfunc="sum", fill_value=0)
        pivot.columns = [f"m_usage_app_time_{safe_token(c, 40)}" for c in pivot.columns]
        pieces.append(pivot)

    return pd.concat(pieces, axis=1).reset_index()


def ambience_row(items: object) -> dict[str, object]:
    seq = list(items) if items is not None else []
    probs = []
    labels = []
    for item in seq:
        if hasattr(item, "__len__") and len(item) >= 2:
            labels.append(str(item[0]))
            try:
                probs.append(float(item[1]))
            except (TypeError, ValueError):
                probs.append(0.0)
    if not probs:
        return {
            "ambience_n": 0,
            "ambience_prob_sum": 0.0,
            "ambience_top_prob": np.nan,
            "ambience_entropy": np.nan,
            "ambience_top_label": "missing",
        }
    arr = np.asarray(probs, dtype=float)
    norm = arr / arr.sum() if arr.sum() > 0 else arr
    entropy = float(-(norm * np.log(np.clip(norm, EPS, 1.0))).sum()) if arr.sum() > 0 else np.nan
    top_idx = int(arr.argmax())
    return {
        "ambience_n": len(arr),
        "ambience_prob_sum": float(arr.sum()),
        "ambience_top_prob": float(arr[top_idx]),
        "ambience_entropy": entropy,
        "ambience_top_label": labels[top_idx],
    }


def daily_ambience_features(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_ambience"])
    df = add_date_key(df)
    stats = pd.DataFrame([ambience_row(items) for items in df["m_ambience"].to_numpy()])
    tmp = pd.concat([df[KEYS].reset_index(drop=True), stats], axis=1)

    daily = tmp.groupby(KEYS, observed=True).agg(
        {
            "ambience_n": ["sum", "mean", "max"],
            "ambience_prob_sum": ["sum", "mean", "max"],
            "ambience_top_prob": ["mean", "std", "min", "max"],
            "ambience_entropy": ["mean", "std", "min", "max"],
        }
    )
    daily = flatten_columns(daily, "m_ambience")
    pieces = [daily]

    top_labels = tmp["ambience_top_label"].value_counts().head(top_n).index
    label_df = tmp[tmp["ambience_top_label"].isin(top_labels)]
    if not label_df.empty:
        counts = label_df.pivot_table(
            index=KEYS,
            columns="ambience_top_label",
            values="ambience_n",
            aggfunc="count",
            fill_value=0,
        )
        counts.columns = [f"m_ambience_top_count_{safe_token(c, 40)}" for c in counts.columns]
        pieces.append(counts)

    return pd.concat(pieces, axis=1).reset_index()


def add_calendar_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    life = pd.to_datetime(out["lifelog_date"])
    sleep = pd.to_datetime(out["sleep_date"])
    out["life_dayofweek"] = life.dt.dayofweek
    out["life_day"] = life.dt.day
    out["life_month"] = life.dt.month
    out["life_dayofyear"] = life.dt.dayofyear
    out["life_weekofyear"] = life.dt.isocalendar().week.astype(int)
    out["sleep_dayofweek"] = sleep.dt.dayofweek
    out["is_weekend"] = out["life_dayofweek"].isin([5, 6]).astype(int)
    out["life_dayofweek_sin"] = np.sin(2 * np.pi * out["life_dayofweek"] / 7)
    out["life_dayofweek_cos"] = np.cos(2 * np.pi * out["life_dayofweek"] / 7)
    out["life_dayofyear_sin"] = np.sin(2 * np.pi * out["life_dayofyear"] / 366)
    out["life_dayofyear_cos"] = np.cos(2 * np.pi * out["life_dayofyear"] / 366)
    out["sleep_gap_days"] = (sleep - life).dt.days
    out["subject_idx"] = out["subject_id"].str.extract(r"(\d+)").astype(float)
    return out


def add_subject_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    life = pd.to_datetime(out["lifelog_date"])
    out["_life_dt"] = life
    out = out.sort_values(["subject_id", "_life_dt"]).reset_index(drop=True)
    first = out.groupby("subject_id")["_life_dt"].transform("min")
    out["subject_days_since_first"] = (out["_life_dt"] - first).dt.days
    out["subject_row_order"] = out.groupby("subject_id").cumcount()

    reserved = {"subject_id", "sleep_date", "lifelog_date", "_life_dt"}
    numeric_cols = [
        c
        for c in out.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(out[c])
    ]
    sensor_cols = [
        c
        for c in numeric_cols
        if not c.startswith(("life_", "sleep_", "is_weekend", "subject_idx", "subject_days", "subject_row"))
    ]
    # Keep rolling expansion useful but bounded for a tiny training set.
    sensor_cols = sensor_cols[:350]
    for window in (3, 7):
        rolled = (
            out.groupby("subject_id", observed=True)[sensor_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolled.columns = [f"{c}_roll{window}_mean" for c in sensor_cols]
        out = pd.concat([out, rolled], axis=1)
    lagged = out.groupby("subject_id", observed=True)[sensor_cols[:120]].shift(1)
    lagged.columns = [f"{c}_lag1" for c in sensor_cols[:120]]
    out = pd.concat([out, lagged], axis=1)
    return out.drop(columns=["_life_dt"])


def build_features(data_dir: Path, output_dir: Path, top_apps: int, top_ambience: int, force: bool) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "feature_cache.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    train = pd.read_csv(data_dir / "ch2026_metrics_train.csv")
    sample = pd.read_csv(data_dir / "ch2026_submission_sample.csv")
    base = pd.concat([train[["subject_id", "sleep_date", "lifelog_date"]], sample[["subject_id", "sleep_date", "lifelog_date"]]], ignore_index=True)
    base = base.drop_duplicates(KEYS).reset_index(drop=True)
    base = add_calendar_features(base)

    item_dir = data_dir / "ch2025_data_items"
    feature_tables = [
        daily_numeric_features(item_dir / "ch2025_mACStatus.parquet", ["m_charging"], "m_ac", add_changes=True),
        daily_numeric_features(item_dir / "ch2025_mActivity.parquet", ["m_activity"], "m_activity", add_changes=True, categorical_col="m_activity"),
        daily_ambience_features(item_dir / "ch2025_mAmbience.parquet", top_ambience),
        daily_nested_numeric_features(item_dir / "ch2025_mBle.parquet", "m_ble", ["rssi"], "m_ble", unique_key="address"),
        daily_nested_numeric_features(item_dir / "ch2025_mGps.parquet", "m_gps", ["altitude", "latitude", "longitude", "speed"], "m_gps"),
        daily_numeric_features(item_dir / "ch2025_mLight.parquet", ["m_light"], "m_light"),
        daily_numeric_features(item_dir / "ch2025_mScreenStatus.parquet", ["m_screen_use"], "m_screen", add_changes=True),
        daily_usage_features(item_dir / "ch2025_mUsageStats.parquet", top_apps),
        daily_nested_numeric_features(item_dir / "ch2025_mWifi.parquet", "m_wifi", ["rssi"], "m_wifi", unique_key="bssid"),
        daily_heart_rate_features(item_dir / "ch2025_wHr.parquet"),
        daily_numeric_features(item_dir / "ch2025_wLight.parquet", ["w_light"], "w_light"),
        daily_numeric_features(
            item_dir / "ch2025_wPedo.parquet",
            ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
            "w_pedo",
        ),
    ]

    features = base
    for table in feature_tables:
        features = features.merge(table, on=KEYS, how="left")

    features = add_subject_relative_features(features)
    features.to_parquet(cache_path, index=False)
    return features


def make_matrix(features: pd.DataFrame, train: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = features.copy()
    frame = pd.get_dummies(frame, columns=["subject_id"], dtype=float)
    drop_cols = ["sleep_date", "lifelog_date"]
    frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.columns = make_unique_columns(frame.columns)

    n_train = len(train)
    x_train = frame.iloc[:n_train].reset_index(drop=True)
    x_test = frame.iloc[n_train : n_train + len(sample)].reset_index(drop=True)
    all_missing = x_train.columns[x_train.isna().all()]
    if len(all_missing):
        x_train = x_train.drop(columns=all_missing)
        x_test = x_test.drop(columns=all_missing)
    return x_train, x_test


def target_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, recent_days: int) -> np.ndarray:
    global_mean = float(history[target].mean())
    subject_stats = history.groupby("subject_id", observed=True)[target].agg(["sum", "count"])
    smooth = (subject_stats["sum"] + 8.0 * global_mean) / (subject_stats["count"] + 8.0)
    subject_prior = rows["subject_id"].map(smooth).fillna(global_mean).astype(float).to_numpy()

    recent_values = {}
    history_sorted = history.sort_values(["subject_id", "lifelog_date"])
    for sid, group in history_sorted.groupby("subject_id", observed=True):
        tail = group[target].tail(recent_days)
        if len(tail):
            recent_values[sid] = (float(tail.sum()) + 4.0 * global_mean) / (len(tail) + 4.0)
    fallback = pd.Series(subject_prior, index=rows.index)
    recent_mapped = rows["subject_id"].map(recent_values)
    recent_prior = recent_mapped.where(recent_mapped.notna(), fallback).astype(float).to_numpy()
    return clip_proba(0.7 * subject_prior + 0.3 * recent_prior)


def make_lgbm(n_estimators: int = 800) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=9,
        min_child_samples=14,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.08,
        reg_lambda=2.5,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def make_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=4,
        max_features=0.65,
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )


def make_logistic() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.45,
            max_iter=4000,
            solver="lbfgs",
            random_state=SEED,
        ),
    )


def fit_predict_single(
    model_name: str,
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_pred: pd.DataFrame,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    n_estimators: int | None = None,
) -> tuple[np.ndarray, int | None, object]:
    if len(np.unique(y_tr)) == 1:
        p = np.full(len(x_pred), float(y_tr[0]))
        return clip_proba(p), None, None

    if model_name == "lgbm":
        model = make_lgbm(n_estimators or 800)
        fit_kwargs = {}
        if x_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            fit_kwargs = {
                "eval_set": [(x_val, y_val)],
                "eval_metric": "binary_logloss",
                "callbacks": [lgb.early_stopping(80, verbose=False)],
            }
        model.fit(x_tr, y_tr, **fit_kwargs)
        pred = model.predict_proba(x_pred)[:, 1]
        best_iter = getattr(model, "best_iteration_", None)
        return clip_proba(pred), best_iter, model

    if model_name == "extra_trees":
        model = make_pipeline(SimpleImputer(strategy="median"), make_extra_trees())
        model.fit(x_tr, y_tr)
        return clip_proba(model.predict_proba(x_pred)[:, 1]), None, model

    if model_name == "logistic":
        model = make_logistic()
        model.fit(x_tr, y_tr)
        return clip_proba(model.predict_proba(x_pred)[:, 1]), None, model

    raise ValueError(f"Unknown model: {model_name}")


def candidate_model_blends() -> dict[str, tuple[float, float, float]]:
    return {
        "lgbm": (1.0, 0.0, 0.0),
        "extra_trees": (0.0, 1.0, 0.0),
        "logistic": (0.0, 0.0, 1.0),
        "lgbm_extra": (0.65, 0.35, 0.0),
        "lgbm_logistic": (0.70, 0.0, 0.30),
        "tree_heavy": (0.58, 0.32, 0.10),
        "balanced": (0.45, 0.35, 0.20),
        "regularized": (0.42, 0.23, 0.35),
        "equal": (1 / 3, 1 / 3, 1 / 3),
    }


def weighted_prediction(preds: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    names = ["lgbm", "extra_trees", "logistic"]
    total = np.zeros(len(next(iter(preds.values()))), dtype=float)
    for name, weight in zip(names, weights):
        total += weight * preds[name]
    return clip_proba(total)


def run_cv(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    n_splits: int,
    recent_days: int,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    y = train[TARGETS].reset_index(drop=True)
    train_meta = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    train_meta["lifelog_date"] = pd.to_datetime(train_meta["lifelog_date"])

    oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    report: dict[str, dict[str, object]] = {}

    for target in TARGETS:
        y_target = y[target].values
        min_class = int(pd.Series(y_target).value_counts().min())
        folds = max(2, min(n_splits, min_class))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

        model_oof = {
            "lgbm": np.zeros(len(train), dtype=float),
            "extra_trees": np.zeros(len(train), dtype=float),
            "logistic": np.zeros(len(train), dtype=float),
        }
        prior_oof = np.zeros(len(train), dtype=float)
        best_iters = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_target), start=1):
            x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            y_tr, y_va = y_target[tr_idx], y_target[va_idx]

            pred, best_iter, _ = fit_predict_single("lgbm", x_tr, y_tr, x_va, x_va, y_va)
            model_oof["lgbm"][va_idx] = pred
            if best_iter:
                best_iters.append(int(best_iter))

            pred, _, _ = fit_predict_single("extra_trees", x_tr, y_tr, x_va)
            model_oof["extra_trees"][va_idx] = pred

            pred, _, _ = fit_predict_single("logistic", x_tr, y_tr, x_va)
            model_oof["logistic"][va_idx] = pred

            history = train_meta.iloc[tr_idx]
            rows = train_meta.iloc[va_idx]
            prior_oof[va_idx] = target_prior(history, rows, target, recent_days)

        blend_scores = {}
        for name, weights in candidate_model_blends().items():
            pred = weighted_prediction(model_oof, weights)
            blend_scores[name] = binary_log_loss(y_target, pred)
        best_blend_name = min(blend_scores, key=blend_scores.get)
        best_weights = candidate_model_blends()[best_blend_name]
        model_pred = weighted_prediction(model_oof, best_weights)

        alpha_scores = {}
        for alpha in np.linspace(0.0, 1.0, 21):
            pred = clip_proba(alpha * model_pred + (1 - alpha) * prior_oof)
            alpha_scores[float(alpha)] = binary_log_loss(y_target, pred)
        best_alpha = min(alpha_scores, key=alpha_scores.get)
        final_oof = clip_proba(best_alpha * model_pred + (1 - best_alpha) * prior_oof)
        oof[target] = final_oof

        report[target] = {
            "oof_logloss": binary_log_loss(y_target, final_oof),
            "model_only_logloss": binary_log_loss(y_target, model_pred),
            "prior_only_logloss": binary_log_loss(y_target, prior_oof),
            "best_model_blend": best_blend_name,
            "model_weights": {
                "lgbm": best_weights[0],
                "extra_trees": best_weights[1],
                "logistic": best_weights[2],
            },
            "model_alpha": best_alpha,
            "lgbm_n_estimators": int(np.median(best_iters) * 1.08 + 20) if best_iters else 500,
            "blend_candidates": blend_scores,
        }
        print(
            f"[CV] {target}: logloss={report[target]['oof_logloss']:.5f}, "
            f"blend={best_blend_name}, alpha={best_alpha:.2f}"
        )

    return report, oof


def train_final_predict(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    report: dict[str, dict[str, object]],
    recent_days: int,
    output_dir: Path,
) -> pd.DataFrame:
    train_meta = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    train_meta["lifelog_date"] = pd.to_datetime(train_meta["lifelog_date"])
    sample_meta = sample[["subject_id", "lifelog_date"]].copy()
    sample_meta["lifelog_date"] = pd.to_datetime(sample_meta["lifelog_date"])

    submission = sample.copy()
    importances = defaultdict(float)

    for target in TARGETS:
        y = train[target].values
        n_estimators = int(report[target]["lgbm_n_estimators"])
        pred_lgbm, _, lgbm_model = fit_predict_single("lgbm", x_train, y, x_test, n_estimators=n_estimators)
        pred_extra, _, _ = fit_predict_single("extra_trees", x_train, y, x_test)
        pred_log, _, _ = fit_predict_single("logistic", x_train, y, x_test)

        weights = report[target]["model_weights"]
        model_pred = clip_proba(
            weights["lgbm"] * pred_lgbm
            + weights["extra_trees"] * pred_extra
            + weights["logistic"] * pred_log
        )
        prior = target_prior(train_meta, sample_meta, target, recent_days)
        alpha = float(report[target]["model_alpha"])
        submission[target] = clip_proba(alpha * model_pred + (1 - alpha) * prior)

        if lgbm_model is not None:
            for name, value in zip(x_train.columns, lgbm_model.feature_importances_):
                importances[name] += float(value)

    if importances:
        importance_df = (
            pd.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    return submission


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "ch2026_metrics_train.csv")
    sample = pd.read_csv(data_dir / "ch2026_submission_sample.csv")

    print("[1/4] Building feature table...")
    features = build_features(data_dir, output_dir, args.top_apps, args.top_ambience, args.force_features)
    x_train, x_test = make_matrix(features, train, sample)
    print(f"      train matrix={x_train.shape}, test matrix={x_test.shape}")

    print("[2/4] Running target-wise out-of-fold validation...")
    report, oof = run_cv(train, x_train, args.n_splits, args.recent_days)
    oof_score = average_log_loss(train[TARGETS], oof[TARGETS])
    report_summary = {
        "average_oof_logloss": oof_score,
        "n_train": len(train),
        "n_test": len(sample),
        "n_features": x_train.shape[1],
        "targets": report,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    oof_out = train[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for target in TARGETS:
        oof_out[target] = oof[target]
    oof_out.to_csv(output_dir / "oof_predictions.csv", index=False)
    print(f"      Average OOF Log-Loss: {oof_score:.5f}")

    print("[3/4] Training final ensemble on all labeled rows...")
    submission = train_final_predict(train, sample, x_train, x_test, report, args.recent_days, output_dir)
    submission_path = next_submission_path(output_dir)
    submission.to_csv(submission_path, index=False)

    print("[4/4] Done")
    print(f"      submission: {submission_path}")
    print(f"      report:     {output_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
