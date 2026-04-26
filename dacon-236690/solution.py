#!/usr/bin/env python3
"""One-file CH2026 pipeline: preprocess, CV, final fit, inference, submission.

Outputs:
  - model/model.ipynb_bundle.joblib
  - outputs/cv_results.json
  - outputs/oof_predictions.csv
  - outputs/bsh_submission_vN.csv
  - submission.csv
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise ImportError(
        "This pipeline needs pandas, numpy, scikit-learn, pyarrow, and joblib. "
        "Install them with `pip install -r requirements.txt`."
    ) from exc

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

ROOT_DIR = Path.cwd()
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
MERGE_KEYS = ["subject_id", "lifelog_date"]
EPS = 1e-5

CONFIG: dict[str, Any] = {
    "project": {"name": "dacon_236690_ch2026_pipeline", "seed": 42},
    "paths": {
        "train_csv": ROOT_DIR / "ch2026_metrics_train.csv",
        "sample_submission_csv": ROOT_DIR / "ch2026_submission_sample.csv",
        "sensor_dir": ROOT_DIR / "ch2025_data_items",
        "output_dir": ROOT_DIR / "outputs",
        "submission_dir": ROOT_DIR / "outputs" / "submissions",
        "model_path": ROOT_DIR / "model" / "model.ipynb_bundle.joblib",
        "cv_output_json": ROOT_DIR / "outputs" / "reports" / "cv_results.json",
        "oof_output_csv": ROOT_DIR / "outputs" / "reports" / "oof_predictions.csv",
        "feature_cache_parquet": ROOT_DIR / "outputs" / "cache" / "feature_cache.parquet",
        "latest_submission_csv": ROOT_DIR / "submission.csv",
        "submission_prefix": "bsh_submission_v",
    },
    "features": {
        "top_apps": 40,
        "top_ambience": 50,
        "relative_feature_limit": 350,
        "lag_feature_limit": 120,
    },
    "training": {
        "n_splits": 5,
        "recent_days": 10,
        "clip_min": EPS,
        "clip_max": 1.0 - EPS,
        "model_blend_power": 2.5,
        "weak_model_loss_ratio": 1.18,
        "postprocess": {"enabled": True},
        "final_bagging_seeds": [42, 2025, 777],
        "prior": {
            "subject_smoothing": 8.0,
            "recent_smoothing": 4.0,
            "default_recent_weight": 0.3,
            "target_settings": {
                "Q1": {"recent_days": 10, "recent_weight": 0.0},
                "Q2": {"recent_days": 15, "recent_weight": 1.0},
                "S3": {"recent_days": 30, "recent_weight": 1.0},
            },
        },
        "run_cv": True,
        "models": {
            "lightgbm": {"enabled": True, "n_estimators": 800},
            "xgboost": {"enabled": True, "n_estimators": 700},
            "catboost": {"enabled": True, "iterations": 800},
            "extra_trees": {"enabled": True, "n_estimators": 700},
            "logistic": {"enabled": True},
        },
    },
}

FAST_DEV_SETTINGS = {
    "enabled": False,
    "n_splits": 3,
    "max_train_rows": 120,
    "max_test_rows": 80,
    "tree_estimators": 120,
    "force_features": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--n-splits", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "ch2026_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def clip_proba(values: Any, low: float = EPS, high: float = 1.0 - EPS) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), low, high)


def logit(values: Any) -> np.ndarray:
    p = clip_proba(values)
    return np.log(p / (1.0 - p))


def sigmoid(values: Any) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(log_loss(np.asarray(y_true, dtype=int), clip_proba(y_pred), labels=[0, 1]))


def average_log_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return float(np.mean([binary_log_loss(y_true[t].values, y_pred[t].values) for t in TARGETS]))


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_joblib(obj: Any, path: str | Path, compress: int = 3) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path, compress=compress)


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
    daily = flatten_columns(df.groupby(MERGE_KEYS, observed=True).agg(agg_map), prefix)
    pieces = [daily]

    for col in value_cols:
        pivot = df.pivot_table(
            index=MERGE_KEYS,
            columns="time_bin",
            values=col,
            aggfunc=["mean", "sum"],
            observed=True,
        )
        pieces.append(flatten_columns(pivot, f"{prefix}_{col}_by_time"))

    if add_changes:
        tmp = df.sort_values(["subject_id", "lifelog_date", "timestamp"]).copy()
        for col in value_cols:
            tmp[f"{col}_changes"] = tmp.groupby(MERGE_KEYS, observed=True)[col].transform(
                lambda s: s.ne(s.shift()).astype(int)
            )
            pieces.append(
                tmp.groupby(MERGE_KEYS, observed=True)[f"{col}_changes"]
                .sum()
                .to_frame(f"{prefix}_{col}_changes")
            )

    if categorical_col:
        counts = df.groupby([*MERGE_KEYS, categorical_col], observed=True).size().unstack(categorical_col, fill_value=0)
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
    stats = pd.DataFrame([summarize_numeric_list(items, numeric_keys) for items in df[list_col].to_numpy()])
    tmp = pd.concat([df[MERGE_KEYS].reset_index(drop=True), stats.add_prefix(f"{prefix}_row_")], axis=1)
    agg_map = {}
    for col in tmp.columns:
        if col in MERGE_KEYS:
            continue
        if col.endswith("_n") or col.endswith("_sum"):
            agg_map[col] = ["sum", "mean", "max"]
        else:
            agg_map[col] = ["mean", "std", "min", "max"]
    daily = flatten_columns(tmp.groupby(MERGE_KEYS, observed=True).agg(agg_map), prefix)
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

        pieces.append(df.groupby(MERGE_KEYS, observed=True)[list_col].apply(unique_count).to_frame(f"{prefix}_{unique_key}_nunique"))

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
    tmp = pd.concat([df[MERGE_KEYS].reset_index(drop=True), pd.DataFrame(records)], axis=1)
    daily = tmp.groupby(MERGE_KEYS, observed=True).agg(
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
    daily = flatten_columns(
        row_df.groupby(MERGE_KEYS, observed=True).agg(
            {
                "usage_events": ["sum", "mean", "max"],
                "usage_total_time": ["sum", "mean", "max"],
                "usage_max_time": ["mean", "max"],
                "usage_unique_apps": ["sum", "mean", "max"],
            }
        ),
        "m_usage",
    )
    pieces = [daily]
    top_apps = {app for app, _ in app_total.most_common(top_n)}
    if app_records and top_apps:
        app_df = pd.DataFrame(app_records, columns=["subject_id", "lifelog_date", "app", "total_time"])
        app_df = app_df[app_df["app"].isin(top_apps)]
        pivot = app_df.pivot_table(index=MERGE_KEYS, columns="app", values="total_time", aggfunc="sum", fill_value=0)
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
    tmp = pd.concat([df[MERGE_KEYS].reset_index(drop=True), stats], axis=1)
    daily = flatten_columns(
        tmp.groupby(MERGE_KEYS, observed=True).agg(
            {
                "ambience_n": ["sum", "mean", "max"],
                "ambience_prob_sum": ["sum", "mean", "max"],
                "ambience_top_prob": ["mean", "std", "min", "max"],
                "ambience_entropy": ["mean", "std", "min", "max"],
            }
        ),
        "m_ambience",
    )
    pieces = [daily]
    top_labels = tmp["ambience_top_label"].value_counts().head(top_n).index
    label_df = tmp[tmp["ambience_top_label"].isin(top_labels)]
    if not label_df.empty:
        counts = label_df.pivot_table(
            index=MERGE_KEYS,
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


def add_subject_relative_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["_life_dt"] = pd.to_datetime(out["lifelog_date"])
    out = out.sort_values(["subject_id", "_life_dt"]).reset_index(drop=True)
    first = out.groupby("subject_id")["_life_dt"].transform("min")
    out["subject_days_since_first"] = (out["_life_dt"] - first).dt.days
    out["subject_row_order"] = out.groupby("subject_id").cumcount()

    reserved = {"subject_id", "sleep_date", "lifelog_date", "_life_dt"}
    numeric_cols = [c for c in out.columns if c not in reserved and pd.api.types.is_numeric_dtype(out[c])]
    sensor_cols = [
        c
        for c in numeric_cols
        if not c.startswith(("life_", "sleep_", "is_weekend", "subject_idx", "subject_days", "subject_row"))
    ]
    sensor_cols = sensor_cols[: int(config["features"]["relative_feature_limit"])]
    for window in (3, 7):
        rolled = (
            out.groupby("subject_id", observed=True)[sensor_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolled.columns = [f"{c}_roll{window}_mean" for c in sensor_cols]
        out = pd.concat([out, rolled], axis=1)

    lag_limit = int(config["features"]["lag_feature_limit"])
    lagged = out.groupby("subject_id", observed=True)[sensor_cols[:lag_limit]].shift(1)
    lagged.columns = [f"{c}_lag1" for c in sensor_cols[:lag_limit]]
    out = pd.concat([out, lagged], axis=1)
    return out.drop(columns=["_life_dt"])


def build_features(config: Mapping[str, Any], force: bool, logger: logging.Logger) -> pd.DataFrame:
    paths = config["paths"]
    cache_path = Path(paths["feature_cache_parquet"])
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        logger.info(f"Loading cached features: {cache_path}")
        return pd.read_parquet(cache_path)

    train = pd.read_csv(paths["train_csv"])
    sample = pd.read_csv(paths["sample_submission_csv"])
    base = pd.concat([train[KEY_COLUMNS], sample[KEY_COLUMNS]], ignore_index=True)
    base = add_calendar_features(base)
    item_dir = Path(paths["sensor_dir"])

    feature_tables = [
        daily_numeric_features(item_dir / "ch2025_mACStatus.parquet", ["m_charging"], "m_ac", add_changes=True),
        daily_numeric_features(item_dir / "ch2025_mActivity.parquet", ["m_activity"], "m_activity", add_changes=True, categorical_col="m_activity"),
        daily_ambience_features(item_dir / "ch2025_mAmbience.parquet", int(config["features"]["top_ambience"])),
        daily_nested_numeric_features(item_dir / "ch2025_mBle.parquet", "m_ble", ["rssi"], "m_ble", unique_key="address"),
        daily_nested_numeric_features(item_dir / "ch2025_mGps.parquet", "m_gps", ["altitude", "latitude", "longitude", "speed"], "m_gps"),
        daily_numeric_features(item_dir / "ch2025_mLight.parquet", ["m_light"], "m_light"),
        daily_numeric_features(item_dir / "ch2025_mScreenStatus.parquet", ["m_screen_use"], "m_screen", add_changes=True),
        daily_usage_features(item_dir / "ch2025_mUsageStats.parquet", int(config["features"]["top_apps"])),
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
        features = features.merge(table, on=MERGE_KEYS, how="left")
    features = add_subject_relative_features(features, config)
    features.to_parquet(cache_path, index=False)
    logger.info(f"Saved feature cache: {cache_path}")
    return features


def make_matrix(features: pd.DataFrame, train: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.get_dummies(features.copy(), columns=["subject_id"], dtype=float)
    frame = frame.drop(columns=[c for c in ["sleep_date", "lifelog_date"] if c in frame.columns])
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


def prior_settings(config: Mapping[str, Any], target: str) -> tuple[int, float, float, float]:
    prior_cfg = config["training"].get("prior", {})
    target_cfg = prior_cfg.get("target_settings", {}).get(target, {})
    recent_days = int(target_cfg.get("recent_days", config["training"].get("recent_days", 10)))
    recent_weight = float(target_cfg.get("recent_weight", prior_cfg.get("default_recent_weight", 0.3)))
    subject_smoothing = float(prior_cfg.get("subject_smoothing", 8.0))
    recent_smoothing = float(prior_cfg.get("recent_smoothing", 4.0))
    return recent_days, recent_weight, subject_smoothing, recent_smoothing


def target_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, config: Mapping[str, Any]) -> np.ndarray:
    recent_days, recent_weight, subject_smoothing, recent_smoothing = prior_settings(config, target)
    global_mean = float(history[target].mean())
    subject_stats = history.groupby("subject_id", observed=True)[target].agg(["sum", "count"])
    smooth = (subject_stats["sum"] + subject_smoothing * global_mean) / (subject_stats["count"] + subject_smoothing)
    subject_prior = rows["subject_id"].map(smooth).fillna(global_mean).astype(float).to_numpy()
    recent_values = {}
    history_sorted = history.sort_values(["subject_id", "lifelog_date"])
    for sid, group in history_sorted.groupby("subject_id", observed=True):
        tail = group[target].tail(recent_days)
        if len(tail):
            recent_values[sid] = (float(tail.sum()) + recent_smoothing * global_mean) / (len(tail) + recent_smoothing)
    fallback = pd.Series(subject_prior, index=rows.index)
    recent_mapped = rows["subject_id"].map(recent_values)
    recent_prior = recent_mapped.where(recent_mapped.notna(), fallback).astype(float).to_numpy()
    return clip_proba((1.0 - recent_weight) * subject_prior + recent_weight * recent_prior)


def available_model_names(config: Mapping[str, Any]) -> list[str]:
    cfg = config["training"]["models"]
    names = []
    if cfg["lightgbm"].get("enabled", True) and lgb is not None:
        names.append("lightgbm")
    if cfg["xgboost"].get("enabled", True) and XGBClassifier is not None:
        names.append("xgboost")
    if cfg["catboost"].get("enabled", True) and CatBoostClassifier is not None:
        names.append("catboost")
    if cfg["extra_trees"].get("enabled", True):
        names.append("extra_trees")
    if cfg["logistic"].get("enabled", True):
        names.append("logistic")
    return names


def build_estimator(model_name: str, config: Mapping[str, Any], seed: int, fast_dev: Mapping[str, Any] | None = None) -> Any:
    fast_dev = fast_dev or {}
    model_cfg = config["training"]["models"]
    fast_n = int(fast_dev.get("tree_estimators", 0) or 0)
    if model_name == "lightgbm":
        n_estimators = fast_n or int(model_cfg["lightgbm"].get("n_estimators", 800))
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.025,
            num_leaves=11,
            min_child_samples=12,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=2.5,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "xgboost":
        n_estimators = fast_n or int(model_cfg["xgboost"].get("n_estimators", 700))
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=4.0,
            subsample=0.85,
            colsample_bytree=0.82,
            reg_lambda=2.0,
            reg_alpha=0.05,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if model_name == "catboost":
        iterations = fast_n or int(model_cfg["catboost"].get("iterations", 800))
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=iterations,
            learning_rate=0.025,
            depth=4,
            l2_leaf_reg=6.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
        )
    if model_name == "extra_trees":
        n_estimators = fast_n or int(model_cfg["extra_trees"].get("n_estimators", 700))
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=4,
                max_features=0.65,
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if model_name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.45, max_iter=4000, solver="lbfgs", random_state=seed),
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    fast_dev: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, Any]:
    if len(np.unique(y_train)) == 1:
        pred = np.full(len(x_pred), float(y_train[0]))
        return clip_proba(pred), None
    model = build_estimator(model_name, config, seed, fast_dev=fast_dev)
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_pred)[:, 1]
    return clip_proba(pred), model


def predict_with_fitted_model(model: Any, x_pred: pd.DataFrame, fallback: float = 0.5) -> np.ndarray:
    if model is None:
        return clip_proba(np.full(len(x_pred), fallback))
    return clip_proba(model.predict_proba(x_pred)[:, 1])


def blend_model_predictions(
    model_oof: dict[str, np.ndarray],
    y_true: np.ndarray,
    power: float,
    weak_model_loss_ratio: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    model_scores = {name: binary_log_loss(y_true, pred) for name, pred in model_oof.items()}
    best_score = min(model_scores.values())
    kept_names = [
        name for name, score in model_scores.items()
        if score <= best_score * weak_model_loss_ratio
    ]
    if not kept_names:
        kept_names = list(model_scores)
    inv = np.asarray([1.0 / max(model_scores[name], EPS) ** power for name in kept_names], dtype=float)
    weights_arr = inv / inv.sum()
    weights = {name: 0.0 for name in model_scores}
    weights.update({name: float(weight) for name, weight in zip(kept_names, weights_arr)})
    blended = np.zeros(len(y_true), dtype=float)
    for name in kept_names:
        blended += weights[name] * model_oof[name]
    return clip_proba(blended), weights, model_scores


def tune_probability_postprocess(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    pred = clip_proba(pred)
    mean_value = float(y_true.mean())
    best = {
        "method": "none",
        "value": None,
        "logloss": binary_log_loss(y_true, pred),
        "target_mean": mean_value,
    }

    for shrink in np.linspace(0.0, 0.4, 41):
        candidate = clip_proba((1.0 - shrink) * pred + shrink * mean_value)
        score = binary_log_loss(y_true, candidate)
        if score < best["logloss"]:
            best = {"method": "mean_shrink", "value": float(shrink), "logloss": score, "target_mean": mean_value}

    mean_logit = float(logit([mean_value])[0])
    for scale in np.linspace(0.5, 1.35, 86):
        candidate = clip_proba(sigmoid(mean_logit + scale * (logit(pred) - mean_logit)))
        score = binary_log_loss(y_true, candidate)
        if score < best["logloss"]:
            best = {"method": "temperature", "value": float(scale), "logloss": score, "target_mean": mean_value}

    return best


def apply_probability_postprocess(pred: np.ndarray, params: Mapping[str, Any] | None) -> np.ndarray:
    if not params or params.get("method") == "none":
        return clip_proba(pred)
    mean_value = float(params["target_mean"])
    value = float(params["value"])
    if params["method"] == "mean_shrink":
        return clip_proba((1.0 - value) * pred + value * mean_value)
    if params["method"] == "temperature":
        mean_logit = float(logit([mean_value])[0])
        return clip_proba(sigmoid(mean_logit + value * (logit(pred) - mean_logit)))
    raise ValueError(f"Unknown postprocess method: {params['method']}")


def run_cv(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    model_names: list[str],
    config: Mapping[str, Any],
    fast_dev: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed = int(config["project"]["seed"])
    n_splits = int(config["training"]["n_splits"])
    power = float(config["training"]["model_blend_power"])
    weak_model_loss_ratio = float(config["training"]["weak_model_loss_ratio"])
    postprocess_enabled = bool(config["training"].get("postprocess", {}).get("enabled", True))
    train_meta = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    train_meta["lifelog_date"] = pd.to_datetime(train_meta["lifelog_date"])
    oof = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    report: dict[str, Any] = {"targets": {}, "models": model_names}

    for target in TARGETS:
        y_target = train[target].astype(int).to_numpy()
        min_class = int(pd.Series(y_target).value_counts().min())
        folds = max(2, min(n_splits, min_class))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        model_oof = {name: np.zeros(len(train), dtype=float) for name in model_names}
        prior_oof = np.zeros(len(train), dtype=float)

        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_target), start=1):
            logger.info(f"CV target={target} fold={fold_id}/{folds}")
            x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            y_tr = y_target[tr_idx]
            for model_name in model_names:
                pred, _ = fit_predict_model(model_name, x_tr, y_tr, x_va, config, seed + fold_id, fast_dev=fast_dev)
                model_oof[model_name][va_idx] = pred
            prior_oof[va_idx] = target_prior(train_meta.iloc[tr_idx], train_meta.iloc[va_idx], target, config)

        model_pred, model_weights, model_scores = blend_model_predictions(
            model_oof, y_target, power, weak_model_loss_ratio
        )
        alpha_scores = {}
        for alpha in np.linspace(0.0, 1.0, 21):
            pred = clip_proba(alpha * model_pred + (1.0 - alpha) * prior_oof)
            alpha_scores[f"{alpha:.2f}"] = binary_log_loss(y_target, pred)
        best_alpha_key = min(alpha_scores, key=alpha_scores.get)
        best_alpha = float(best_alpha_key)
        final_oof = clip_proba(best_alpha * model_pred + (1.0 - best_alpha) * prior_oof)
        postprocess_params = tune_probability_postprocess(y_target, final_oof) if postprocess_enabled else None
        final_oof = apply_probability_postprocess(final_oof, postprocess_params)
        oof[target] = final_oof

        report["targets"][target] = {
            "oof_logloss": binary_log_loss(y_target, final_oof),
            "model_only_logloss": binary_log_loss(y_target, model_pred),
            "prior_only_logloss": binary_log_loss(y_target, prior_oof),
            "model_scores": model_scores,
            "model_weights": model_weights,
            "model_alpha": best_alpha,
            "postprocess": postprocess_params,
            "alpha_scores": alpha_scores,
        }
        logger.info(
            f"CV target={target} logloss={report['targets'][target]['oof_logloss']:.6f} "
            f"alpha={best_alpha:.2f}"
        )

    report["average_oof_logloss"] = average_log_loss(train[TARGETS], oof[TARGETS])
    return report, oof


@dataclass
class TrainedTargetModel:
    target: str
    models: dict[str, Any]
    model_weights: dict[str, float]
    model_alpha: float


@dataclass
class PipelineBundle:
    feature_columns: list[str]
    target_models: dict[str, TrainedTargetModel]
    config: dict[str, Any]
    cv_results: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_final_and_predict(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_names: list[str],
    cv_results: dict[str, Any] | None,
    config: Mapping[str, Any],
    fast_dev: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, PipelineBundle]:
    seed = int(config["project"]["seed"])
    power = float(config["training"]["model_blend_power"])
    weak_model_loss_ratio = float(config["training"]["weak_model_loss_ratio"])
    bagging_seeds = [int(seed) for seed in config["training"].get("final_bagging_seeds", [seed])]
    train_meta = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    train_meta["lifelog_date"] = pd.to_datetime(train_meta["lifelog_date"])
    sample_meta = sample[["subject_id", "lifelog_date"]].copy()
    sample_meta["lifelog_date"] = pd.to_datetime(sample_meta["lifelog_date"])
    submission = sample[KEY_COLUMNS + TARGETS].copy()
    target_models: dict[str, TrainedTargetModel] = {}
    importances = defaultdict(float)

    for target in TARGETS:
        logger.info(f"Final fit target={target}")
        y = train[target].astype(int).to_numpy()
        fitted_models = {}
        test_preds = {}
        train_preds = {}
        for model_name in model_names:
            seed_preds = []
            seed_train_preds = []
            seed_models = []
            for bag_seed in bagging_seeds:
                test_pred, model = fit_predict_model(model_name, x_train, y, x_test, config, bag_seed, fast_dev=fast_dev)
                seed_preds.append(test_pred)
                seed_models.append(model)
                if not cv_results:
                    seed_train_preds.append(predict_with_fitted_model(model, x_train, fallback=float(y.mean())))
            test_preds[model_name] = clip_proba(np.mean(np.vstack(seed_preds), axis=0))
            fitted_models[model_name] = seed_models
            if not cv_results:
                train_preds[model_name] = clip_proba(np.mean(np.vstack(seed_train_preds), axis=0))
            if model_name == "lightgbm":
                for model in seed_models:
                    if model is not None:
                        for name, value in zip(x_train.columns, model.feature_importances_):
                            importances[name] += float(value)

        if cv_results and target in cv_results.get("targets", {}):
            target_cv = cv_results["targets"][target]
            model_weights = dict(target_cv["model_weights"])
            alpha = float(target_cv["model_alpha"])
        else:
            _, model_weights, _ = blend_model_predictions(train_preds, y, power, weak_model_loss_ratio)
            alpha = 0.80

        model_pred = np.zeros(len(sample), dtype=float)
        for model_name, weight in model_weights.items():
            model_pred += float(weight) * test_preds[model_name]
        prior = target_prior(train_meta, sample_meta, target, config)
        pred = clip_proba(alpha * model_pred + (1.0 - alpha) * prior)
        postprocess_params = None
        if cv_results and target in cv_results.get("targets", {}):
            postprocess_params = cv_results["targets"][target].get("postprocess")
        submission[target] = apply_probability_postprocess(pred, postprocess_params)
        target_models[target] = TrainedTargetModel(target, fitted_models, model_weights, alpha)

    if importances:
        importance_df = (
            pd.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_path = Path(config["paths"]["cv_output_json"]).parent / "feature_importance.csv"
        ensure_dir(importance_path.parent)
        importance_df.to_csv(importance_path, index=False)

    bundle = PipelineBundle(
        feature_columns=x_train.columns.tolist(),
        target_models=target_models,
        config=copy.deepcopy(dict(config)),
        cv_results=cv_results,
        metadata={
            "targets": TARGETS,
            "models": model_names,
            "metric": "Average Log-Loss",
            "n_train": len(train),
            "n_test": len(sample),
            "n_features": x_train.shape[1],
        },
    )
    return submission, bundle


def next_submission_path(output_dir: Path, prefix: str) -> tuple[Path, Path, int]:
    ensure_dir(output_dir)
    counter_name = prefix.removesuffix("_v").rstrip("_")
    counter_path = output_dir / f".{counter_name}_counter"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.csv$")
    versions = []
    for directory in (ROOT_DIR, output_dir):
        for path in directory.glob(f"{prefix}*.csv"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
    if counter_path.exists():
        raw_counter = counter_path.read_text(encoding="utf-8").strip()
        if raw_counter.isdigit():
            versions.append(int(raw_counter))
    next_version = max(versions, default=0) + 1
    return output_dir / f"{prefix}{next_version}.csv", counter_path, next_version


def apply_fast_dev(config: Mapping[str, Any], fast_dev: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    if not fast_dev.get("enabled", False):
        return out
    fast_output_dir = ROOT_DIR / "outputs" / "fast_dev"
    out["paths"]["output_dir"] = fast_output_dir
    out["paths"]["submission_dir"] = fast_output_dir / "submissions"
    out["paths"]["model_path"] = ROOT_DIR / "model" / "model.fast_dev_bundle.joblib"
    out["paths"]["cv_output_json"] = fast_output_dir / "reports" / "cv_results.json"
    out["paths"]["oof_output_csv"] = fast_output_dir / "reports" / "oof_predictions.csv"
    out["paths"]["feature_cache_parquet"] = fast_output_dir / "cache" / "feature_cache.parquet"
    out["paths"]["latest_submission_csv"] = fast_output_dir / "submission_fast_dev.csv"
    out["training"]["n_splits"] = int(fast_dev.get("n_splits", 3))
    for name in ("lightgbm", "xgboost", "extra_trees"):
        if name in out["training"]["models"]:
            out["training"]["models"][name]["n_estimators"] = int(fast_dev.get("tree_estimators", 120))
    if "catboost" in out["training"]["models"]:
        out["training"]["models"]["catboost"]["iterations"] = int(fast_dev.get("tree_estimators", 120))
    return out


def load_competition_data(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config["paths"]["train_csv"])
    sample = pd.read_csv(config["paths"]["sample_submission_csv"])
    return train, sample


def run_full_pipeline(
    config: Mapping[str, Any] | None = None,
    fast_dev: Mapping[str, Any] | None = None,
    force_features: bool = False,
    param_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline.

    Parameters
    ----------
    param_overrides:
        Flat dict of dot-path overrides applied on top of CONFIG before the run.
        Example::

            {
                "training.model_blend_power": 3.0,
                "training.models.lightgbm.n_estimators": 1000,
                "training.prior.subject_smoothing": 10.0,
            }
    """
    raw_config = copy.deepcopy(config or CONFIG)

    # Apply flat dot-path overrides (for auto-tuning)
    if param_overrides:
        for dotpath, value in param_overrides.items():
            keys = dotpath.split(".")
            node = raw_config
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value

    fast_dev = copy.deepcopy(fast_dev or FAST_DEV_SETTINGS)
    cfg = apply_fast_dev(raw_config, fast_dev)
    logger = get_logger("ipynb_pipeline")
    set_seed(int(cfg["project"]["seed"]))
    ensure_dir(cfg["paths"]["output_dir"])

    logger.info("Loading data")
    train, sample = load_competition_data(cfg)
    if fast_dev.get("enabled", False):
        train = train.head(int(fast_dev.get("max_train_rows", len(train)))).reset_index(drop=True)
        sample = sample.head(int(fast_dev.get("max_test_rows", len(sample)))).reset_index(drop=True)
        logger.info(f"Fast dev mode: train={len(train)}, sample={len(sample)}")

    logger.info("Preprocessing and feature extraction")
    features = build_features(cfg, force=force_features or bool(fast_dev.get("force_features", False)), logger=logger)
    if fast_dev.get("enabled", False):
        features = pd.concat(
            [features.iloc[: len(train)], features.iloc[len(pd.read_csv(cfg["paths"]["train_csv"])) : len(pd.read_csv(cfg["paths"]["train_csv"])) + len(sample)]],
            ignore_index=True,
        )
    x_train, x_test = make_matrix(features, train, sample)
    logger.info(f"Feature matrix: train={x_train.shape}, test={x_test.shape}")

    model_names = available_model_names(cfg)
    if not model_names:
        raise RuntimeError("No enabled model is available. Check installed packages and CONFIG['training']['models'].")
    logger.info(f"Enabled models: {', '.join(model_names)}")

    cv_results = None
    oof = None
    if cfg["training"].get("run_cv", True):
        logger.info("Cross-validation")
        cv_results, oof = run_cv(train, x_train, model_names, cfg, fast_dev, logger)
        cv_results.update(
            {
                "metric": "Average Log-Loss",
                "n_train": len(train),
                "n_test": len(sample),
                "n_features": x_train.shape[1],
            }
        )
        save_json(cfg["paths"]["cv_output_json"], cv_results)
        oof_out = train[KEY_COLUMNS].copy()
        for target in TARGETS:
            oof_out[target] = oof[target]
        oof_out.to_csv(cfg["paths"]["oof_output_csv"], index=False)
        logger.info(f"Average OOF Log-Loss: {cv_results['average_oof_logloss']:.6f}")

    logger.info("Final training and inference")
    submission, bundle = fit_final_and_predict(train, sample, x_train, x_test, model_names, cv_results, cfg, fast_dev, logger)
    save_joblib(bundle, cfg["paths"]["model_path"], compress=3)

    numbered_path, counter_path, version = next_submission_path(Path(cfg["paths"]["submission_dir"]), str(cfg["paths"]["submission_prefix"]))
    submission.to_csv(numbered_path, index=False)
    counter_path.write_text(f"{version}\n", encoding="utf-8")
    submission.to_csv(cfg["paths"]["latest_submission_csv"], index=False)
    logger.info(f"Saved numbered submission: {numbered_path}")
    logger.info(f"Saved latest submission: {cfg['paths']['latest_submission_csv']}")
    logger.info(f"Saved model bundle: {cfg['paths']['model_path']}")

    best_logloss = float(cv_results["average_oof_logloss"]) if cv_results else float("inf")
    return {
        "config": cfg,
        "bundle": bundle,
        "cv_results": cv_results,
        "submission": submission,
        "submission_path": numbered_path,
        "latest_submission_path": Path(cfg["paths"]["latest_submission_csv"]),
        "model_path": Path(cfg["paths"]["model_path"]),
        "best_logloss": best_logloss,
    }


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    fast_dev = copy.deepcopy(FAST_DEV_SETTINGS)
    if args.fast_dev:
        fast_dev["enabled"] = True
    if args.skip_cv:
        config["training"]["run_cv"] = False
    if args.n_splits is not None:
        config["training"]["n_splits"] = args.n_splits
    artifacts = run_full_pipeline(config=config, fast_dev=fast_dev, force_features=args.force_features)
    print("submission_path:", artifacts["submission_path"])
    print("latest_submission_path:", artifacts["latest_submission_path"])
    print("model_path:", artifacts["model_path"])
    if artifacts["cv_results"]:
        print("average_oof_logloss:", artifacts["cv_results"]["average_oof_logloss"])


if __name__ == "__main__":
    main()
