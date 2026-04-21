"""
ETRI Lifelog 2026 - MAE-First Advanced Ensemble
================================================
Run AFTER feature caches exist (or let this script build them) to search for
probability models that minimize OOF MAE first, while retaining group-wise
validation as a guardrail against unstable overfitting.

Usage:
    ./.venv/bin/python -u advanced_ensemble.py

Core ideas:
  - Primary validation: subject-wise time-blocked CV (future-date holdout)
  - Guardrail validation: GroupKFold over subjects
  - Primary metric: MAE, tie-breakers Brier then Log-Loss
  - Strong subject prior / calibration / shrinkage / blending search
  - Feature subset screening before expensive tuning
  - Optional extra candidates: TabPFN / AutoGluon (skip if unavailable)
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "ch2025_data_items"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR = OUTPUT_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [t.strip() for t in os.getenv("TARGETS", "Q1,Q2,Q3,S1,S2,S3,S4").split(",") if t.strip()]
EPS = 1e-5
SEED_BASE = int(os.getenv("SEED", "42"))
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "mae").strip().lower()
PRIMARY_TIEBREAKERS = ("brier", "logloss")
N_FOLDS = int(os.getenv("N_FOLDS", "5"))
TIME_FOLDS = int(os.getenv("TIME_FOLDS", "3"))
TIME_WARMUP_FRAC = float(os.getenv("TIME_WARMUP_FRAC", "0.25"))
TIME_MIN_TRAIN = int(os.getenv("TIME_MIN_TRAIN", "8"))
GROUP_GUARDRAIL_PCT = float(os.getenv("GROUP_GUARDRAIL_PCT", "0.02"))
N_OPTUNA = int(os.getenv("N_OPTUNA", "400"))
TUNE_FOLDS = int(os.getenv("TUNE_FOLDS", "2"))
N_SEEDS = int(os.getenv("N_SEEDS", "2"))
FAST_TREES = int(os.getenv("FAST_TREES", "240"))
N_JOBS = -1
MAX_MISSING_RATE = float(os.getenv("MAX_MISSING_RATE", "0.55"))
CORR_THRESHOLD = float(os.getenv("CORR_THRESHOLD", "0.995"))
FEATURE_SUBSETS_RAW = os.getenv("FEATURE_SUBSETS", "40,80,120,200,all")
SCREEN_KEEP = int(os.getenv("SCREEN_KEEP", "2"))
SCREEN_LGB_TREES = int(os.getenv("SCREEN_LGB_TREES", "200"))
ENABLE_PSEUDO = os.getenv("ENABLE_PSEUDO", "0").strip() == "1"
ENABLE_TABPFN = os.getenv("ENABLE_TABPFN", "1").strip() == "1"
ENABLE_AUTOGLUON = os.getenv("ENABLE_AUTOGLUON", "0").strip() == "1"
ENABLE_PYTORCH_MLP = os.getenv("ENABLE_PYTORCH_MLP", "0").strip() == "1"
MAX_TABPFN_FEATURES = int(os.getenv("MAX_TABPFN_FEATURES", "200"))
AUTOGLUON_PRESET = os.getenv("AUTOGLUON_PRESET", "high_quality")
AUTOGLUON_TIME_LIMIT = int(os.getenv("AUTOGLUON_TIME_LIMIT", "120"))
BLEND_GRID_STEPS = int(os.getenv("BLEND_GRID_STEPS", "21"))
BLEND_REFINE_TRIALS = int(os.getenv("BLEND_REFINE_TRIALS", "30"))
STUDY_VERSION = os.getenv("STUDY_VERSION", "mae_v2").strip() or "mae_v2"

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


@dataclass
class FeatureSet:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


@dataclass
class SourceResult:
    name: str
    feature_set: str
    primary_oof: np.ndarray
    primary_mask: np.ndarray
    group_oof: np.ndarray
    test_pred: np.ndarray
    metadata: dict


@dataclass
class CandidateResult:
    name: str
    source_name: str
    feature_set: str
    primary_pred: np.ndarray
    primary_mask: np.ndarray
    group_pred: np.ndarray
    test_pred: np.ndarray
    primary_metrics: dict
    group_metrics: dict
    metadata: dict


# ---------------------------------------------------------------------------
# Shared imports from existing files
# ---------------------------------------------------------------------------

def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


solution_mod = load_module("solution_mod", Path(__file__).parent / "solution.py")
baseline_mod = load_module("baseline_mod", Path(__file__).parent / "best_baseline.py")

build_expanded_features = solution_mod.build_features
add_relative_features = solution_mod.add_relative_features
add_lag_features = solution_mod.add_lag_features
build_compact_features = baseline_mod.build_features
make_compact_matrix = baseline_mod.make_matrix
candidate_model_blends = baseline_mod.candidate_model_blends


# ---------------------------------------------------------------------------
# Basic utils
# ---------------------------------------------------------------------------

def input_path(filename: str) -> Path:
    for base in (ROOT_DIR, DATA_DIR):
        path = base / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in {ROOT_DIR} or {DATA_DIR}")


def next_submission_path(output_dir: Path) -> Path:
    versions = []
    for path in output_dir.glob("submission_v*.csv"):
        suffix = path.stem.removeprefix("submission_v")
        if suffix.isdigit():
            versions.append(int(suffix))
    next_version = max(versions, default=0) + 1
    return output_dir / f"submission_v{next_version}.csv"


def parse_subset_sizes(raw: str) -> list[int | str]:
    out: list[int | str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            out.append("all")
        else:
            out.append(int(token))
    return out or [40, 80, 120, 200, "all"]


FEATURE_SUBSETS = parse_subset_sizes(FEATURE_SUBSETS_RAW)


def clip_proba(x, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def safe_json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def metric_tuple(metrics: dict) -> tuple[float, float, float]:
    if PRIMARY_METRIC == "mae":
        return (
            float(metrics["mae"]),
            float(metrics["brier"]),
            float(metrics["logloss"]),
        )
    if PRIMARY_METRIC == "logloss":
        return (
            float(metrics["logloss"]),
            float(metrics["mae"]),
            float(metrics["brier"]),
        )
    return (
        float(metrics[PRIMARY_METRIC]),
        float(metrics[PRIMARY_TIEBREAKERS[0]]),
        float(metrics[PRIMARY_TIEBREAKERS[1]]),
    )


def probability_metrics(y_true, proba, mask: np.ndarray | None = None, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = clip_proba(proba)
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(proba)
    if not np.any(valid):
        return {
            "coverage": 0.0,
            "mae": float("inf"),
            "brier": float("inf"),
            "logloss": float("inf"),
            "macro_f1": 0.0,
        }
    y = y_true[valid]
    p = clip_proba(proba[valid])
    return {
        "coverage": float(valid.mean()),
        "mae": float(mean_absolute_error(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "macro_f1": float(f1_score(y, (p >= threshold).astype(int), average="macro")),
    }


def combine_primary_with_group(primary_pred, primary_mask, group_pred):
    primary_pred = np.asarray(primary_pred, dtype=float)
    group_pred = np.asarray(group_pred, dtype=float)
    primary_mask = np.asarray(primary_mask, dtype=bool)
    out = group_pred.copy()
    out[primary_mask] = primary_pred[primary_mask]
    return clip_proba(out)


def positive_scale_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / max(pos, 1.0)


def ensure_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    out = frame[numeric_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def impute_with_train_means(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = train_df.mean(axis=0, numeric_only=True)
    means = means.fillna(0.0)
    return train_df.fillna(means), test_df.fillna(means)


# ---------------------------------------------------------------------------
# Feature loading / screening
# ---------------------------------------------------------------------------

def load_expanded_feature_frame(metrics_train: pd.DataFrame, submission: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = OUTPUT_DIR / "solution_features.csv"
    parquet_path = OUTPUT_DIR / "solution_features.parquet"
    if csv_path.exists():
        feat_df = pd.read_csv(csv_path)
        print(f"Loading expanded feature cache: {csv_path}")
    elif parquet_path.exists():
        feat_df = pd.read_parquet(parquet_path)
        print(f"Loading expanded feature cache: {parquet_path}")
    else:
        print("Building expanded feature cache via solution.py...")
        all_metrics = pd.concat(
            [
                metrics_train[["subject_id", "lifelog_date", "sleep_date"]],
                submission[["subject_id", "lifelog_date", "sleep_date"]],
            ],
            ignore_index=True,
        )
        feat_df = build_expanded_features(all_metrics, verbose=True)
        feat_df.to_csv(csv_path, index=False)
        try:
            feat_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    print("Adding expanded relative/lag features...")
    feat_df = add_relative_features(feat_df)
    feat_df = add_lag_features(feat_df, lags=[1, 2, 3])
    feat_df = feat_df.merge(
        metrics_train[["subject_id", "lifelog_date"] + TARGETS],
        on=["subject_id", "lifelog_date"],
        how="left",
    )
    feat_df = feat_df.merge(
        pd.concat(
            [
                metrics_train[["subject_id", "lifelog_date"]].assign(split="train"),
                submission[["subject_id", "lifelog_date"]].assign(split="test"),
            ],
            ignore_index=True,
        ),
        on=["subject_id", "lifelog_date"],
        how="left",
    )

    drop_cols = ["subject_id", "lifelog_date", "sleep_date", "split"] + TARGETS
    feat_cols = [c for c in feat_df.columns if c not in drop_cols and feat_df[c].dtype != object]
    train_frame = ensure_numeric_frame(feat_df.loc[feat_df["split"] == "train", feat_cols].reset_index(drop=True))
    test_frame = ensure_numeric_frame(feat_df.loc[feat_df["split"] == "test", feat_cols].reset_index(drop=True))
    train_frame, test_frame = impute_with_train_means(train_frame, test_frame)
    return train_frame, test_frame


def load_compact_feature_frame(metrics_train: pd.DataFrame, submission: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    compact_cache_dir = CACHE_DIR / "compact"
    compact_cache_dir.mkdir(parents=True, exist_ok=True)

    existing_parquet = OUTPUT_DIR / "feature_cache.parquet"
    existing_csv = OUTPUT_DIR / "feature_cache.csv"
    if existing_parquet.exists():
        features = pd.read_parquet(existing_parquet)
        print(f"Loading compact feature cache: {existing_parquet}")
    elif existing_csv.exists():
        features = pd.read_csv(existing_csv)
        print(f"Loading compact feature cache: {existing_csv}")
    else:
        stage_dir = CACHE_DIR / "compact_stage"
        item_stage = stage_dir / "ch2025_data_items"
        stage_dir.mkdir(parents=True, exist_ok=True)
        item_stage.mkdir(parents=True, exist_ok=True)

        train_src = input_path("ch2026_metrics_train.csv")
        sample_src = input_path("ch2026_submission_sample.csv")
        train_dst = stage_dir / "ch2026_metrics_train.csv"
        sample_dst = stage_dir / "ch2026_submission_sample.csv"
        if not train_dst.exists():
            train_dst.symlink_to(train_src.resolve())
        if not sample_dst.exists():
            sample_dst.symlink_to(sample_src.resolve())

        for parquet_src in sorted((DATA_DIR).glob("*.parquet")):
            parquet_dst = item_stage / parquet_src.name
            if not parquet_dst.exists():
                parquet_dst.symlink_to(parquet_src.resolve())

        features = build_compact_features(stage_dir, compact_cache_dir, top_apps=40, top_ambience=50, force=False)
    x_train, x_test = make_compact_matrix(features, metrics_train, submission)
    x_train = ensure_numeric_frame(x_train)
    x_test = ensure_numeric_frame(x_test)
    x_train, x_test = impute_with_train_means(x_train, x_test)
    return x_train.reset_index(drop=True), x_test.reset_index(drop=True)


def filter_columns_for_stability(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing_rate = train_df.isna().mean(axis=0)
    std = train_df.std(axis=0, numeric_only=True).fillna(0.0)
    keep = (~(missing_rate > MAX_MISSING_RATE)) & (std > 0)
    kept_cols = train_df.columns[keep].tolist()
    info = {
        "input_features": int(train_df.shape[1]),
        "kept_features": int(len(kept_cols)),
        "dropped_high_missing": int((missing_rate > MAX_MISSING_RATE).sum()),
        "dropped_constant": int((std <= 0).sum()),
    }
    return train_df[kept_cols].copy(), test_df[kept_cols].copy(), info


def quick_feature_ranking(train_df: pd.DataFrame, train_meta: pd.DataFrame, time_folds: list[tuple[np.ndarray, np.ndarray]]) -> list[str]:
    importance = pd.Series(0.0, index=train_df.columns)
    params = {
        "objective": "binary",
        "n_estimators": SCREEN_LGB_TREES,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": SEED_BASE,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }
    for target in TARGETS:
        y = train_meta[target].values.astype(int)
        for fold_idx, (tr_idx, va_idx) in enumerate(time_folds):
            y_tr = y[tr_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            model = lgb.LGBMClassifier(**params)
            model.fit(train_df.iloc[tr_idx], y_tr)
            importance += pd.Series(model.feature_importances_, index=train_df.columns, dtype=float)
            del model
        gc.collect()
    ranked = importance.sort_values(ascending=False).index.tolist()
    return ranked


def correlation_prune(train_df: pd.DataFrame, ranked_cols: list[str]) -> list[str]:
    if not ranked_cols:
        return []
    corr = train_df[ranked_cols].corr().abs().fillna(0.0)
    selected: list[str] = []
    for col in ranked_cols:
        if not selected:
            selected.append(col)
            continue
        if corr.loc[col, selected].max() <= CORR_THRESHOLD:
            selected.append(col)
    return selected


def build_feature_sets(metrics_train: pd.DataFrame, submission: pd.DataFrame) -> tuple[dict[str, FeatureSet], dict]:
    expanded_train, expanded_test = load_expanded_feature_frame(metrics_train, submission)
    compact_train, compact_test = load_compact_feature_frame(metrics_train, submission)

    expanded_train, expanded_test, filter_info = filter_columns_for_stability(expanded_train, expanded_test)

    time_folds = make_time_blocked_folds(metrics_train)
    ranked = quick_feature_ranking(expanded_train, metrics_train, time_folds)
    pruned_ranked = correlation_prune(expanded_train, ranked)

    ranking_path = CACHE_DIR / "expanded_feature_ranking.csv"
    pd.DataFrame({"feature": pruned_ranked}).to_csv(ranking_path, index=False)

    feature_sets: dict[str, FeatureSet] = {}
    for size in FEATURE_SUBSETS:
        if size == "all":
            cols = pruned_ranked[:]
            name = "expanded_all"
        else:
            cols = pruned_ranked[: int(size)]
            name = f"expanded_top{int(size)}"
        if not cols:
            continue
        feature_sets[name] = FeatureSet(
            name=name,
            train=expanded_train[cols].copy(),
            test=expanded_test[cols].copy(),
            metadata={"family": "expanded", "feature_count": len(cols)},
        )

    feature_sets["compact_all"] = FeatureSet(
        name="compact_all",
        train=compact_train.copy(),
        test=compact_test.copy(),
        metadata={"family": "compact", "feature_count": int(compact_train.shape[1])},
    )

    info = {
        "expanded_filter": filter_info,
        "expanded_ranked_feature_count": int(len(pruned_ranked)),
        "ranking_path": str(ranking_path),
    }
    return feature_sets, info


# ---------------------------------------------------------------------------
# CV schemes
# ---------------------------------------------------------------------------

def make_time_blocked_folds(train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[list[int], list[int]]] = []
    for _ in range(TIME_FOLDS):
        folds.append(([], []))

    order_df = train_df[["subject_id", "lifelog_date"]].copy()
    order_df["_life_dt"] = pd.to_datetime(order_df["lifelog_date"])
    order_df = order_df.sort_values(["subject_id", "_life_dt"]).reset_index()

    for _, subject_rows in order_df.groupby("subject_id", observed=True):
        idxs = subject_rows["index"].to_numpy()
        n = len(idxs)
        warmup = max(TIME_MIN_TRAIN, int(np.floor(n * TIME_WARMUP_FRAC)))
        warmup = min(max(warmup, 1), n - 1)
        future = idxs[warmup:]
        parts = [part for part in np.array_split(future, TIME_FOLDS) if len(part)]
        for fold_idx, part in enumerate(parts[:TIME_FOLDS]):
            start = int(np.where(idxs == part[0])[0][0])
            train_part = idxs[:start]
            if len(train_part) < TIME_MIN_TRAIN:
                continue
            folds[fold_idx][0].extend(train_part.tolist())
            folds[fold_idx][1].extend(part.tolist())

    ready: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in folds:
        if tr_idx and va_idx:
            ready.append((np.asarray(tr_idx, dtype=int), np.asarray(va_idx, dtype=int)))
    if not ready:
        raise RuntimeError("Could not create time-blocked folds.")
    return ready


def make_group_folds(train_df: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = train_df["subject_id"].values
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
    return [(tr_idx, va_idx) for tr_idx, va_idx in gkf.split(np.zeros(len(y)), y, groups)]


# ---------------------------------------------------------------------------
# Prior variants
# ---------------------------------------------------------------------------

def compute_subject_mean_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, smooth: float = 8.0) -> np.ndarray:
    global_mean = float(history[target].mean())
    stats = history.groupby("subject_id", observed=True)[target].agg(["sum", "count"])
    smooth_mean = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
    return clip_proba(rows["subject_id"].map(smooth_mean).fillna(global_mean).astype(float).to_numpy())


def compute_recent_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, recent_days: int, smooth: float = 4.0) -> np.ndarray:
    global_mean = float(history[target].mean())
    history = history.copy()
    history["lifelog_date"] = pd.to_datetime(history["lifelog_date"])
    recent_values: dict[str, float] = {}
    for sid, group in history.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id", observed=True):
        tail = group[target].tail(recent_days)
        if len(tail):
            recent_values[sid] = (float(tail.sum()) + smooth * global_mean) / (len(tail) + smooth)
    base = compute_subject_mean_prior(history, rows, target)
    recent = rows["subject_id"].map(recent_values)
    out = recent.where(recent.notna(), pd.Series(base, index=rows.index)).astype(float).to_numpy()
    return clip_proba(out)


def compute_global_mean_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str) -> np.ndarray:
    global_mean = float(history[target].mean())
    return clip_proba(np.full(len(rows), global_mean, dtype=float))


def compute_exp_decay_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, half_life_days: float = 7.0) -> np.ndarray:
    history = history.copy()
    rows = rows.copy()
    history["lifelog_date"] = pd.to_datetime(history["lifelog_date"])
    rows["lifelog_date"] = pd.to_datetime(rows["lifelog_date"])
    global_mean = float(history[target].mean())
    out = np.full(len(rows), global_mean, dtype=float)
    for pos, (_, row) in enumerate(rows.iterrows()):
        sub_hist = history[history["subject_id"] == row["subject_id"]]
        if sub_hist.empty:
            continue
        delta = (row["lifelog_date"] - sub_hist["lifelog_date"]).dt.days.to_numpy(dtype=float)
        mask = delta >= 0
        if not np.any(mask):
            continue
        weights = 0.5 ** (delta[mask] / max(half_life_days, 1e-6))
        values = sub_hist.loc[mask, target].to_numpy(dtype=float)
        out[pos] = float(np.average(values, weights=weights)) if weights.sum() > 0 else global_mean
    return clip_proba(out)


def compute_weekday_prior(history: pd.DataFrame, rows: pd.DataFrame, target: str, smooth: float = 3.0) -> np.ndarray:
    history = history.copy()
    rows = rows.copy()
    history["lifelog_date"] = pd.to_datetime(history["lifelog_date"])
    rows["lifelog_date"] = pd.to_datetime(rows["lifelog_date"])
    history["dow"] = history["lifelog_date"].dt.dayofweek
    rows["dow"] = rows["lifelog_date"].dt.dayofweek
    global_mean = float(history[target].mean())
    grp = history.groupby(["subject_id", "dow"], observed=True)[target].agg(["sum", "count"])
    smooth_mean = (grp["sum"] + smooth * global_mean) / (grp["count"] + smooth)
    keys = list(zip(rows["subject_id"], rows["dow"]))
    return clip_proba(np.asarray([smooth_mean.get(key, global_mean) for key in keys], dtype=float))


PRIOR_VARIANTS: dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], np.ndarray]] = {
    "global_mean": lambda history, rows, target: compute_global_mean_prior(history, rows, target),
    "subject_mean": lambda history, rows, target: compute_subject_mean_prior(history, rows, target, smooth=8.0),
    "recent_3": lambda history, rows, target: compute_recent_prior(history, rows, target, recent_days=3, smooth=4.0),
    "recent_5": lambda history, rows, target: compute_recent_prior(history, rows, target, recent_days=5, smooth=4.0),
    "recent_7": lambda history, rows, target: compute_recent_prior(history, rows, target, recent_days=7, smooth=4.0),
    "recent_10": lambda history, rows, target: compute_recent_prior(history, rows, target, recent_days=10, smooth=4.0),
    "exp_decay_7": lambda history, rows, target: compute_exp_decay_prior(history, rows, target, half_life_days=7.0),
    "weekday_subject": lambda history, rows, target: compute_weekday_prior(history, rows, target, smooth=3.0),
}


def compute_prior_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variant_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variant = PRIOR_VARIANTS[variant_name]
    pred = np.full(len(train_df), np.nan, dtype=float)
    mask = np.zeros(len(train_df), dtype=bool)
    for tr_idx, va_idx in folds:
        history = train_df.iloc[tr_idx][["subject_id", "lifelog_date", target]].copy()
        rows = train_df.iloc[va_idx][["subject_id", "lifelog_date"]].copy()
        pred[va_idx] = variant(history, rows, target)
        mask[va_idx] = True
    test_pred = variant(
        train_df[["subject_id", "lifelog_date", target]].copy(),
        test_df[["subject_id", "lifelog_date"]].copy(),
        target,
    )
    return clip_proba(pred), mask, clip_proba(test_pred)


# ---------------------------------------------------------------------------
# Model builders / tuning
# ---------------------------------------------------------------------------

def lgb_params_from_trial(trial):
    return {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 150, max(150, FAST_TREES)),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "weight_mode": trial.suggest_categorical("weight_mode", ["none", "balanced", "scale_pos_weight"]),
    }


def resolve_lgb_params(params, y_tr: np.ndarray, seed: int):
    out = dict(params)
    weight_mode = out.pop("weight_mode", "none")
    out.update({"random_state": seed, "n_jobs": N_JOBS, "verbose": -1})
    if weight_mode == "balanced":
        out["class_weight"] = "balanced"
    elif weight_mode == "scale_pos_weight":
        out["scale_pos_weight"] = positive_scale_weight(y_tr)
    return out


def xgb_params_from_trial(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, max(150, FAST_TREES)),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "weight_mode": trial.suggest_categorical("weight_mode", ["none", "scale_pos_weight"]),
    }


def resolve_xgb_params(params, y_tr: np.ndarray, seed: int):
    out = dict(params)
    weight_mode = out.pop("weight_mode", "none")
    out.update({
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": N_JOBS,
        "verbosity": 0,
    })
    if weight_mode == "scale_pos_weight":
        out["scale_pos_weight"] = positive_scale_weight(y_tr)
    return out


def cat_params_from_trial(trial):
    return {
        "iterations": trial.suggest_int("iterations", 150, max(150, FAST_TREES)),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "weight_mode": trial.suggest_categorical("weight_mode", ["none", "balanced"]),
    }


def resolve_cat_params(params, seed: int):
    out = dict(params)
    weight_mode = out.pop("weight_mode", "none")
    out.update({
        "loss_function": "Logloss",
        "allow_writing_files": False,
        "random_seed": seed,
        "verbose": 0,
    })
    if weight_mode == "balanced":
        out["auto_class_weights"] = "Balanced"
    return out


DEFAULT_MODEL_PARAMS = {
    "lgb": {
        "objective": "binary",
        "n_estimators": min(max(FAST_TREES, 180), 320),
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "weight_mode": "none",
    },
    "xgb": {
        "n_estimators": min(max(FAST_TREES, 180), 320),
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "weight_mode": "none",
    },
    "cat": {
        "iterations": min(max(FAST_TREES, 180), 320),
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "weight_mode": "none",
    },
}


def get_study_storage(target: str, feature_set: str, model_name: str) -> str:
    db_name = (
        f"{STUDY_VERSION}__ft{FAST_TREES}__tf{TUNE_FOLDS}"
        f"__{target}__{feature_set}__{model_name}.sqlite3"
    )
    db_path = (OPTUNA_DIR / db_name).resolve()
    return f"sqlite:///{db_path}"


def optimize_single_metric(model_name: str, params: dict, X: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        pred_val, _ = fit_predict_single_model(model_name, X_tr, y_tr, X_va, None, params, SEED_BASE + fold_idx)
        fold_scores.append(probability_metrics(y_va, pred_val)[PRIMARY_METRIC])
    return float(np.mean(fold_scores))


def tune_model(model_name: str, target: str, feature_set: str, X: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, float]:
    if model_name not in ("lgb", "xgb", "cat"):
        return DEFAULT_MODEL_PARAMS.get(model_name, {}), float("nan")

    def build_params(trial):
        if model_name == "lgb":
            return lgb_params_from_trial(trial)
        if model_name == "xgb":
            return xgb_params_from_trial(trial)
        return cat_params_from_trial(trial)

    def objective(trial: optuna.Trial) -> float:
        params = build_params(trial)
        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds[: max(1, min(TUNE_FOLDS, len(folds)))]):
            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]
            pred_val, _ = fit_predict_single_model(model_name, X_tr, y_tr, X_va, None, params, SEED_BASE + fold_idx)
            fold_metric = probability_metrics(y_va, pred_val)[PRIMARY_METRIC]
            scores.append(fold_metric)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=SEED_BASE,
        multivariate=True,
        group=True,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, min(30, N_OPTUNA // 10)), n_warmup_steps=0)
    study = optuna.create_study(
        study_name=f"{STUDY_VERSION}__ft{FAST_TREES}__tf{TUNE_FOLDS}__{target}__{feature_set}__{model_name}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=get_study_storage(target, feature_set, model_name),
        load_if_exists=True,
    )
    remaining = max(0, N_OPTUNA - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    best_params = study.best_params if study.best_trial is not None else DEFAULT_MODEL_PARAMS[model_name]
    best_value = float(study.best_value if study.best_trial is not None else np.nan)
    return best_params, best_value


# ---------------------------------------------------------------------------
# Model prediction wrappers
# ---------------------------------------------------------------------------

def constant_prediction(y_tr: np.ndarray, size: int) -> np.ndarray:
    return clip_proba(np.full(size, float(np.mean(y_tr) if len(y_tr) else 0.5), dtype=float))


def fit_predict_single_model(
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_pred: pd.DataFrame,
    X_test: pd.DataFrame | None,
    params: dict | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    y_tr = np.asarray(y_tr, dtype=int)
    if len(np.unique(y_tr)) < 2:
        pred_val = constant_prediction(y_tr, len(X_pred))
        pred_test = constant_prediction(y_tr, len(X_test)) if X_test is not None else None
        return pred_val, pred_test

    params = dict(params or {})

    try:
        if model_name == "lgb":
            model = lgb.LGBMClassifier(**resolve_lgb_params(params or DEFAULT_MODEL_PARAMS["lgb"], y_tr, seed))
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "xgb":
            model = xgb.XGBClassifier(**resolve_xgb_params(params or DEFAULT_MODEL_PARAMS["xgb"], y_tr, seed))
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "cat":
            model = cb.CatBoostClassifier(**resolve_cat_params(params or DEFAULT_MODEL_PARAMS["cat"], seed))
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=max(200, min(FAST_TREES, 500)),
                max_depth=None,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight=None,
                random_state=seed,
                n_jobs=N_JOBS,
            )
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "et":
            model = ExtraTreesClassifier(
                n_estimators=max(250, min(FAST_TREES, 600)),
                max_depth=None,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight=None,
                random_state=seed,
                n_jobs=N_JOBS,
            )
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "logistic":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.45,
                            max_iter=4000,
                            solver="lbfgs",
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            val = clip_proba(model.predict_proba(X_pred)[:, 1])
            test = clip_proba(model.predict_proba(X_test)[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "tabpfn":
            from tabpfn import TabPFNClassifier  # type: ignore

            model = TabPFNClassifier(device="auto")
            model.fit(X_tr.to_numpy(dtype=np.float32), y_tr)
            val = clip_proba(model.predict_proba(X_pred.to_numpy(dtype=np.float32))[:, 1])
            test = clip_proba(model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]) if X_test is not None else None
            return val, test

        if model_name == "autogluon":
            from autogluon.tabular import TabularPredictor  # type: ignore

            workdir = CACHE_DIR / "autogluon_runtime" / f"seed_{seed}"
            workdir.mkdir(parents=True, exist_ok=True)
            train_df = X_tr.copy()
            train_df["label"] = y_tr
            predictor = TabularPredictor(
                label="label",
                problem_type="binary",
                eval_metric="log_loss",
                path=str(workdir),
            ).fit(
                train_data=train_df,
                presets=AUTOGLUON_PRESET,
                time_limit=AUTOGLUON_TIME_LIMIT,
                keep_only_best=True,
                save_space=True,
                verbosity=0,
            )
            val = clip_proba(predictor.predict_proba(X_pred)[1].to_numpy())
            test = clip_proba(predictor.predict_proba(X_test)[1].to_numpy()) if X_test is not None else None
            return val, test

    except Exception:
        pred_val = constant_prediction(y_tr, len(X_pred))
        pred_test = constant_prediction(y_tr, len(X_test)) if X_test is not None else None
        return pred_val, pred_test

    raise ValueError(f"Unknown model_name: {model_name}")


def cross_validated_source(
    model_name: str,
    feature_set: FeatureSet,
    y: np.ndarray,
    primary_folds: list[tuple[np.ndarray, np.ndarray]],
    group_folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict | None,
    target: str,
) -> SourceResult:
    n = len(y)
    seeds = [SEED_BASE] if model_name in {"logistic", "tabpfn", "autogluon"} else [SEED_BASE + i for i in range(N_SEEDS)]

    primary_sum = np.zeros(n, dtype=float)
    primary_count = np.zeros(n, dtype=float)
    group_sum = np.zeros(n, dtype=float)
    group_count = np.zeros(n, dtype=float)
    test_preds = []

    for seed in seeds:
        for tr_idx, va_idx in primary_folds:
            pred_val, _ = fit_predict_single_model(
                model_name,
                feature_set.train.iloc[tr_idx],
                y[tr_idx],
                feature_set.train.iloc[va_idx],
                None,
                params,
                seed,
            )
            primary_sum[va_idx] += pred_val
            primary_count[va_idx] += 1.0

        for tr_idx, va_idx in group_folds:
            pred_val, _ = fit_predict_single_model(
                model_name,
                feature_set.train.iloc[tr_idx],
                y[tr_idx],
                feature_set.train.iloc[va_idx],
                None,
                params,
                seed,
            )
            group_sum[va_idx] += pred_val
            group_count[va_idx] += 1.0

        _, pred_test = fit_predict_single_model(
            model_name,
            feature_set.train,
            y,
            feature_set.train.iloc[:1],
            feature_set.test,
            params,
            seed,
        )
        if pred_test is not None:
            test_preds.append(pred_test)

    primary_mask = primary_count > 0
    group_mask = group_count > 0
    primary_oof = np.full(n, np.nan, dtype=float)
    group_oof = np.full(n, np.nan, dtype=float)
    primary_oof[primary_mask] = primary_sum[primary_mask] / primary_count[primary_mask]
    group_oof[group_mask] = group_sum[group_mask] / group_count[group_mask]
    if test_preds:
        test_pred = clip_proba(np.mean(np.vstack(test_preds), axis=0))
    else:
        test_pred = clip_proba(np.full(len(feature_set.test), float(np.mean(y)), dtype=float))

    return SourceResult(
        name=f"{feature_set.name}::{model_name}",
        feature_set=feature_set.name,
        primary_oof=clip_proba(np.where(primary_mask, primary_oof, np.nanmean(group_oof[group_mask]) if np.any(group_mask) else np.mean(y))),
        primary_mask=primary_mask,
        group_oof=clip_proba(np.where(group_mask, group_oof, np.mean(y))),
        test_pred=test_pred,
        metadata={"model_name": model_name, "feature_count": int(feature_set.train.shape[1]), "target": target},
    )


def blend_source_results(name: str, results: list[SourceResult], weights: list[float]) -> SourceResult:
    primary = np.zeros_like(results[0].primary_oof, dtype=float)
    group = np.zeros_like(results[0].group_oof, dtype=float)
    test = np.zeros_like(results[0].test_pred, dtype=float)
    for weight, result in zip(weights, results):
        primary += weight * result.primary_oof
        group += weight * result.group_oof
        test += weight * result.test_pred
    primary_mask = np.logical_and.reduce([r.primary_mask for r in results])
    return SourceResult(
        name=name,
        feature_set=results[0].feature_set,
        primary_oof=clip_proba(primary),
        primary_mask=primary_mask,
        group_oof=clip_proba(group),
        test_pred=clip_proba(test),
        metadata={"blend_weights": {r.name: float(w) for r, w in zip(results, weights)}},
    )


# ---------------------------------------------------------------------------
# Calibration / post-processing
# ---------------------------------------------------------------------------

def calibrate_sigmoid(pred, y_true, mask) -> tuple[np.ndarray, LogisticRegression | None]:
    pred = clip_proba(pred)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(pred)
    y = np.asarray(y_true, dtype=int)[valid]
    x = pred[valid]
    counts = np.bincount(y, minlength=2)
    n_splits = int(min(5, counts.min())) if len(y) else 0
    if n_splits < 2:
        return pred.copy(), None
    calibrated = pred.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_BASE)
    for tr_idx, va_idx in skf.split(x.reshape(-1, 1), y):
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED_BASE)
        model.fit(x[tr_idx].reshape(-1, 1), y[tr_idx])
        calibrated_idx = np.flatnonzero(valid)[va_idx]
        calibrated[calibrated_idx] = model.predict_proba(x[va_idx].reshape(-1, 1))[:, 1]
    final_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED_BASE)
    final_model.fit(x.reshape(-1, 1), y)
    return clip_proba(calibrated), final_model


def calibrate_isotonic(pred, y_true, mask) -> tuple[np.ndarray, IsotonicRegression | None]:
    pred = clip_proba(pred)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(pred)
    y = np.asarray(y_true, dtype=int)[valid]
    x = pred[valid]
    counts = np.bincount(y, minlength=2)
    n_splits = int(min(5, counts.min())) if len(y) else 0
    if n_splits < 3:
        return pred.copy(), None
    calibrated = pred.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_BASE)
    for tr_idx, va_idx in skf.split(x.reshape(-1, 1), y):
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x[tr_idx], y[tr_idx])
        calibrated_idx = np.flatnonzero(valid)[va_idx]
        calibrated[calibrated_idx] = model.predict(x[va_idx])
    final_model = IsotonicRegression(out_of_bounds="clip")
    final_model.fit(x, y)
    return clip_proba(calibrated), final_model


def apply_calibrator(model, pred):
    pred = clip_proba(pred)
    if model is None:
        return pred
    if isinstance(model, LogisticRegression):
        return clip_proba(model.predict_proba(pred.reshape(-1, 1))[:, 1])
    return clip_proba(model.predict(pred))


def refine_alpha_with_optuna(y_true, primary_base, primary_prior, primary_mask, alpha0: float) -> float:
    if BLEND_REFINE_TRIALS <= 0:
        return alpha0
    lower = max(0.0, alpha0 - 0.15)
    upper = min(1.0, alpha0 + 0.15)
    if upper - lower < 1e-6:
        return alpha0

    def objective(trial):
        alpha = trial.suggest_float("alpha", lower, upper)
        pred = clip_proba(alpha * primary_base + (1.0 - alpha) * primary_prior)
        return probability_metrics(y_true, pred, primary_mask)[PRIMARY_METRIC]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED_BASE))
    study.optimize(objective, n_trials=BLEND_REFINE_TRIALS, show_progress_bar=False)
    return float(study.best_params.get("alpha", alpha0))


def make_candidate(
    name: str,
    source_name: str,
    feature_set: str,
    primary_pred: np.ndarray,
    primary_mask: np.ndarray,
    group_pred: np.ndarray,
    y_true: np.ndarray,
    test_pred: np.ndarray,
    metadata: dict,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        source_name=source_name,
        feature_set=feature_set,
        primary_pred=clip_proba(primary_pred),
        primary_mask=np.asarray(primary_mask, dtype=bool),
        group_pred=clip_proba(group_pred),
        test_pred=clip_proba(test_pred),
        primary_metrics=probability_metrics(y_true, primary_pred, mask=primary_mask),
        group_metrics=probability_metrics(y_true, group_pred),
        metadata=metadata,
    )


def postprocess_source(
    source: SourceResult,
    y_true: np.ndarray,
    prior_sources: dict[str, SourceResult],
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    target_mean = float(np.mean(y_true))

    # Raw / clip-only variants.
    for eps in (1e-5, 1e-4, 1e-3, 0.01, 0.02):
        candidates.append(
            make_candidate(
                name=f"{source.name}::raw_clip_{eps:g}",
                source_name=source.name,
                feature_set=source.feature_set,
                primary_pred=clip_proba(source.primary_oof, eps),
                primary_mask=source.primary_mask,
                group_pred=clip_proba(source.group_oof, eps),
                y_true=y_true,
                test_pred=clip_proba(source.test_pred, eps),
                metadata={"postprocess": "clip", "eps": eps},
            )
        )

    # Mean shrinkage.
    for shrink in (0.02, 0.05, 0.10, 0.15):
        candidates.append(
            make_candidate(
                name=f"{source.name}::target_mean_shrink_{shrink:.2f}",
                source_name=source.name,
                feature_set=source.feature_set,
                primary_pred=(1.0 - shrink) * source.primary_oof + shrink * target_mean,
                primary_mask=source.primary_mask,
                group_pred=(1.0 - shrink) * source.group_oof + shrink * target_mean,
                y_true=y_true,
                test_pred=(1.0 - shrink) * source.test_pred + shrink * target_mean,
                metadata={"postprocess": "mean_shrink", "shrink": shrink},
            )
        )

    # Calibration.
    sigmoid_primary, sigmoid_model = calibrate_sigmoid(source.primary_oof, y_true, source.primary_mask)
    sigmoid_group, _ = calibrate_sigmoid(source.group_oof, y_true, np.ones(len(y_true), dtype=bool))
    sigmoid_test = apply_calibrator(sigmoid_model, source.test_pred)
    candidates.append(
        make_candidate(
            name=f"{source.name}::sigmoid",
            source_name=source.name,
            feature_set=source.feature_set,
            primary_pred=sigmoid_primary,
            primary_mask=source.primary_mask,
            group_pred=sigmoid_group,
            y_true=y_true,
            test_pred=sigmoid_test,
            metadata={"postprocess": "sigmoid"},
        )
    )

    isotonic_primary, isotonic_model = calibrate_isotonic(source.primary_oof, y_true, source.primary_mask)
    if isotonic_model is not None:
        isotonic_group, _ = calibrate_isotonic(source.group_oof, y_true, np.ones(len(y_true), dtype=bool))
        isotonic_test = apply_calibrator(isotonic_model, source.test_pred)
        candidates.append(
            make_candidate(
                name=f"{source.name}::isotonic",
                source_name=source.name,
                feature_set=source.feature_set,
                primary_pred=isotonic_primary,
                primary_mask=source.primary_mask,
                group_pred=isotonic_group,
                y_true=y_true,
                test_pred=isotonic_test,
                metadata={"postprocess": "isotonic"},
            )
        )

    base_variants = {
        "raw": (source.primary_oof, source.group_oof, source.test_pred),
        "sigmoid": (sigmoid_primary, sigmoid_group, sigmoid_test),
    }
    if isotonic_model is not None:
        base_variants["isotonic"] = (isotonic_primary, isotonic_group, isotonic_test)

    # Prior-only candidates.
    for prior_name, prior_source in prior_sources.items():
        candidates.append(
            make_candidate(
                name=f"prior_only::{prior_name}",
                source_name=prior_name,
                feature_set=prior_source.feature_set,
                primary_pred=prior_source.primary_oof,
                primary_mask=prior_source.primary_mask,
                group_pred=prior_source.group_oof,
                y_true=y_true,
                test_pred=prior_source.test_pred,
                metadata={"postprocess": "prior_only", "prior_variant": prior_name},
            )
        )

    # Best blend per prior and per calibrated/raw base.
    for variant_name, (primary_base, group_base, test_base) in base_variants.items():
        for prior_name, prior_source in prior_sources.items():
            grid_candidates = []
            for alpha in np.linspace(0.0, 1.0, BLEND_GRID_STEPS):
                blended_primary = clip_proba(alpha * primary_base + (1.0 - alpha) * prior_source.primary_oof)
                grid_metrics = probability_metrics(y_true, blended_primary, mask=source.primary_mask)
                grid_candidates.append((alpha, grid_metrics))
            best_grid_alpha, _ = min(grid_candidates, key=lambda item: metric_tuple(item[1]))
            best_alpha = refine_alpha_with_optuna(
                y_true,
                primary_base,
                prior_source.primary_oof,
                source.primary_mask,
                best_grid_alpha,
            )
            candidates.append(
                make_candidate(
                    name=f"{source.name}::{variant_name}__blend__{prior_name}",
                    source_name=source.name,
                    feature_set=source.feature_set,
                    primary_pred=best_alpha * primary_base + (1.0 - best_alpha) * prior_source.primary_oof,
                    primary_mask=source.primary_mask,
                    group_pred=best_alpha * group_base + (1.0 - best_alpha) * prior_source.group_oof,
                    y_true=y_true,
                    test_pred=best_alpha * test_base + (1.0 - best_alpha) * prior_source.test_pred,
                    metadata={
                        "postprocess": "prior_blend",
                        "base_variant": variant_name,
                        "prior_variant": prior_name,
                        "alpha": float(best_alpha),
                    },
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Optional model availability
# ---------------------------------------------------------------------------

def optional_model_available(model_name: str) -> tuple[bool, str | None]:
    if model_name == "tabpfn":
        if not ENABLE_TABPFN:
            return False, "ENABLE_TABPFN=0"
        try:
            import tabpfn  # noqa: F401
            return True, None
        except Exception as exc:
            return False, str(exc)
    if model_name == "autogluon":
        if not ENABLE_AUTOGLUON:
            return False, "ENABLE_AUTOGLUON=0"
        try:
            import autogluon.tabular  # noqa: F401
            return True, None
        except Exception as exc:
            return False, str(exc)
    if model_name == "torch_mlp":
        if not ENABLE_PYTORCH_MLP:
            return False, "ENABLE_PYTORCH_MLP=0"
        return False, "torch_mlp not implemented in this engine"
    return False, "unknown optional model"


# ---------------------------------------------------------------------------
# Feature-set screening and target loop
# ---------------------------------------------------------------------------

def screen_feature_sets(feature_sets: dict[str, FeatureSet], train_df: pd.DataFrame, target: str, primary_folds):
    y = train_df[target].values.astype(int)
    scores = []
    for name, feat in feature_sets.items():
        if name == "compact_all":
            # Always keep the compact baseline set for diversity.
            continue
        score = optimize_single_metric("lgb", DEFAULT_MODEL_PARAMS["lgb"], feat.train, y, primary_folds)
        scores.append({"feature_set": name, "metric": float(score), "feature_count": int(feat.train.shape[1])})
    scores = sorted(scores, key=lambda item: item["metric"])
    keep = [item["feature_set"] for item in scores[: max(1, SCREEN_KEEP)]]
    if "compact_all" in feature_sets:
        keep.append("compact_all")
    return keep, scores


def build_optional_sources(feature_sets: dict[str, FeatureSet], target: str, y: np.ndarray, primary_folds, group_folds, skip_report: dict) -> list[SourceResult]:
    out: list[SourceResult] = []

    tabpfn_ok, tabpfn_reason = optional_model_available("tabpfn")
    if tabpfn_ok:
        tabpfn_feature = None
        for name in ("expanded_top200", "expanded_top120", "compact_all", "expanded_top80"):
            feat = feature_sets.get(name)
            if feat is not None and feat.train.shape[1] <= MAX_TABPFN_FEATURES:
                tabpfn_feature = feat
                break
        if tabpfn_feature is not None:
            out.append(cross_validated_source("tabpfn", tabpfn_feature, y, primary_folds, group_folds, None, target))
        else:
            skip_report["tabpfn"] = f"No feature set <= {MAX_TABPFN_FEATURES} features"
    else:
        skip_report["tabpfn"] = tabpfn_reason

    autogluon_ok, autogluon_reason = optional_model_available("autogluon")
    if autogluon_ok:
        feat = feature_sets.get("compact_all") or next(iter(feature_sets.values()))
        out.append(cross_validated_source("autogluon", feat, y, primary_folds, group_folds, None, target))
    else:
        skip_report["autogluon"] = autogluon_reason

    torch_ok, torch_reason = optional_model_available("torch_mlp")
    if not torch_ok:
        skip_report["torch_mlp"] = torch_reason

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def baseline_mae_from_existing_oof(train_df: pd.DataFrame) -> dict | None:
    path = OUTPUT_DIR / "oof_predictions.csv"
    if not path.exists():
        return None
    oof = pd.read_csv(path)
    metrics = {}
    maes = []
    for target in TARGETS:
        y = train_df[target].values.astype(int)
        p = clip_proba(oof[target].values)
        m = probability_metrics(y, p)
        metrics[target] = m
        maes.append(m["mae"])
    return {
        "path": str(path),
        "average_oof_mae": float(np.mean(maes)),
        "targets": metrics,
    }


def main():
    print("=" * 60)
    print("MAE-First Advanced Ensemble")
    print("=" * 60)

    metrics_train = pd.read_csv(input_path("ch2026_metrics_train.csv"))
    submission = pd.read_csv(input_path("ch2026_submission_sample.csv"))
    feature_sets, feature_info = build_feature_sets(metrics_train, submission)

    baseline_metrics = baseline_mae_from_existing_oof(metrics_train)
    primary_folds = make_time_blocked_folds(metrics_train)

    summary_rows = []
    chosen_oof = metrics_train[["subject_id", "sleep_date", "lifelog_date"]].copy()
    chosen_submission = submission[["subject_id", "sleep_date", "lifelog_date"]].copy()
    mae_hunt_report = {
        "primary_metric": PRIMARY_METRIC,
        "tie_breakers": list(PRIMARY_TIEBREAKERS),
        "settings": {
            "n_folds": N_FOLDS,
            "time_folds": TIME_FOLDS,
            "time_warmup_frac": TIME_WARMUP_FRAC,
            "time_min_train": TIME_MIN_TRAIN,
            "group_guardrail_pct": GROUP_GUARDRAIL_PCT,
            "n_optuna": N_OPTUNA,
            "tune_folds": TUNE_FOLDS,
            "n_seeds": N_SEEDS,
            "fast_trees": FAST_TREES,
            "feature_subsets": FEATURE_SUBSETS_RAW,
            "screen_keep": SCREEN_KEEP,
            "enable_pseudo": ENABLE_PSEUDO,
            "enable_tabpfn": ENABLE_TABPFN,
            "enable_autogluon": ENABLE_AUTOGLUON,
            "enable_pytorch_mlp": ENABLE_PYTORCH_MLP,
        },
        "baseline_reproduction": baseline_metrics,
        "feature_info": feature_info,
        "targets": {},
    }

    optuna_best_params: dict[str, dict] = {}

    for target in TARGETS:
        print(f"\n{'─' * 60}")
        print(f"TARGET: {target}")
        y = metrics_train[target].values.astype(int)
        group_folds = make_group_folds(metrics_train, y)

        keep_feature_sets, screening = screen_feature_sets(feature_sets, metrics_train, target, primary_folds)
        print("  Feature screening:")
        for item in screening[:5]:
            print(f"    {item['feature_set']}: mae={item['metric']:.5f}  features={item['feature_count']}")
        print(f"  Keeping: {', '.join(keep_feature_sets)}")

        selected_feature_sets = {name: feature_sets[name] for name in keep_feature_sets}
        target_optuna = {}
        source_results: list[SourceResult] = []

        # Prior-only sources first.
        prior_sources: dict[str, SourceResult] = {}
        for prior_name in PRIOR_VARIANTS:
            primary_pred, primary_mask, test_pred = compute_prior_predictions(metrics_train, submission, target, primary_folds, prior_name)
            group_pred, _, _ = compute_prior_predictions(metrics_train, submission, target, group_folds, prior_name)
            prior_sources[prior_name] = SourceResult(
                name=f"prior::{prior_name}",
                feature_set="prior",
                primary_oof=primary_pred,
                primary_mask=primary_mask,
                group_oof=group_pred,
                test_pred=test_pred,
                metadata={"prior_variant": prior_name},
            )

        # Stage 2: tune models on selected feature sets.
        for feature_name, feature_set in selected_feature_sets.items():
            print(f"  Tuning / evaluating feature set: {feature_name} ({feature_set.train.shape[1]} features)")
            model_names = ["lgb", "xgb", "cat", "rf", "et"]
            if feature_name == "compact_all":
                model_names.append("logistic")

            for model_name in model_names:
                if model_name in {"lgb", "xgb", "cat"}:
                    best_params, best_value = tune_model(model_name, target, feature_name, feature_set.train, y, primary_folds)
                else:
                    best_params, best_value = ({}, float("nan"))
                target_optuna.setdefault(feature_name, {})[model_name] = {
                    "best_params": safe_json(best_params),
                    "best_primary_metric": best_value,
                }
                result = cross_validated_source(model_name, feature_set, y, primary_folds, group_folds, best_params, target)
                source_results.append(result)
                primary_metrics = probability_metrics(y, result.primary_oof, result.primary_mask)
                print(
                    f"    {result.name}: primary_mae={primary_metrics['mae']:.5f}  "
                    f"group_mae={probability_metrics(y, result.group_oof)['mae']:.5f}"
                )

            if feature_name == "compact_all":
                compact_map = {r.metadata["model_name"]: r for r in source_results if r.feature_set == feature_name and r.metadata["model_name"] in {"lgb", "et", "logistic"}}
                if set(compact_map) >= {"lgb", "et", "logistic"}:
                    for blend_name, weights in candidate_model_blends().items():
                        blend_result = blend_source_results(
                            name=f"compact_all::{blend_name}",
                            results=[compact_map["lgb"], compact_map["et"], compact_map["logistic"]],
                            weights=list(weights),
                        )
                        source_results.append(blend_result)

        # Add equal-weight ensemble sources.
        for feature_name in selected_feature_sets:
            bundle = [r for r in source_results if r.feature_set == feature_name and r.metadata.get("model_name") in {"lgb", "xgb", "cat", "rf", "et"}]
            if len(bundle) >= 3:
                source_results.append(
                    blend_source_results(
                        name=f"{feature_name}::advanced_tree_mean",
                        results=bundle,
                        weights=[1.0 / len(bundle)] * len(bundle),
                    )
                )
                boost_only = [r for r in bundle if r.metadata.get("model_name") in {"lgb", "xgb", "cat"}]
                if len(boost_only) == 3:
                    source_results.append(
                        blend_source_results(
                            name=f"{feature_name}::boosted_mean",
                            results=boost_only,
                            weights=[1 / 3, 1 / 3, 1 / 3],
                        )
                    )

        # Optional sources.
        skip_report: dict[str, str | None] = {}
        source_results.extend(build_optional_sources(selected_feature_sets, target, y, primary_folds, group_folds, skip_report))

        # Post-processing and selection.
        candidate_results: list[CandidateResult] = []
        for prior_name, prior_source in prior_sources.items():
            candidate_results.append(
                make_candidate(
                    name=f"prior_only::{prior_name}",
                    source_name=prior_source.name,
                    feature_set="prior",
                    primary_pred=prior_source.primary_oof,
                    primary_mask=prior_source.primary_mask,
                    group_pred=prior_source.group_oof,
                    y_true=y,
                    test_pred=prior_source.test_pred,
                    metadata={"kind": "prior_only", "prior_variant": prior_name},
                )
            )
        for source in source_results:
            candidate_results.extend(postprocess_source(source, y, prior_sources))

        best_group_mae = min(c.group_metrics["mae"] for c in candidate_results)
        guardrail_limit = best_group_mae * (1.0 + GROUP_GUARDRAIL_PCT)
        guardrail_candidates = [c for c in candidate_results if c.group_metrics["mae"] <= guardrail_limit]
        chosen = min(guardrail_candidates or candidate_results, key=lambda item: metric_tuple(item.primary_metrics))

        chosen_blended_oof = combine_primary_with_group(chosen.primary_pred, chosen.primary_mask, chosen.group_pred)
        chosen_oof[target] = chosen_blended_oof
        chosen_submission[target] = clip_proba(chosen.test_pred)

        summary_rows.append(
            {
                "target": target,
                "chosen_source": chosen.source_name,
                "chosen_candidate": chosen.name,
                "chosen_feature_set": chosen.feature_set,
                "primary_mae": float(chosen.primary_metrics["mae"]),
                "primary_brier": float(chosen.primary_metrics["brier"]),
                "primary_logloss": float(chosen.primary_metrics["logloss"]),
                "group_mae": float(chosen.group_metrics["mae"]),
                "group_brier": float(chosen.group_metrics["brier"]),
                "group_logloss": float(chosen.group_metrics["logloss"]),
                "coverage": float(chosen.primary_metrics["coverage"]),
            }
        )

        target_details = {
            "feature_screening": screening,
            "kept_feature_sets": keep_feature_sets,
            "optuna": target_optuna,
            "optional_skips": skip_report,
            "guardrail_group_mae_limit": float(guardrail_limit),
            "chosen_source": chosen.source_name,
            "chosen_candidate": chosen.name,
            "chosen_feature_set": chosen.feature_set,
            "chosen_metrics": {
                "primary": chosen.primary_metrics,
                "group": chosen.group_metrics,
            },
            "chosen_metadata": safe_json(chosen.metadata),
            "candidate_rankings": [
                {
                    "name": c.name,
                    "source_name": c.source_name,
                    "feature_set": c.feature_set,
                    "primary_metrics": c.primary_metrics,
                    "group_metrics": c.group_metrics,
                    "metadata": safe_json(c.metadata),
                }
                for c in sorted(candidate_results, key=lambda item: metric_tuple(item.primary_metrics))[:30]
            ],
        }
        mae_hunt_report["targets"][target] = target_details
        optuna_best_params[target] = target_optuna

        print(
            f"  Chosen -> {chosen.name}: primary_mae={chosen.primary_metrics['mae']:.5f}  "
            f"group_mae={chosen.group_metrics['mae']:.5f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    avg_primary_mae = float(summary_df["primary_mae"].mean())
    avg_group_mae = float(summary_df["group_mae"].mean())

    mae_hunt_report["summary"] = {
        "average_primary_oof_mae": avg_primary_mae,
        "average_group_oof_mae": avg_group_mae,
        "targets": summary_rows,
    }

    model_comparison_report = {
        "primary_metric": "average_oof_mae",
        "secondary_metrics": ["brier", "logloss", "macro_f1"],
        "average_oof_mae": avg_primary_mae,
        "average_group_oof_mae": avg_group_mae,
        "targets": summary_rows,
    }

    mae_report_path = OUTPUT_DIR / "mae_hunt_report.json"
    comparison_path = OUTPUT_DIR / "model_comparison_report.json"
    oof_path = OUTPUT_DIR / "oof_predictions_mae.csv"
    params_path = OUTPUT_DIR / "optuna_best_params.json"
    submission_path = next_submission_path(OUTPUT_DIR)

    with open(mae_report_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(mae_hunt_report), f, indent=2, ensure_ascii=False)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(model_comparison_report), f, indent=2, ensure_ascii=False)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(optuna_best_params), f, indent=2, ensure_ascii=False)
    chosen_oof.to_csv(oof_path, index=False)
    chosen_submission.to_csv(submission_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Primary average MAE: {avg_primary_mae:.5f}")
    print(f"Guardrail average Group MAE: {avg_group_mae:.5f}")
    print(f"MAE hunt report:       {mae_report_path}")
    print(f"Model comparison:      {comparison_path}")
    print(f"Optuna best params:    {params_path}")
    print(f"OOF predictions:       {oof_path}")
    print(f"Submission:            {submission_path}")


if __name__ == "__main__":
    main()
