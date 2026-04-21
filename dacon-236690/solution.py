"""
ETRI Lifelog 2026 Challenge - Full Ensemble Solution
====================================================
Target: Q1, Q2, Q3 (subjective sleep quality/fatigue/stress)
        S1, S2, S3, S4 (objective sleep metrics: TST, SE, SOL, WASO)

Strategy:
  1. Feature engineering per (subject_id, lifelog_date) across all sensors
  2. Multiple time windows: morning / afternoon / evening / night / pre-sleep
  3. Per-subject normalization (relative features)
  4. LightGBM + XGBoost + CatBoost + RF + ExtraTrees + stacking meta-learner
  5. Subject-aware GroupKFold CV
"""

import os, warnings, gc, sys
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
ROOT_DIR   = Path(".")
DATA_DIR   = ROOT_DIR / "ch2025_data_items"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output hygiene for future agents:
# - Keep generated CSV/JSON/Parquet/log artifacts under OUTPUT_DIR.
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

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEED    = 42
N_FOLDS = 5          # GroupKFold per subject
N_JOBS  = -1
FAST_TREES = int(os.getenv("FAST_TREES", "250"))
CLASS_WEIGHT_MODE = os.getenv("CLASS_WEIGHT_MODE", "none").strip().lower()

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore")


def class_weight_value():
    return "balanced" if CLASS_WEIGHT_MODE == "balanced" else None


def input_path(filename: str) -> Path:
    """Find challenge CSVs whether they are in repo root or data folder."""
    for base in (ROOT_DIR, DATA_DIR):
        path = base / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in {ROOT_DIR} or {DATA_DIR}")


def next_submission_path(output_dir: Path) -> Path:
    """Return outputs/submission_vN.csv with the next available version number."""
    versions = []
    for path in output_dir.glob("submission_v*.csv"):
        suffix = path.stem.removeprefix("submission_v")
        if suffix.isdigit():
            versions.append(int(suffix))
    next_version = max(versions, default=0) + 1
    return output_dir / f"submission_v{next_version}.csv"


def load_cached_features(output_dir: Path) -> pd.DataFrame | None:
    """Load feature cache, preferring CSV so pyarrow is not needed for reruns."""
    csv_path = output_dir / "solution_features.csv"
    parquet_path = output_dir / "solution_features.parquet"
    if csv_path.exists():
        print(f"Loading cached features: {csv_path}")
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        print(f"Loading cached features: {parquet_path}")
        try:
            return pd.read_parquet(parquet_path)
        except ImportError as exc:
            raise ImportError(
                "Parquet cache exists but this Python cannot read parquet. "
                "Run `python -m pip install -r requirements.txt`, or use "
                "`./.venv/bin/python -u solution.py`."
            ) from exc
    return None


def save_feature_cache(feat_df: pd.DataFrame, output_dir: Path) -> None:
    """Save CSV plus best-effort parquet cache."""
    csv_path = output_dir / "solution_features.csv"
    parquet_path = output_dir / "solution_features.parquet"
    feat_df.to_csv(csv_path, index=False)
    try:
        feat_df.to_parquet(parquet_path, index=False)
        print(f"Cached features saved to: {parquet_path} and {csv_path}")
    except ImportError:
        print(f"Cached features saved to: {csv_path}")

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def safe_mean(x):   return np.nanmean(x) if len(x) else np.nan
def safe_std(x):    return np.nanstd(x)  if len(x) else np.nan
def safe_max(x):    return np.nanmax(x)  if len(x) else np.nan
def safe_min(x):    return np.nanmin(x)  if len(x) else np.nan
def safe_sum(x):    return np.nansum(x)  if len(x) else 0.0
def safe_median(x): return np.nanmedian(x) if len(x) else np.nan

def hour_mask(ts_series, h_start, h_end):
    """Boolean mask for timestamps in [h_start, h_end) hours."""
    h = ts_series.dt.hour
    if h_start < h_end:
        return (h >= h_start) & (h < h_end)
    else:  # wraps midnight
        return (h >= h_start) | (h < h_end)

def flatten_hr_array(arr):
    """Flatten numpy array of HR readings from wHr."""
    if arr is None:
        return np.array([])
    try:
        flat = np.array(arr, dtype=float).flatten()
        return flat[~np.isnan(flat)]
    except Exception:
        return np.array([])

def read_parquet_cached(path, cache={}):
    """Read parquet with simple in-process cache."""
    key = str(path)
    if key not in cache:
        cache[key] = pd.read_parquet(path)
    return cache[key]


# ──────────────────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTIONS (per sensor, per day)
# ──────────────────────────────────────────────────────────

def extract_wPedo_features(df_subj, lifelog_date):
    """
    Wearable pedometer features for a given subject on lifelog_date.
    Windows: full-day / morning (6-12) / afternoon (12-18) / evening (18-24)
    """
    d = pd.Timestamp(lifelog_date).date()
    day_df = df_subj[df_subj["timestamp"].dt.date == d]
    feat = {}

    for win_name, (hs, he) in [
        ("day",  (0, 24)), ("morn", (6, 12)),
        ("aft",  (12,18)), ("eve",  (18, 24))
    ]:
        w = day_df[hour_mask(day_df["timestamp"], hs, he)] if win_name != "day" else day_df

        feat[f"pedo_{win_name}_steps"]      = safe_sum(w["step"])
        feat[f"pedo_{win_name}_run_steps"]  = safe_sum(w["running_step"])
        feat[f"pedo_{win_name}_walk_steps"] = safe_sum(w["walking_step"])
        feat[f"pedo_{win_name}_dist"]       = safe_sum(w["distance"])
        feat[f"pedo_{win_name}_kcal"]       = safe_sum(w["burned_calories"])
        feat[f"pedo_{win_name}_speed_mean"] = safe_mean(w["speed"])
        feat[f"pedo_{win_name}_speed_max"]  = safe_max(w["speed"])

    # Active bouts (non-zero steps)
    active = day_df[day_df["step"] > 0]
    feat["pedo_active_minutes"] = len(active)
    feat["pedo_run_ratio"] = (
        safe_sum(day_df["running_step"]) / (safe_sum(day_df["step"]) + 1e-6)
    )
    return feat


def extract_wLight_features(df_subj, lifelog_date, sleep_date):
    """
    Wearable (Withings) light during sleep window:
    22:00 lifelog_date → 10:00 sleep_date
    """
    ld = pd.Timestamp(lifelog_date)
    sd = pd.Timestamp(sleep_date)
    sleep_start = ld.replace(hour=22, minute=0, second=0)
    sleep_end   = sd.replace(hour=10, minute=0, second=0)

    sleep_df = df_subj[
        (df_subj["timestamp"] >= sleep_start) &
        (df_subj["timestamp"] <= sleep_end)
    ]
    day_df = df_subj[df_subj["timestamp"].dt.date == ld.date()]

    feat = {}
    feat["wlight_sleep_mean"]    = safe_mean(sleep_df["w_light"])
    feat["wlight_sleep_std"]     = safe_std(sleep_df["w_light"])
    feat["wlight_sleep_max"]     = safe_max(sleep_df["w_light"])
    feat["wlight_sleep_dark_pct"]= (
        (sleep_df["w_light"] < 10).sum() / (len(sleep_df) + 1e-6)
    )
    # Daytime light exposure
    feat["wlight_day_mean"]      = safe_mean(day_df["w_light"])
    feat["wlight_day_max"]       = safe_max(day_df["w_light"])
    return feat


def extract_wHr_features(df_subj, lifelog_date, sleep_date):
    """
    Heart rate features during sleep window (22:00 → 10:00).
    Also resting HR proxy (min over sleep period).
    """
    ld = pd.Timestamp(lifelog_date)
    sd = pd.Timestamp(sleep_date)
    sleep_start = ld.replace(hour=22, minute=0, second=0)
    sleep_end   = sd.replace(hour=10, minute=0, second=0)

    sleep_df = df_subj[
        (df_subj["timestamp"] >= sleep_start) &
        (df_subj["timestamp"] <= sleep_end)
    ]

    all_hr = np.concatenate(
        sleep_df["heart_rate"].apply(flatten_hr_array).tolist()
    ) if len(sleep_df) else np.array([])

    feat = {}
    feat["hr_sleep_mean"]   = safe_mean(all_hr)
    feat["hr_sleep_std"]    = safe_std(all_hr)
    feat["hr_sleep_min"]    = safe_min(all_hr)
    feat["hr_sleep_max"]    = safe_max(all_hr)
    feat["hr_sleep_median"] = safe_median(all_hr)
    feat["hr_sleep_n"]      = len(all_hr)

    # HRV proxy: RMSSD-like (successive diffs)
    if len(all_hr) > 1:
        diffs = np.diff(all_hr)
        feat["hr_rmssd_proxy"] = np.sqrt(safe_mean(diffs**2))
        feat["hr_range"]       = safe_max(all_hr) - safe_min(all_hr)
    else:
        feat["hr_rmssd_proxy"] = np.nan
        feat["hr_range"]       = np.nan

    # Daytime HR (activity proxy)
    day_df = df_subj[df_subj["timestamp"].dt.date == ld.date()]
    day_hr = np.concatenate(
        day_df["heart_rate"].apply(flatten_hr_array).tolist()
    ) if len(day_df) else np.array([])
    feat["hr_day_mean"] = safe_mean(day_hr)
    feat["hr_day_max"]  = safe_max(day_hr)
    return feat


def extract_mActivity_features(df_subj, lifelog_date):
    """
    Activity codes: 0=vehicle, 1=bicycle, 3=foot/walk, 4=still, 7=tilting, 8=running
    """
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]
    total = len(df) + 1e-6

    feat = {}
    for win_name, (hs, he) in [
        ("day", (0, 24)), ("morn", (6, 12)),
        ("aft", (12,18)), ("eve",  (18, 24))
    ]:
        w = df[hour_mask(df["timestamp"], hs, he)] if win_name != "day" else df
        n = len(w) + 1e-6
        feat[f"act_{win_name}_still_pct"]  = (w["m_activity"] == 4).sum() / n
        feat[f"act_{win_name}_walk_pct"]   = (w["m_activity"] == 3).sum() / n
        feat[f"act_{win_name}_vehicle_pct"]= (w["m_activity"] == 0).sum() / n
        feat[f"act_{win_name}_run_pct"]    = (w["m_activity"] == 8).sum() / n

    # Evening sedentary (18-24): proxy for pre-sleep restfulness
    eve = df[hour_mask(df["timestamp"], 18, 24)]
    feat["act_eve_sedentary_ratio"] = (eve["m_activity"].isin([4, 7])).sum() / (len(eve) + 1e-6)
    return feat


def extract_mScreen_features(df_subj, lifelog_date):
    """
    Screen usage: on-time ratio, last screen use hour (pre-sleep hygiene).
    """
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]
    total = len(df) + 1e-6

    feat = {}
    feat["screen_on_pct"]   = (df["m_screen_use"] == 1).sum() / total
    feat["screen_on_count"] = (df["m_screen_use"] == 1).sum()

    # Evening screen use (18:00–24:00)
    eve = df[hour_mask(df["timestamp"], 18, 24)]
    feat["screen_eve_on_pct"] = (eve["m_screen_use"] == 1).sum() / (len(eve) + 1e-6)

    # Late-night screen (21:00–24:00)
    late = df[hour_mask(df["timestamp"], 21, 24)]
    feat["screen_late_on_pct"]  = (late["m_screen_use"] == 1).sum() / (len(late) + 1e-6)
    feat["screen_late_on_count"]= (late["m_screen_use"] == 1).sum()

    # Last screen-on timestamp hour
    on_rows = df[df["m_screen_use"] == 1]
    feat["screen_last_on_hour"] = on_rows["timestamp"].dt.hour.max() if len(on_rows) else np.nan
    feat["screen_first_on_hour"]= on_rows["timestamp"].dt.hour.min() if len(on_rows) else np.nan
    return feat


def extract_mLight_features(df_subj, lifelog_date):
    """Phone light sensor — ambient light proxy."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    feat = {}
    feat["mlight_day_mean"]  = safe_mean(df["m_light"])
    feat["mlight_day_std"]   = safe_std(df["m_light"])
    feat["mlight_day_max"]   = safe_max(df["m_light"])
    feat["mlight_dark_pct"]  = (df["m_light"] < 5).sum() / (len(df) + 1e-6)

    eve = df[hour_mask(df["timestamp"], 21, 24)]
    feat["mlight_eve_mean"]  = safe_mean(eve["m_light"])
    return feat


def extract_mACStatus_features(df_subj, lifelog_date):
    """Charging status — proxy for time spent at desk / sleeping."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    feat = {}
    feat["charge_pct"]       = (df["m_charging"] == 1).sum() / (len(df) + 1e-6)
    night = df[hour_mask(df["timestamp"], 22, 24)]
    feat["charge_night_pct"] = (night["m_charging"] == 1).sum() / (len(night) + 1e-6)
    return feat


def extract_mUsageStats_features(df_subj, lifelog_date):
    """App usage statistics: total screen time, categories."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    total_time = 0.0
    social_time = 0.0
    social_keywords = ["카톡", "카카오", "인스타", "페이스북", "트위터", "유튜브",
                       "틱톡", "YouTube", "Instagram", "Twitter", "Facebook",
                       "KakaoTalk", "Naver", "네이버"]
    n_apps = 0
    for _, row in df.iterrows():
        stats = row["m_usage_stats"]
        if stats is None: continue
        for item in stats:
            try:
                t = float(item["total_time"])
                total_time += t
                n_apps += 1
                if any(k in str(item.get("app_name", "")) for k in social_keywords):
                    social_time += t
            except Exception:
                continue

    feat = {}
    feat["usage_total_sec"]   = total_time
    feat["usage_social_sec"]  = social_time
    feat["usage_social_ratio"]= social_time / (total_time + 1e-6)
    feat["usage_n_apps"]      = n_apps
    return feat


def extract_mAmbience_features(df_subj, lifelog_date):
    """Sound environment classification probabilities."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    keys = ["Music", "Speech", "Silence", "Vehicle", "Outside, urban or manmade",
            "Inside, small room"]
    accum = {k: [] for k in keys}

    for _, row in df.iterrows():
        amb = row["m_ambience"]
        if amb is None: continue
        for item in amb:
            try:
                label = str(item[0])
                prob  = float(item[1])
                for k in keys:
                    if k in label:
                        accum[k].append(prob)
            except Exception:
                continue

    feat = {}
    for k in keys:
        safe_k = k.replace(" ", "_").replace(",", "")
        feat[f"amb_{safe_k}_mean"] = safe_mean(accum[k])
    return feat


def extract_mWifi_features(df_subj, lifelog_date):
    """WiFi scanning - proxy for mobility (unique SSIDs seen)."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    unique_bssids = set()
    rssi_vals = []
    for _, row in df.iterrows():
        wifi = row["m_wifi"]
        if wifi is None: continue
        for ap in wifi:
            try:
                unique_bssids.add(ap.get("bssid", ""))
                rssi_vals.append(float(ap.get("rssi", np.nan)))
            except Exception:
                continue

    feat = {}
    feat["wifi_unique_ssids"] = len(unique_bssids)
    feat["wifi_rssi_mean"]    = safe_mean(rssi_vals)
    feat["wifi_rssi_max"]     = safe_max(rssi_vals)
    return feat


def extract_mGps_features(df_subj, lifelog_date):
    """GPS mobility: variance in lat/lon as proxy for physical movement."""
    d  = pd.Timestamp(lifelog_date).date()
    df = df_subj[df_subj["timestamp"].dt.date == d]

    lats, lons, alts = [], [], []
    for _, row in df.iterrows():
        gps = row["m_gps"]
        if gps is None: continue
        for pt in gps:
            try:
                lats.append(float(pt.get("latitude", np.nan)))
                lons.append(float(pt.get("longitude", np.nan)))
                alts.append(float(pt.get("altitude", np.nan)))
            except Exception:
                continue

    feat = {}
    feat["gps_lat_std"]   = safe_std(lats)
    feat["gps_lon_std"]   = safe_std(lons)
    feat["gps_alt_mean"]  = safe_mean(alts)
    feat["gps_n_points"]  = len(lats)
    # Mobility score: combined lat/lon variance
    feat["gps_mobility"]  = safe_std(lats) + safe_std(lons)
    return feat


# ──────────────────────────────────────────────────────────
# BUILD FEATURE MATRIX
# ──────────────────────────────────────────────────────────

def build_features(metrics_df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Build full feature matrix for all rows in metrics_df.
    metrics_df must have: subject_id, lifelog_date, sleep_date
    """
    if verbose:
        print("Loading sensor data...")

    # Load all sensor data upfront (by reference)
    wPedo       = pd.read_parquet(DATA_DIR / "ch2025_wPedo.parquet")
    wLight      = pd.read_parquet(DATA_DIR / "ch2025_wLight.parquet")
    wHr         = pd.read_parquet(DATA_DIR / "ch2025_wHr.parquet")
    mActivity   = pd.read_parquet(DATA_DIR / "ch2025_mActivity.parquet")
    mScreenStatus = pd.read_parquet(DATA_DIR / "ch2025_mScreenStatus.parquet")
    mLight      = pd.read_parquet(DATA_DIR / "ch2025_mLight.parquet")
    mACStatus   = pd.read_parquet(DATA_DIR / "ch2025_mACStatus.parquet")
    mUsageStats = pd.read_parquet(DATA_DIR / "ch2025_mUsageStats.parquet")
    mAmbience   = pd.read_parquet(DATA_DIR / "ch2025_mAmbience.parquet")
    mWifi       = pd.read_parquet(DATA_DIR / "ch2025_mWifi.parquet")
    mGps        = pd.read_parquet(DATA_DIR / "ch2025_mGps.parquet")

    if verbose:
        print("Sensor data loaded. Building features per subject...")

    all_feats = []
    subjects = metrics_df["subject_id"].unique()

    for sid in subjects:
        if verbose:
            print(f"  Processing subject: {sid}")

        rows = metrics_df[metrics_df["subject_id"] == sid]

        # Subset each sensor for this subject
        wp   = wPedo[wPedo["subject_id"] == sid].copy()
        wl   = wLight[wLight["subject_id"] == sid].copy()
        wh   = wHr[wHr["subject_id"] == sid].copy()
        ma   = mActivity[mActivity["subject_id"] == sid].copy()
        ms   = mScreenStatus[mScreenStatus["subject_id"] == sid].copy()
        ml   = mLight[mLight["subject_id"] == sid].copy()
        mac  = mACStatus[mACStatus["subject_id"] == sid].copy()
        mu   = mUsageStats[mUsageStats["subject_id"] == sid].copy()
        mamb = mAmbience[mAmbience["subject_id"] == sid].copy()
        mwf  = mWifi[mWifi["subject_id"] == sid].copy()
        mg   = mGps[mGps["subject_id"] == sid].copy()

        for _, row in rows.iterrows():
            lifelog_date = row["lifelog_date"]
            sleep_date   = row["sleep_date"]

            feat = {
                "subject_id":   sid,
                "lifelog_date": lifelog_date,
                "sleep_date":   sleep_date,
            }

            try: feat.update(extract_wPedo_features(wp, lifelog_date))
            except Exception as e: print(f"    wPedo error {e}")

            try: feat.update(extract_wLight_features(wl, lifelog_date, sleep_date))
            except Exception as e: print(f"    wLight error {e}")

            try: feat.update(extract_wHr_features(wh, lifelog_date, sleep_date))
            except Exception as e: print(f"    wHr error {e}")

            try: feat.update(extract_mActivity_features(ma, lifelog_date))
            except Exception as e: print(f"    mActivity error {e}")

            try: feat.update(extract_mScreen_features(ms, lifelog_date))
            except Exception as e: print(f"    mScreen error {e}")

            try: feat.update(extract_mLight_features(ml, lifelog_date))
            except Exception as e: print(f"    mLight error {e}")

            try: feat.update(extract_mACStatus_features(mac, lifelog_date))
            except Exception as e: print(f"    mACStatus error {e}")

            try: feat.update(extract_mUsageStats_features(mu, lifelog_date))
            except Exception as e: print(f"    mUsageStats error {e}")

            try: feat.update(extract_mAmbience_features(mamb, lifelog_date))
            except Exception as e: print(f"    mAmbience error {e}")

            try: feat.update(extract_mWifi_features(mwf, lifelog_date))
            except Exception as e: print(f"    mWifi error {e}")

            try: feat.update(extract_mGps_features(mg, lifelog_date))
            except Exception as e: print(f"    mGps error {e}")

            all_feats.append(feat)

        # Free per-subject copies
        del wp, wl, wh, ma, ms, ml, mac, mu, mamb, mwf, mg
        gc.collect()

    # Free global sensor data
    del wPedo, wLight, wHr, mActivity, mScreenStatus, mLight
    del mACStatus, mUsageStats, mAmbience, mWifi, mGps
    gc.collect()

    feat_df = pd.DataFrame(all_feats)

    if verbose:
        print(f"Feature matrix shape: {feat_df.shape}")

    return feat_df


def add_relative_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-subject relative features (value vs. subject's own mean/std).
    This is crucial since Q1/Q2/Q3 are defined relative to personal averages.
    """
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = ["subject_id", "lifelog_date", "sleep_date"]
    feat_cols = [c for c in numeric_cols if c not in meta_cols]

    rel_df = pd.DataFrame(index=feat_df.index)
    for c in feat_cols:
        mean = feat_df.groupby("subject_id")[c].transform("mean")
        std = feat_df.groupby("subject_id")[c].transform("std")
        rel_df[f"{c}_rel"] = (feat_df[c] - mean) / std.replace(0, np.nan)
    return pd.concat([feat_df, rel_df], axis=1)


def add_lag_features(feat_df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    """Add lag features per subject (yesterday's values)."""
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols    = ["subject_id", "lifelog_date", "sleep_date"]
    feat_cols    = [c for c in numeric_cols if c not in meta_cols
                    and not c.endswith("_rel")]

    lag_dfs = []
    for lag in lags:
        lag_df = (
            feat_df[["subject_id", "lifelog_date"] + feat_cols]
            .copy()
        )
        lag_df["lifelog_date_shifted"] = (
            pd.to_datetime(lag_df["lifelog_date"]) + pd.Timedelta(days=lag)
        ).dt.strftime("%Y-%m-%d")
        lag_df = lag_df.rename(columns={c: f"{c}_lag{lag}" for c in feat_cols})
        lag_df = lag_df.rename(columns={"lifelog_date": "lifelog_date_orig",
                                        "lifelog_date_shifted": "lifelog_date"})
        lag_dfs.append(lag_df[["subject_id", "lifelog_date"] +
                               [f"{c}_lag{lag}" for c in feat_cols]])

    result = feat_df.copy()
    for lag_df in lag_dfs:
        result = result.merge(lag_df, on=["subject_id", "lifelog_date"], how="left")

    return result


# ──────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def make_lgb(target: str, seed: int = SEED) -> lgb.LGBMClassifier:
    """LightGBM with tuned hyperparameters."""
    # Different priors for Q vs S targets
    is_q = target.startswith("Q")
    return lgb.LGBMClassifier(
        n_estimators=FAST_TREES,
        learning_rate=0.02,
        num_leaves=31 if is_q else 63,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight=class_weight_value(),
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
    )


def make_xgb(target: str, seed: int = SEED) -> xgb.XGBClassifier:
    is_q = target.startswith("Q")
    return xgb.XGBClassifier(
        n_estimators=FAST_TREES,
        learning_rate=0.03,
        max_depth=4 if is_q else 6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=0,
    )


def make_cat(target: str, seed: int = SEED) -> cb.CatBoostClassifier:
    is_q = target.startswith("Q")
    return cb.CatBoostClassifier(
        iterations=FAST_TREES,
        learning_rate=0.03,
        depth=4 if is_q else 6,
        l2_leaf_reg=3,
        bagging_temperature=0.5,
        random_strength=1,
        random_seed=seed,
        verbose=0,
    )


def make_rf(seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=FAST_TREES,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight_value(),
        random_state=seed,
        n_jobs=N_JOBS,
    )


def make_et(seed: int = SEED) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=FAST_TREES,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight_value(),
        random_state=seed,
        n_jobs=N_JOBS,
    )


# ──────────────────────────────────────────────────────────
# STACKING ENSEMBLE
# ──────────────────────────────────────────────────────────

def train_stacking_ensemble(X_train, y_train, groups, target: str, seed: int = SEED):
    """
    Two-level stacking:
      Level 1: LGB + XGB + CatBoost + RF + ExtraTrees (OOF predictions)
      Level 2: Logistic Regression meta-learner
    Returns fitted stacking classifier.
    """
    from sklearn.model_selection import StratifiedKFold

    base_estimators = [
        ("lgb",  make_lgb(target, seed)),
        ("xgb",  make_xgb(target, seed)),
        ("cat",  make_cat(target, seed)),
        ("rf",   make_rf(seed)),
        ("et",   make_et(seed)),
    ]

    meta_learner = LogisticRegression(
        C=1.0, class_weight=class_weight_value(), random_state=seed, max_iter=1000
    )

    # StackingClassifier cannot receive groups for its internal CV in older
    # sklearn versions, so use stratified folds internally and keep GroupKFold
    # for the explicit OOF evaluation below.
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,          # keep meta learner on base probabilities only
        n_jobs=1,                   # avoid nested parallelism issues
    )

    # StackingClassifier doesn't support groups in cv directly,
    # so we pass groups via a wrapper approach:
    # For simplicity, fit directly (groups-aware CV is done in eval below)
    print(f"    Fitting stacking ensemble for {target}...")
    stack.fit(X_train, y_train)
    return stack


def train_voting_ensemble(X_train, y_train, target: str, seed: int = SEED):
    """Soft voting ensemble as a fast alternative / complement."""
    estimators = [
        ("lgb", make_lgb(target, seed)),
        ("xgb", make_xgb(target, seed)),
        ("cat", make_cat(target, seed)),
        ("rf",  make_rf(seed)),
        ("et",  make_et(seed)),
    ]
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    print(f"    Fitting voting ensemble for {target}...")
    voting.fit(X_train, y_train)
    return voting


# ──────────────────────────────────────────────────────────
# OOF EVALUATION
# ──────────────────────────────────────────────────────────

def evaluate_oof(X, y, groups, target: str, seed: int = SEED):
    """
    Out-of-fold F1 macro via GroupKFold to measure generalization.
    Returns mean and std of F1 across folds.
    """
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
    f1_scores = []

    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        lgb_m = make_lgb(target, seed + fold_idx)
        xgb_m = make_xgb(target, seed + fold_idx)
        cat_m = make_cat(target, seed + fold_idx)

        lgb_m.fit(X_tr, y_tr)
        xgb_m.fit(X_tr, y_tr)
        cat_m.fit(X_tr, y_tr)

        # Average probabilities
        proba_lgb = lgb_m.predict_proba(X_va)[:, 1]
        proba_xgb = xgb_m.predict_proba(X_va)[:, 1]
        proba_cat = cat_m.predict_proba(X_va)[:, 1]

        avg_proba = (proba_lgb + proba_xgb + proba_cat) / 3
        preds     = (avg_proba >= 0.5).astype(int)

        f1 = f1_score(y_va, preds, average="macro")
        f1_scores.append(f1)
        print(f"      Fold {fold_idx+1}: F1={f1:.4f}")

    return np.mean(f1_scores), np.std(f1_scores)


# ──────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ETRI Lifelog 2026 - Full Ensemble Pipeline")
    print("=" * 60)

    # 1. Load labels
    metrics_train = pd.read_csv(input_path("ch2026_metrics_train.csv"))
    submission    = pd.read_csv(input_path("ch2026_submission_sample.csv"))

    print(f"Train rows: {len(metrics_train)}, Test rows: {len(submission)}")

    # 2. Combine train + test for joint feature extraction
    metrics_train["split"] = "train"
    submission["split"]    = "test"
    all_metrics = pd.concat(
        [metrics_train[["subject_id", "lifelog_date", "sleep_date", "split"]],
         submission[  ["subject_id", "lifelog_date", "sleep_date", "split"]]],
        ignore_index=True
    )

    # 3. Build raw features
    feat_df = load_cached_features(OUTPUT_DIR)
    if feat_df is None:
        feat_df = build_features(all_metrics, verbose=True)
        save_feature_cache(feat_df, OUTPUT_DIR)

    # 4. Add relative + lag features
    print("Adding relative features...")
    feat_df = add_relative_features(feat_df)
    print("Adding lag features...")
    feat_df = add_lag_features(feat_df, lags=[1, 2, 3])

    # 5. Merge targets back
    feat_df = feat_df.merge(
        metrics_train[["subject_id", "lifelog_date"] + TARGETS],
        on=["subject_id", "lifelog_date"],
        how="left"
    )
    feat_df = feat_df.merge(
        all_metrics[["subject_id", "lifelog_date", "split"]],
        on=["subject_id", "lifelog_date"],
        how="left"
    )

    # 6. Split train / test
    train_df = feat_df[feat_df["split"] == "train"].copy()
    test_df  = feat_df[feat_df["split"] == "test"].copy()

    # Feature columns
    drop_cols = ["subject_id", "lifelog_date", "sleep_date", "split"] + TARGETS
    feat_cols = [c for c in feat_df.columns if c not in drop_cols]
    feat_cols = [c for c in feat_cols if feat_df[c].dtype != object]

    print(f"Total features: {len(feat_cols)}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    X_test  = test_df[feat_cols].values.astype(np.float32)
    groups  = train_df["subject_id"].values

    # Fill NaNs
    col_means = np.nanmean(X_train, axis=0)
    nan_mask  = np.isnan(col_means)
    col_means[nan_mask] = 0.0
    X_train = np.where(np.isnan(X_train), col_means, X_train)
    X_test  = np.where(np.isnan(X_test),  col_means, X_test)

    # 7. Train per-target and predict
    predictions = {}
    oof_results = {}

    for target in TARGETS:
        print(f"\n{'─'*50}")
        print(f"Target: {target}")
        y_train = train_df[target].values.astype(int)

        # OOF evaluation
        print(f"  OOF evaluation ({N_FOLDS}-fold GroupKFold):")
        oof_mean, oof_std = evaluate_oof(X_train, y_train, groups, target)
        oof_results[target] = (oof_mean, oof_std)
        print(f"  OOF F1 macro: {oof_mean:.4f} ± {oof_std:.4f}")

        # Train stacking + voting ensembles
        stacking = train_stacking_ensemble(X_train, y_train, groups, target)
        voting   = train_voting_ensemble(X_train, y_train, target)

        # Final prediction: average of stacking + voting probas
        p_stack  = stacking.predict_proba(X_test)[:, 1]
        p_vote   = voting.predict_proba(X_test)[:, 1]
        final_p  = (p_stack * 0.6 + p_vote * 0.4)
        predictions[target] = np.clip(final_p, 1e-5, 1 - 1e-5)

        print(f"  Test probability range: "
              f"{predictions[target].min():.4f}–{predictions[target].max():.4f}")

        del stacking, voting
        gc.collect()

    # 8. Build submission
    result = submission[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for target in TARGETS:
        result[target] = predictions[target]

    out_path = next_submission_path(OUTPUT_DIR)
    result.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Submission saved to: {out_path}")

    # Print OOF summary
    print("\nOOF F1 Summary:")
    for t, (m, s) in oof_results.items():
        print(f"  {t}: {m:.4f} ± {s:.4f}")
    overall_mean = np.mean([v[0] for v in oof_results.values()])
    print(f"  Overall mean: {overall_mean:.4f}")

    return result


if __name__ == "__main__":
    main()
