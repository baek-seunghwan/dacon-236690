#!/usr/bin/env python3
"""auto_tune.py — 반복적 하이퍼파라미터 자동 튜닝 스크립트

사용 예시
---------
# 기본 실행 (무한 루프, Ctrl+C로 종료)
python3 auto_tune.py

# 최대 20회 trial 후 자동 종료
python3 auto_tune.py --max-trials 20

# 특정 전략만 실행
python3 auto_tune.py --strategy random

# fast-dev 모드로 빠르게 탐색 (데이터 서브샘플 사용)
python3 auto_tune.py --fast-dev --max-trials 10

동작 원리
---------
1. 현재 baseline OOF log-loss 측정 (최초 1회)
2. 검색 공간에서 하이퍼파라미터 샘플링
3. solution.run_full_pipeline(param_overrides=...) 호출 → OOF log-loss 계산
4. 개선된 경우 새 best로 갱신 + configs/best_params.json 저장
5. 모든 trial 결과를 tune_history.jsonl에 누적 기록
6. 2~5 반복

결과 파일
---------
- tune_history.jsonl  : 매 trial 결과 (JSON lines)
- configs/best_params.json : 현재까지 최저 OOF 파라미터
- outputs/submissions/ : 각 trial의 submission (개선 시에만 저장)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ── solution.py와 동일 디렉터리에서 실행 가정 ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import solution  # noqa: E402  (solution.py 임포트)

# ── 튜닝 히스토리 / 설정 저장 경로 ────────────────────────────────────────
TUNE_HISTORY_PATH = Path("tune_history.jsonl")
BEST_PARAMS_DIR = Path("configs")
BEST_PARAMS_PATH = BEST_PARAMS_DIR / "best_params.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검색 공간 정의
# 각 항목: (dot_path, 분포타입, 파라미터)
#   "uniform"  → (low, high)          실수 균등분포
#   "loguniform" → (low, high)        로그 스케일 균등분포
#   "int"      → (low, high, step)    정수 균등분포
#   "choice"   → [val1, val2, ...]    이산 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEARCH_SPACE: list[tuple[str, str, Any]] = [
    # ── 모델 앙상블 & 블렌딩 ──────────────────────────────────────────────
    ("training.model_blend_power",           "uniform",    (1.5, 5.0)),
    ("training.weak_model_loss_ratio",       "uniform",    (1.05, 1.35)),

    # ── Prior 가중치 ──────────────────────────────────────────────────────
    ("training.prior.subject_smoothing",     "uniform",    (3.0, 20.0)),
    ("training.prior.recent_smoothing",      "uniform",    (1.0, 10.0)),
    ("training.prior.default_recent_weight", "uniform",    (0.0, 0.7)),

    # ── LightGBM ──────────────────────────────────────────────────────────
    ("training.models.lightgbm.n_estimators","int",        (500, 1500, 100)),
    # LightGBM 세부 파라미터 (build_estimator에서 CONFIG에서 읽도록 확장 가능)
    # 아래는 solution.py의 build_estimator 함수에서 직접 읽는 파라미터가 아니므로
    # 현재는 n_estimators 만 CONFIG 경유로 제어하고, 나머지는 별도 오버라이드 키로 관리

    # ── XGBoost ───────────────────────────────────────────────────────────
    ("training.models.xgboost.n_estimators", "int",        (400, 1400, 100)),

    # ── CatBoost ──────────────────────────────────────────────────────────
    ("training.models.catboost.iterations",  "int",        (500, 1500, 100)),

    # ── ExtraTrees ────────────────────────────────────────────────────────
    ("training.models.extra_trees.n_estimators","int",     (400, 1200, 100)),

    # ── CV splits ─────────────────────────────────────────────────────────
    ("training.n_splits",                    "choice",     [5, 7, 10]),
]

# ── 한 번에 몇 개 파라미터를 동시에 변경할지 (1~3 권장) ────────────────────
N_PARAMS_PER_TRIAL_MIN = 1
N_PARAMS_PER_TRIAL_MAX = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 샘플링 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sample_value(dist_type: str, params: Any) -> Any:
    if dist_type == "uniform":
        lo, hi = params
        return round(random.uniform(lo, hi), 6)
    if dist_type == "loguniform":
        lo, hi = params
        return round(math.exp(random.uniform(math.log(lo), math.log(hi))), 8)
    if dist_type == "int":
        lo, hi, step = params
        choices = list(range(lo, hi + 1, step))
        return random.choice(choices)
    if dist_type == "choice":
        return random.choice(params)
    raise ValueError(f"Unknown dist_type: {dist_type}")


def sample_trial_params(n_params: int | None = None) -> dict[str, Any]:
    """검색 공간에서 n_params개 파라미터를 랜덤하게 선택해 오버라이드 딕트 반환."""
    if n_params is None:
        n_params = random.randint(N_PARAMS_PER_TRIAL_MIN, N_PARAMS_PER_TRIAL_MAX)
    selected = random.sample(SEARCH_SPACE, min(n_params, len(SEARCH_SPACE)))
    return {dotpath: sample_value(dist_type, params) for dotpath, dist_type, params in selected}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 히스토리 & 결과 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def append_history(record: dict[str, Any]) -> None:
    with TUNE_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_history() -> list[dict[str, Any]]:
    if not TUNE_HISTORY_PATH.exists():
        return []
    records = []
    for line in TUNE_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def save_best_params(params: dict[str, Any], logloss: float) -> None:
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(
        json.dumps({"logloss": logloss, "params": params, "saved_at": datetime.now().isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_best_params() -> tuple[dict[str, Any] | None, float]:
    if not BEST_PARAMS_PATH.exists():
        return None, float("inf")
    data = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return data.get("params", {}), float(data.get("logloss", float("inf")))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 trial 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_trial(
    trial_id: int,
    param_overrides: dict[str, Any],
    fast_dev: bool = False,
    force_features: bool = False,
) -> dict[str, Any]:
    """단일 trial 실행 후 결과 반환."""
    fast_dev_settings = copy.deepcopy(solution.FAST_DEV_SETTINGS)
    if fast_dev:
        fast_dev_settings["enabled"] = True

    t0 = time.time()
    artifacts = solution.run_full_pipeline(
        config=None,
        fast_dev=fast_dev_settings,
        force_features=force_features,
        param_overrides=param_overrides,
    )
    elapsed = time.time() - t0
    logloss = artifacts["best_logloss"]
    return {
        "trial_id": trial_id,
        "logloss": logloss,
        "elapsed_sec": round(elapsed, 1),
        "param_overrides": param_overrides,
        "submission_path": str(artifacts["submission_path"]),
        "timestamp": datetime.now().isoformat(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전략별 파라미터 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def next_params_random() -> dict[str, Any]:
    """순수 랜덤 탐색."""
    return sample_trial_params()


def next_params_perturb(best_params: dict[str, Any]) -> dict[str, Any]:
    """현재 best 파라미터를 기반으로 일부만 교란 (local search)."""
    if not best_params:
        return sample_trial_params()
    # best 파라미터 중 1~2개만 재샘플링
    keys_to_change = random.sample(list(best_params.keys()), k=random.randint(1, min(2, len(best_params))))
    new_params = copy.deepcopy(best_params)
    for key in keys_to_change:
        # 검색 공간에서 해당 키의 분포를 찾아 재샘플링
        for dotpath, dist_type, params in SEARCH_SPACE:
            if dotpath == key:
                new_params[key] = sample_value(dist_type, params)
                break
    return new_params


def next_params_sobol(trial_id: int) -> dict[str, Any]:
    """Quasi-random (Sobol-like) low-discrepancy 탐색 — 간단 구현."""
    # 완전한 Sobol 없이 van der Corput 시퀀스로 균일 커버리지 근사
    def van_der_corput(n: int, base: int = 2) -> float:
        vdc, denom = 0.0, 1.0
        while n:
            denom *= base
            n, rem = divmod(n, base)
            vdc += rem / denom
        return vdc

    overrides: dict[str, Any] = {}
    for idx, (dotpath, dist_type, params) in enumerate(SEARCH_SPACE):
        t = van_der_corput(trial_id + 1, base=idx + 2)
        if dist_type == "uniform":
            lo, hi = params
            overrides[dotpath] = round(lo + t * (hi - lo), 6)
        elif dist_type == "loguniform":
            lo, hi = params
            overrides[dotpath] = round(math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo))), 8)
        elif dist_type == "int":
            lo, hi, step = params
            choices = list(range(lo, hi + 1, step))
            overrides[dotpath] = choices[int(t * len(choices)) % len(choices)]
        elif dist_type == "choice":
            overrides[dotpath] = params[int(t * len(params)) % len(params)]
    return overrides


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          CH2026 Auto-Tuner  ·  OOF Log-Loss 최소화              ║
║  Ctrl+C 로 종료 → 최적 파라미터는 configs/best_params.json 에 저장 ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_trial_header(trial_id: int, strategy: str, overrides: dict[str, Any]) -> None:
    print(f"\n{'─'*65}")
    print(f"  Trial #{trial_id:03d}  |  전략: {strategy}")
    print(f"  파라미터 변경:")
    for k, v in overrides.items():
        print(f"    {k} = {v}")
    print(f"{'─'*65}")


def print_trial_result(result: dict[str, Any], best_logloss: float, improved: bool) -> None:
    mark = "✅ 개선!" if improved else "  "
    print(f"  {mark} logloss={result['logloss']:.6f}  (best={best_logloss:.6f})  "
          f"소요={result['elapsed_sec']:.0f}s")
    if improved:
        print(f"  → submission: {result['submission_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CH2026 자동 하이퍼파라미터 튜닝")
    parser.add_argument("--max-trials", type=int, default=0,
                        help="최대 trial 수 (0=무한)")
    parser.add_argument("--strategy", choices=["random", "perturb", "sobol", "mixed"],
                        default="mixed", help="탐색 전략")
    parser.add_argument("--fast-dev", action="store_true",
                        help="빠른 데이터 서브샘플 모드")
    parser.add_argument("--force-features", action="store_true",
                        help="feature cache를 강제 재생성")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드 고정")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print_banner()

    # 기존 히스토리 로드
    history = load_history()
    best_params, best_logloss = load_best_params()
    start_trial = len(history) + 1

    if best_params is not None:
        print(f"📂 기존 best 로드: logloss={best_logloss:.6f} ({len(history)} trials 완료)")
        print(f"   파라미터: {json.dumps(best_params, ensure_ascii=False)}\n")
    else:
        print("📂 이전 기록 없음 — 첫 번째 trial은 기본 CONFIG로 baseline 측정\n")

    # ── 전략 선택 함수 ────────────────────────────────────────────────────
    def pick_strategy(trial_id: int) -> tuple[str, dict[str, Any]]:
        if args.strategy == "random":
            return "random", next_params_random()
        if args.strategy == "perturb":
            return "perturb", next_params_perturb(best_params or {})
        if args.strategy == "sobol":
            return "sobol", next_params_sobol(trial_id)
        # mixed: 첫 2회는 sobol로 공간 탐색, 이후 perturb 60% / random 40%
        if trial_id <= 2:
            return "sobol", next_params_sobol(trial_id)
        if random.random() < 0.6 and best_params:
            return "perturb", next_params_perturb(best_params)
        return "random", next_params_random()

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    trial_id = start_trial
    try:
        while True:
            if args.max_trials > 0 and (trial_id - start_trial) >= args.max_trials:
                print(f"\n🏁 max-trials={args.max_trials} 도달, 종료합니다.")
                break

            # 첫 번째 trial이고 기존 best가 없으면 기본 CONFIG 사용
            if trial_id == 1 and best_params is None:
                strategy_name = "baseline"
                overrides: dict[str, Any] = {}
            else:
                strategy_name, overrides = pick_strategy(trial_id)

            print_trial_header(trial_id, strategy_name, overrides)

            try:
                result = run_trial(
                    trial_id=trial_id,
                    param_overrides=overrides,
                    fast_dev=args.fast_dev,
                    force_features=args.force_features and trial_id == start_trial,
                )
                result["strategy"] = strategy_name
                result["error"] = None

                improved = result["logloss"] < best_logloss
                if improved:
                    best_logloss = result["logloss"]
                    best_params = overrides
                    save_best_params(best_params, best_logloss)

                print_trial_result(result, best_logloss, improved)
                append_history(result)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"\n❌ Trial #{trial_id} 오류: {e}")
                print(err_msg[:500])
                result = {
                    "trial_id": trial_id,
                    "logloss": float("inf"),
                    "elapsed_sec": 0,
                    "param_overrides": overrides,
                    "strategy": strategy_name,
                    "timestamp": datetime.now().isoformat(),
                    "error": err_msg,
                }
                append_history(result)

            trial_id += 1

    except KeyboardInterrupt:
        print("\n\n⏹  중단됨.")

    # ── 최종 요약 출력 ────────────────────────────────────────────────────
    all_history = load_history()
    completed = [r for r in all_history if r.get("error") is None]
    print(f"\n{'═'*65}")
    print(f"  완료된 trial: {len(completed)} / {len(all_history)}")
    if completed:
        top = sorted(completed, key=lambda r: r["logloss"])[:5]
        print(f"  Top-5 결과:")
        for i, r in enumerate(top, 1):
            print(f"    {i}. logloss={r['logloss']:.6f}  "
                  f"trial=#{r['trial_id']}  "
                  f"params={json.dumps(r['param_overrides'], ensure_ascii=False)}")
    print(f"\n  🏆 최적 logloss: {best_logloss:.6f}")
    print(f"  📄 최적 파라미터: {BEST_PARAMS_PATH}")
    print(f"  📜 전체 기록:    {TUNE_HISTORY_PATH}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
