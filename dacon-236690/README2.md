# ETRI Lifelog 2026 Challenge — 앙상블 예측 파이프라인

이 코드는 단순히 `train.csv`를 넣고 모델 하나를 돌리는 구조가 아닙니다.

전체 흐름은 아래처럼 진행됩니다.

```text
데이터 로드
→ 센서 데이터 전처리
→ 일 단위 / 시간대별 시계열 피처 생성
→ 피험자별 상대 피처 생성
→ lag 피처 생성
→ Stage 1 모델 학습
→ Stage 2 스태킹 / 멀티시드 학습
→ 앙상블
→ 블렌딩
→ 테스트 예측
→ 제출 파일 저장
```

목표는 `Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4` 총 7개 타깃에 대해 확률값을 예측하는 것입니다.

평가 산식은 Average Log-Loss이므로 제출 파일에는 `0` 또는 `1`이 아니라 `0~1` 사이의 확률값이 들어가야 합니다.

---

## 1. 목표 변수

예측해야 하는 타깃은 아래 7개입니다.

```text
Q1
Q2
Q3
S1
S2
S3
S4
```

각 타깃은 이진 분류 문제입니다.

```text
0 또는 1 정답값을 직접 맞추는 것이 아니라,
각 타깃이 1일 확률을 예측합니다.
```

예측 예시:

```text
Q1 = 0.5123
Q2 = 0.4381
Q3 = 0.6910
S1 = 0.7024
S2 = 0.6115
S3 = 0.5812
S4 = 0.4977
```

Log-Loss에서는 너무 확신하는 예측이 틀리면 손해가 크기 때문에, 모델 출력은 마지막에 안정적인 확률 범위로 정리합니다.

---

## 2. 사용 파일

주요 실행 파일은 아래와 같습니다.

```text
solution.py
advanced_ensemble.py
best_baseline.py
requirements.txt
README.md
```

각 파일의 역할은 다음과 같습니다.

```text
solution.py
→ 기본 앙상블 파이프라인
→ LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees 사용
→ Stacking + Voting 결과를 섞어 제출 파일 생성

advanced_ensemble.py
→ Log-Loss 기준 Optuna 탐색, 멀티시드 학습, subject prior 블렌딩 포함
→ calibration / shrinkage 후보를 비교해 확률 예측을 안정화하는 용도

best_baseline.py
→ 빠르게 비교할 수 있는 별도 베이스라인
→ 센서 집계 + subject prior + 모델 앙상블 구조
```

데이터 파일은 루트 또는 `ch2025_data_items/` 폴더 안에 있으면 자동으로 찾습니다.

```text
ch2026_metrics_train.csv
ch2026_submission_sample.csv
ch2025_data_items/*.parquet
```

---

## 3. 핵심 실행 파라미터

코드는 환경변수로 실행 강도를 조절합니다.

```text
FAST_TREES
→ 각 트리 모델의 트리 개수 또는 반복 수를 조절
→ 값이 커지면 더 오래 학습하지만 모델 표현력이 커짐

N_SEEDS
→ advanced_ensemble.py에서 여러 random seed로 반복 학습할 횟수
→ 값이 커지면 예측이 안정적이지만 시간이 오래 걸림

N_OPTUNA
→ advanced_ensemble.py에서 Optuna 탐색 횟수
→ 값이 커지면 더 많은 하이퍼파라미터 조합을 탐색

N_FOLDS
→ 교차검증 fold 수
→ 기본값은 코드에서 설정된 값을 사용

CLASS_WEIGHT_MODE
→ solution.py에서 balanced class weight를 쓸지 선택
→ 기본값은 none

ENABLE_PSEUDO
→ advanced_ensemble.py에서 pseudo-labeling을 켤지 선택
→ 기본값은 0, 즉 OFF
```

주요 기본값:

```text
SEED = 42
N_FOLDS = 5
FAST_TREES = 실행 명령에서 지정
N_SEEDS = 실행 명령에서 지정
N_OPTUNA = 실행 명령에서 지정
CLASS_WEIGHT_MODE = none
ENABLE_PSEUDO = 0
```

`SEED = 42`는 재현성을 위한 랜덤 시드입니다.

같은 설정으로 다시 실행하면 가능한 한 비슷한 결과가 나오도록 합니다.

---

## 4. 데이터 전처리

전처리는 단순 결측치 제거가 아니라 센서별 특성을 고려해서 진행됩니다.

처리 흐름:

```text
1. train / test 메타 데이터 로드
2. subject_id, lifelog_date, sleep_date 기준으로 행 구성
3. 각 센서 parquet 파일 로드
4. subject_id별 센서 데이터 분리
5. lifelog_date 또는 sleep window 기준으로 피처 계산
6. train/test가 같은 구조의 feature matrix를 갖도록 병합
```

사용하는 센서 데이터:

```text
ch2025_wPedo.parquet
ch2025_wLight.parquet
ch2025_wHr.parquet
ch2025_mActivity.parquet
ch2025_mScreenStatus.parquet
ch2025_mLight.parquet
ch2025_mACStatus.parquet
ch2025_mUsageStats.parquet
ch2025_mAmbience.parquet
ch2025_mWifi.parquet
ch2025_mGps.parquet
```

센서별로 데이터 형태가 다릅니다.

예를 들어 `wPedo`는 숫자형 컬럼이 많고, `mWifi`, `mGps`, `mUsageStats`, `mAmbience`는 리스트 또는 dict 형태의 값이 들어 있습니다.

따라서 센서마다 별도의 feature extraction 함수가 필요합니다.

---

## 5. 시계열 피처 생성

이 문제는 하루 단위 lifelog를 보고 수면 관련 타깃을 예측합니다.

그래서 단순 평균값만 쓰지 않고 시간 흐름을 반영합니다.

대표적인 시간 구간:

```text
아침   : 06시 ~ 12시
오후   : 12시 ~ 18시
저녁   : 18시 ~ 24시
취침창 : lifelog_date 22시 ~ sleep_date 10시
```

센서별 예시:

```text
wPedo
→ 하루 전체 걸음 수
→ 아침 / 오후 / 저녁 걸음 수
→ 이동거리, 칼로리, 달리기 비율

wLight
→ 취침창 평균 조도
→ 취침창 최대 조도
→ 어두운 시간 비율

wHr
→ 취침창 평균 심박
→ 최소 / 최대 심박
→ 심박 변동성 대리지표

mScreenStatus
→ 하루 화면 사용 비율
→ 저녁 화면 사용 비율
→ 늦은 밤 화면 사용 횟수
→ 마지막 화면 사용 시각

mActivity
→ 정지 / 걷기 / 차량 / 달리기 비율
→ 저녁 시간대 정적 활동 비율

mUsageStats
→ 총 앱 사용 시간
→ SNS 또는 주요 앱 사용 시간
→ 앱 사용 개수

mWifi / mGps
→ 이동성 또는 장소 변화의 대리지표
```

왜 필요한가:

```text
수면 관련 타깃은 하루 전체 평균보다
저녁 행동, 취침 직전 화면 사용, 야간 조도, 심박 안정성 같은 흐름에 영향을 많이 받을 수 있습니다.
```

따라서 시간대별 피처가 모델에 중요한 신호를 제공합니다.

---

## 6. 상대 피처와 lag 피처

Q 계열 타깃은 개인의 주관적 상태와 관련되어 있습니다.

같은 걸음 수라도 사람마다 의미가 다를 수 있습니다.

예시:

```text
A 피험자에게 8,000보는 평소보다 적은 활동일 수 있음
B 피험자에게 8,000보는 평소보다 많은 활동일 수 있음
```

그래서 피험자별 상대 피처를 만듭니다.

```python
relative_feature = (value - subject_mean) / subject_std
```

lag 피처도 추가합니다.

```text
lag1
lag2
lag3
```

의미:

```text
lag1 → 전날 값
lag2 → 2일 전 값
lag3 → 3일 전 값
```

lag 피처는 최근 며칠간의 활동 패턴과 수면 패턴을 모델이 이해하도록 돕습니다.

---

## 7. 타깃 정보와 누수 방지

이 프로젝트에서는 타깃값을 직접 test feature로 사용할 수 없습니다.

따라서 train에서만 알 수 있는 정답 정보를 test에 새어나가게 만들면 안 됩니다.

주의해야 하는 상황:

```text
train의 정답 평균을 그대로 validation에 넣는 경우
같은 subject의 행이 train과 validation에 섞이는 경우
미래 날짜의 정보를 과거 예측에 사용하는 경우
```

이를 줄이기 위해 코드에서는 `subject_id` 기준 GroupKFold를 사용합니다.

```python
GroupKFold(n_splits=5)
```

의미:

```text
같은 피험자의 데이터가 train fold와 validation fold에 동시에 들어가지 않도록 나눕니다.
```

`best_baseline.py`에서는 subject별 과거 타깃 분포를 prior처럼 사용합니다.

이때도 fold 안에서 validation 정답을 직접 보지 않도록 OOF 방식으로 계산합니다.

---

## 8. Stage 1 학습

Stage 1은 일반 센서 피처와 시계열 피처를 중심으로 학습합니다.

사용 모델:

```text
LightGBM
XGBoost
CatBoost
RandomForest
ExtraTrees
```

각 타깃별로 모델을 따로 학습합니다.

```text
Q1 모델
Q2 모델
Q3 모델
S1 모델
S2 모델
S3 모델
S4 모델
```

Stage 1의 역할:

```text
센서 피처만으로 각 타깃의 기본적인 확률 예측을 만듭니다.
```

OOF 평가도 진행합니다.

```text
각 fold에서 validation 예측을 만들고,
타깃별 점수를 확인합니다.
```

이 단계는 모델이 과도하게 한 피험자나 특정 날짜 패턴에 맞춰지는 것을 줄이기 위한 기본 검증 단계입니다.

---

## 9. Stage 2 학습과 앙상블

Stage 2는 Stage 1보다 더 복합적인 조합을 사용합니다.

`solution.py`에서는 두 가지 흐름을 함께 사용합니다.

```text
1. Stacking
2. Soft Voting
```

### Stacking

base model들이 먼저 예측합니다.

```text
LightGBM
XGBoost
CatBoost
RandomForest
ExtraTrees
```

그 다음 Logistic Regression이 base model의 예측값과 원래 feature를 함께 보고 최종 확률을 만듭니다.

```text
base model 예측
→ meta learner 입력
→ 최종 확률 예측
```

### Soft Voting

여러 모델의 확률 예측을 평균냅니다.

```text
LGB 확률
XGB 확률
CAT 확률
RF 확률
ET 확률
→ 평균 확률
```

최종적으로 `solution.py`에서는 stacking과 voting을 섞습니다.

```python
final_p = p_stack * 0.6 + p_vote * 0.4
```

의미:

```text
Stacking이 잡는 복합 패턴과
Voting이 주는 안정성을 함께 사용합니다.
```

---

## 10. advanced_ensemble.py 흐름

`advanced_ensemble.py`는 더 오래 돌려서 예측을 안정화하는 실행 파일입니다.

진행 흐름:

```text
1. feature cache 로드
2. 상대 피처와 lag 피처 추가
3. 타깃별 Optuna 하이퍼파라미터 탐색 (binary Log-Loss minimize)
4. 여러 seed로 반복 학습
5. GroupKFold OOF 예측
6. subject prior와 모델 예측 블렌딩
7. raw / sigmoid / isotonic / shrinkage 후보 비교
8. threshold는 F1 참고 지표로만 기록
9. pseudo-labeling은 기본 OFF
10. test 예측
11. submission_v다음번호.csv 저장
```

사용 모델:

```text
LightGBM
XGBoost
CatBoost
RandomForest
ExtraTrees
```

Optuna는 각 타깃별로 모델 파라미터 후보를 탐색합니다.

모델 선택 기준은 `macro-F1`이 아니라 `binary Log-Loss`입니다.

```text
primary   : OOF Log-Loss 낮을수록 좋음
secondary : MAE, Brier 낮을수록 좋음
reference : macro-F1 높을수록 좋음
```

```text
learning_rate
n_estimators
num_leaves
max_depth
subsample
subsample_freq
colsample_bytree
regularization
class_weight / scale_pos_weight 후보
```

멀티시드는 같은 구조를 여러 random seed로 반복 학습하여 예측 분산을 줄입니다.

```text
seed 42
seed 43
seed 44
...
```

이 방식은 계산량이 늘어나지만 예측이 한 번의 random split에 덜 흔들리게 만듭니다.

---

## 11. 블렌딩과 후처리

여러 모델의 예측을 그대로 하나만 쓰지 않고 섞습니다.

사용되는 조합:

```text
Stacking + Voting
LightGBM + XGBoost + CatBoost
Tree model + subject prior
Multi-seed average
sigmoid calibration
isotonic calibration
target mean shrinkage
```

블렌딩을 하는 이유:

```text
모델마다 잘 맞추는 패턴이 다르기 때문입니다.

LightGBM은 tabular feature에 강하고,
XGBoost는 다른 tree boosting 구조를 제공하고,
CatBoost는 범주형 / 작은 데이터에서 안정적인 경우가 있습니다.
RandomForest와 ExtraTrees는 boosting 모델과 다른 방식의 다양성을 제공합니다.
```

최종 예측은 제출 전에 확률값으로 정리합니다.

```text
너무 작은 값은 0 근처로 고정
너무 큰 값은 1 근처로 고정
결측값이 없도록 확인
```

Log-Loss에서는 확률값의 안정성이 중요하기 때문에 이 단계가 필요합니다.

후처리 후보의 OOF 지표는 아래 파일에 저장됩니다.

```text
outputs/model_comparison_report.json
```

---

## 12. 제출 파일 저장

제출 파일 이름은 자동으로 숫자 버전이 증가합니다.

```text
outputs/submission_v1.csv
outputs/submission_v2.csv
outputs/submission_v3.csv
outputs/submission_v4.csv
```

가장 큰 번호가 가장 최근 생성된 제출 파일입니다.

예전처럼 아래 이름은 새로 만들지 않습니다.

```text
submission.csv
submission_advanced.csv
submission_best_baseline.csv
```

저장되는 제출 파일 구조:

```text
subject_id
sleep_date
lifelog_date
Q1
Q2
Q3
S1
S2
S3
S4
```

---

## 13. 실행 명령어

먼저 프로젝트 폴더로 이동합니다.

```bash
cd /Users/samrobert/Documents/GitHub/dacon-236690/dacon-236690
```

패키지를 설치합니다.

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

기본 버전 실행:

```bash
FAST_TREES=120 ./.venv/bin/python -u solution.py
```

advanced 빠른 실행:

```bash
N_SEEDS=1 N_OPTUNA=3 FAST_TREES=80 ./.venv/bin/python -u advanced_ensemble.py
```

advanced 4시간 권장 실행:

```bash
N_SEEDS=7 N_OPTUNA=60 FAST_TREES=260 ./.venv/bin/python -u advanced_ensemble.py
```

advanced 더 무거운 실행:

```bash
N_SEEDS=10 N_OPTUNA=80 FAST_TREES=300 ./.venv/bin/python -u advanced_ensemble.py
```

`python` 명령이 다른 환경을 볼 수 있으므로, 항상 `./.venv/bin/python`을 직접 지정합니다.

---

## 14. 4시간 실행 로그 확인

4시간 정도 돌리는 실행은 아래 설정을 사용합니다.

```bash
N_SEEDS=7 N_OPTUNA=60 FAST_TREES=260 ./.venv/bin/python -u advanced_ensemble.py
```

백그라운드 실행을 걸어둔 경우 로그는 아래 명령어로 확인합니다.

```bash
tail -f "$(cat outputs/advanced_4h.logpath)"
```

프로세스가 살아있는지 확인합니다.

```bash
ps -p "$(cat outputs/advanced_4h.pid)" -o pid,etime,%cpu,%mem,command
```

중간에 멈추고 싶으면 아래 명령어를 사용합니다.

```bash
kill "$(cat outputs/advanced_4h.pid)"
```

---

## 15. 캐시 파일

처음 실행하면 feature cache가 생성됩니다.

```text
outputs/solution_features.csv
outputs/solution_features.parquet
```

다음 실행부터는 센서 feature를 다시 만들지 않고 cache를 읽습니다.

`pyarrow` 문제를 피하기 위해 코드는 `solution_features.csv`를 먼저 읽습니다.

즉, parquet 엔진이 없는 Python 환경에서도 cache 읽기 단계에서 덜 터지도록 구성했습니다.

---

## 16. Python 환경 확인

현재 터미널의 `python`이 어떤 환경을 보는지 확인하려면 아래 명령어를 사용합니다.

```bash
which python
```

```bash
python -c "import sys; print(sys.executable)"
```

정상 실행은 아래처럼 직접 `.venv` Python을 사용하는 방식입니다.

```bash
./.venv/bin/python -u solution.py
```

이렇게 하면 시스템 Python이 잡혀서 `pyarrow`를 못 찾는 문제를 피할 수 있습니다.
