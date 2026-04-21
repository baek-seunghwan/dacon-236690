# DACON CH2026 Strong Baseline

이 베이스라인은 `Average Log-Loss`에 맞춰 7개 이진 타깃(`Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4`)의 확률값을 예측합니다.

## 구성

- `best_baseline.py`: feature engineering, OOF 검증, 최종 학습, submission 생성까지 한 번에 실행
- `requirements.txt`: 실행에 필요한 최소 패키지
- `outputs/submission_best_baseline.csv`: 제출용 파일
- `outputs/validation_report.json`: 타깃별 OOF Log-Loss, 선택된 앙상블/블렌딩 정보
- `outputs/feature_importance.csv`: 최종 LightGBM 중요도 합산

## 실행

```bash
pip install -r requirements.txt
python best_baseline.py
```

feature를 다시 만들고 싶으면 다음처럼 실행합니다.

```bash
python best_baseline.py --force-features
```

## 현재 검증 결과

현재 스크립트 기준 5-fold OOF `Average Log-Loss`는 `0.60337`입니다.

단순 global mean baseline의 train log-loss는 약 `0.66413`이므로, 센서 일 단위 집계와 subject prior를 통해 의미 있는 개선을 만든 상태입니다.

## 방법 요약

- 모든 parquet 센서 로그를 `subject_id + lifelog_date` 단위로 집계
- 시간대별 feature, rolling/lag feature, 앱/ambience top category feature 생성
- 타깃별로 `LightGBM`, `ExtraTrees`, `LogisticRegression` 후보를 OOF에서 비교
- 같은 subject의 과거 라벨 분포를 empirical Bayes prior로 만들고, 모델 예측과 OOF 기준으로 블렌딩
- 최종 제출값은 `1e-5`와 `1 - 1e-5` 사이로 clipping
