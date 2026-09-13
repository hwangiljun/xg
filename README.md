# DFL 7경기 xG 모델

슛 하나가 골이 될 확률(xG)을 계산하는 모델. DFL이 공개한 분데스리가 7경기 데이터를 대상으로 한다.

DFL 7경기는 페널티를 빼면 골이 16개뿐이라 이것만으로는 학습도 평가도 어렵다. 그래서 StatsBomb이 공개한 라리가 2015/16 시즌 380경기로 먼저 모델을 만들고, DFL에는 그 모델을 가져와 조정해서 쓴다.

- v1 (`src/train_v1.py`): StatsBomb 이벤트 데이터(슛 순간 선수 위치 포함)로 학습하고 평가
- v2 (`src/train_v2.py`): v1에서 DFL에도 있는 정보만 쓴 모델을 DFL에 맞춰 조정하고, DFL 트래킹 정보(슈터 속도, 수비수가 좁혀온 거리)를 추가
- 비교 기준 (`src/train_dfl.py`): StatsBomb 없이 DFL로만 학습한 모델

## 현재 결과

2026년 9월 13일 기준. AUC는 골이 된 슛과 안 된 슛을 하나씩 뽑았을 때 모델이 골이 된 쪽에 더 높은 확률을 줄 가능성이다.

| 평가 데이터 | 모델 | AUC |
|---|---|---|
| StatsBomb 시즌 마지막 76경기 (최종 점검) | v1 최종 모델 | 0.812 |
| | StatsBomb 자체 xG (참고) | 0.831 |
| DFL 159슛 (한 경기씩 빼고 평가) | v2, StatsBomb 모델 그대로 | 0.771 |
| | v2, 리그 보정 + 트래킹 | 0.760 |
| | DFL로만 학습 (같은 정보) | 0.720 |
| | DFL 제공 xG (참고) | 0.823 |

DFL은 골이 16개라 95% 구간이 0.5~0.9 정도로 넓다. 목표였던 AUC 0.83에는 아직 못 미친다.

## 재현 방법

### 1. 환경

Python 3.12에서 테스트했다.

```
git clone https://github.com/hwangiljun/xg.git
cd xg
pip install -r requirements.txt
```

### 2. 데이터

**StatsBomb**: 따로 받을 필요 없다. 실행할 때 `src/fetch_statsbomb.py`가 [StatsBomb Open Data](https://github.com/statsbomb/open-data)에서 라리가 2015/16 380경기를 `data/statsbomb/`로 받는다 (약 110MB).

**DFL**: [An integrated dataset of spatiotemporal and event data in elite soccer (figshare)](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)에서 7경기의 파일을 받아 `Dfl/` 폴더에 넣는다. 경기마다 아래 세 종류, 모두 21개다.

| 파일 | 내용 |
|---|---|
| `DFL_02_01_matchinformation_*.xml` | 경기와 선수 정보 |
| `DFL_03_02_events_raw_*.xml` | 이벤트 |
| `DFL_04_03_positions_raw_observed_*.xml` | 트래킹 (25Hz, 약 2.5GB) |

DFL 원본이 없어도 돌아간다. 그때는 저장소에 들어 있는 DFL 가공 결과(`out/ds_dfl.csv` 등)를 그대로 쓰고 StatsBomb 쪽만 처음부터 계산한다. 노트북에서는 DFL 트래킹 장면 그림 하나만 빠진다.

### 3. 실행

`explore.ipynb`를 열고 모두 실행하면 된다. 맨 위 0번 셀이 데이터 준비부터 학습과 평가까지 다시 계산하고, 나머지 셀이 그 결과로 표와 그래프를 그린다. 이미 계산된 결과만 보려면 0번 셀의 `RUN_PIPELINE`을 `False`로 바꾼다.

터미널에서 돌려도 결과는 같다.

```
python run_all.py            # 전체 (DFL 원본 필요)
python run_all.py --list     # 단계 목록
python run_all.py --from train
```

### 4. 결과가 같은지

난수 시드를 고정했다. 같은 환경에서 다시 돌리면 AUC 같은 숫자가 소수점까지 같게 나온다.

## 파이프라인

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| matchinfo | `src/parse_matchinfo.py` | DFL 경기와 선수 정보 |
| events | `src/parse_events.py` | DFL 슛 이벤트 |
| freeze | `src/extract_freeze.py` | DFL 트래킹에서 슛 앞뒤 1초 프레임만 추출 |
| statsbomb | `src/fetch_statsbomb.py` | StatsBomb 데이터 받기 |
| dataset | `src/build_dataset.py` | 두 데이터에서 같은 정의로 피처 계산 (`src/geometry.py`) |
| train | `src/train_v1.py` | v1 학습과 평가 |
| finetune | `src/train_v2.py` | v2, DFL에 맞춰 조정 |
| dfl_only | `src/train_dfl.py` | DFL로만 학습한 비교 모델 |

`src/`의 `eda.py`, `feasibility.py`, `ceiling.py`, `search.py`, `nested.py` 등은 DFL만으로 모델을 만들 수 있는지 처음에 따져본 분석이다. `python run_all.py --analysis`로 함께 돌린다.

## 평가 방법

- 페널티킥은 뺐다.
- StatsBomb은 경기 단위로 날짜순으로 나눴다. 앞 80%(304경기)는 모델 비교용으로 쓰고 그 안에서 경기 단위 5-fold로 검증했다. 마지막 20%(76경기)는 고른 모델을 한 번만 확인하는 최종 점검용이다.
- DFL은 7경기 중 1경기를 빼고 나머지로 학습하거나 조정한 뒤 빠진 경기를 예측하는 것을 7번 반복했다.
- AUC 95% 구간은 경기를 다시 뽑는 부트스트랩 1,000번으로 구했다.

## 데이터 출처

- DFL: *An integrated dataset of spatiotemporal and event data in elite soccer*, figshare, [링크](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)
- StatsBomb: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
