"""DFL 단독 xG 모델 — StatsBomb 데이터를 전혀 쓰지 않는 비교 기준.

v2(train_v2.py)는 StatsBomb으로 학습한 모델을 DFL에 맞춰 조정한다.
이 스크립트는 같은 DFL 159슛 · 같은 평가(경기 단위 LOMO 7폴드)로 'DFL만으로 학습'하면 어디까지 되는지 본다.
StatsBomb 데이터는 읽지 않는다. train_v1/v2에서 가져오는 건 피처 이름 목록과 평가 함수뿐이다.

모델 (전부 사전 고정 — DFL 결과를 보고 고르지 않음)
    DFL0  거리·각도만                  가장 단순한 xG
    DFL1  공통 피처 21개               v2 베이스(M0)와 같은 피처, 학습 데이터만 DFL
    DFL2  공통 피처 + 트래킹 2개        v2 M2와 같은 입력
    규제: LogReg C=0.1 (= train_v2의 λ=10과 같은 강도). 골 16개라 규제가 약하면 과적합한다.

비교 (out/v2_oof.csv가 있으면): 같은 입력에서 'StatsBomb 사전학습 − DFL 단독'
    M0 − DFL1   StatsBomb으로 학습한 효과
    M2 − DFL2   트래킹까지 넣었을 때
"""
import os, sys, time, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))  # 물리 코어 탐지(wmic) 트레이스백 방지
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v1 import COMMON_NUM, OUT, cluster_boot, design
from train_v2 import TRACK

warnings.filterwarnings("ignore")
C_REG = 0.1

MODELS = {
    "DFL0": ("거리·각도만", ["dist", "angle"]),
    "DFL1": ("공통 피처 (M0와 같은 피처)", COMMON_NUM),
    "DFL2": ("공통 피처 + 트래킹 (M2와 같은 입력)", COMMON_NUM + TRACK),
}


def lomo(X, y, g):
    """경기 하나씩 빼고 나머지 6경기로 학습 -> 빠진 경기 예측. 표준화·결측 채움도 학습 경기에서만 맞춘다."""
    p = np.zeros(len(y))
    for m in np.unique(g):
        tr, te = g != m, g == m
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=C_REG, max_iter=3000))
        p[te] = mdl.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    t0 = time.time()
    dfl = pd.read_csv(os.path.join(OUT, "ds_dfl.csv"))
    y, g = dfl.goal.values, dfl.match_id.values
    print(f"DFL 단독: {len(dfl)}슛 · {y.sum()}골 · {len(np.unique(g))}경기 -> 경기 단위 LOMO "
          f"(StatsBomb 사용 안 함, LogReg C={C_REG})\n", flush=True)

    P, desc, nfeat = {}, {}, {}
    for key, (d, num) in MODELS.items():
        X = design(dfl, num, []).values                  # log_dist가 붙는다 (v2 베이스와 같은 방식)
        P[key], desc[key], nfeat[key] = lomo(X, y, g), d, X.shape[1]
    P["REF"], desc["REF"], nfeat["REF"] = dfl.ref_xg.values, "DFL 제공 xG (참고)", np.nan

    rows = []
    for key, p in P.items():
        ci, dci = cluster_boot(y, p, g, P["DFL0"])
        rows.append(dict(model=key, desc=desc[key], n_feat=nfeat[key],
                         auc=roc_auc_score(y, p), auc_lo=ci[0], auc_hi=ci[1],
                         d_vs_dfl0=roc_auc_score(y, p) - roc_auc_score(y, P["DFL0"]), d_lo=dci[0], d_hi=dci[1],
                         logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                         pred_goals=p.sum(), calib=p.sum() / y.sum()))
        r = rows[-1]
        print(f"  {key:4s} {r['desc']:28s} AUC {r['auc']:.3f} [{r['auc_lo']:.2f}, {r['auc_hi']:.2f}]  "
              f"logloss {r['logloss']:.4f}  예측 골 {r['pred_goals']:.1f} / 실제 {y.sum()}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "dfl_results.csv"), index=False)
    pd.DataFrame({"shot_id": dfl.shot_id, "match_id": g, "goal": y, **P}).to_csv(
        os.path.join(OUT, "dfl_oof.csv"), index=False)

    v2_path = os.path.join(OUT, "v2_oof.csv")
    if os.path.exists(v2_path):
        v2 = pd.read_csv(v2_path).set_index("shot_id").loc[dfl.shot_id]
        print("\nStatsBomb 사전학습(v2) − DFL 단독 · 같은 슛, 같은 LOMO  (AUC 차이 [경기 부트스트랩 95% CI], logloss 차이)")
        for a, b, what in [("M0", "DFL1", "같은 피처 21개"), ("M2", "DFL2", "같은 피처 + 트래킹")]:
            pa = v2[a].values
            _, dci = cluster_boot(y, pa, g, P[b])
            print(f"  {a} − {b}  ({what:12s})  AUC {roc_auc_score(y, pa) - roc_auc_score(y, P[b]):+.3f} "
                  f"[{dci[0]:+.2f}, {dci[1]:+.2f}]  logloss {log_loss(y, pa) - log_loss(y, P[b]):+.4f} (음수면 v2가 나음)")

    print(f"\n-> out/dfl_results.csv, dfl_oof.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
