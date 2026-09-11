"""v2: StatsBomb으로 학습한 공통 피처 모델(P)을 DFL 7경기에 맞춰 파인튜닝.

SB 모델은 고정하고, 그 로짓 z_SB 위에 조정값 몇 개만 얹는다.
    xG_DFL = σ( z_SB + a + b·(z_SB − z̄) + γ·트래킹 )
    a   리그 보정 — 분데스리가 전체 득점 수준 (0이면 SB 모델 그대로)
    b   리그 보정 — SB 모델의 확신 정도를 키우거나 줄임 (0이면 그대로)
    γ   DFL 트래킹 전용 피처 효과 — v2의 핵심
    조정값 전부에 L2 규제 -> 데이터가 강하게 가리키지 않으면 SB 모델에서 벗어나지 않는다.
    골 16개로 가중치 20여 개를 다시 맞추면 노이즈를 학습하므로 '몇 개만' 조정한다.

비교 (DFL 경기 단위 LOMO 7폴드: 6경기로 조정 -> 빠진 1경기 예측. 설정은 전부 사전 고정)
    M0   SB 모델 그대로          DFL을 한 번도 안 봄 = 외부 테스트
    M1   + 리그 보정 (a, b)
    M2   + 리그 보정 + 트래킹 (a, b, γ)
    REF  DFL 제공 xG             참고용

주의: DFL 비페널티 16골 -> AUC CI 폭이 0.3 안팎. 여기서 나오는 차이는 '방향 확인용'이지 증명이 아니다.
"""
import os, sys, time, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))  # 물리 코어 탐지(wmic) 트레이스백 방지
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v1 import OUT, VARIANTS, cluster_boot, design, models

warnings.filterwarnings("ignore")
# 사전 선택 (DFL 골과의 관계를 보고 고르지 않음). 골 16개면 피처 1~2개가 한계.
TRACK = ["shooter_speed", "def_closing"]    # 슈터 속도(km/h), 가까운 수비 3명이 1초간 좁힌 거리(m)
LAM = 10.0                                  # L2 강도 — 조정값의 사전 표준편차 ≈ 1/√10 ≈ 0.3 (로짓)


def fit_adjust(z, F, y, lam=LAM):
    """y ~ σ(z + F·θ)에서 θ만 학습. 손실 = 로그손실 합 + lam/2·|θ|². z(SB 로짓)는 고정."""
    def loss(th):
        s = z + F @ th
        return (np.sum(np.logaddexp(0, s) - y * s) + 0.5 * lam * th @ th,
                F.T @ (expit(s) - y) + lam * th)
    return minimize(loss, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B").x


def adj_inputs(z, T, fit_idx, idx, track):
    """θ에 곱해질 입력 [1, z − z̄, 트래킹 표준화]. 중심·척도는 fit_idx(학습 경기)에서만 계산."""
    cols = [np.ones(len(idx)), z[idx] - z[fit_idx].mean()]
    if track:
        mu, sd = T[fit_idx].mean(axis=0), T[fit_idx].std(axis=0)
        cols += list(((T[idx] - mu) / sd).T)
    return np.column_stack(cols)


def main():
    t0 = time.time()
    sb = pd.read_csv(os.path.join(OUT, "ds_statsbomb.csv"))
    sb = sb[sb.shot_type != "Penalty"].reset_index(drop=True)
    dfl = pd.read_csv(os.path.join(OUT, "ds_dfl.csv"))
    _, num, _ = VARIANTS["P"]

    # 1) 베이스: 공통 피처 LogReg를 SB 비페널티 전체로 학습 (DFL은 보지 않음)
    Xs = design(sb, num, [])
    base = models()["LogReg"]().fit(Xs.values, sb.goal.values)
    z = base.decision_function(design(dfl, num, []).values)      # DFL 슛마다 SB 모델의 로짓
    y, g = dfl.goal.values, dfl.match_id.values
    T = dfl[TRACK].astype(float).values
    ms = np.unique(g)
    print(f"베이스: 공통 피처 LogReg — SB {len(sb):,}슛으로 학습, 피처 {Xs.shape[1]}개")
    print(f"DFL: {len(dfl)}슛 · {y.sum()}골 · {len(ms)}경기 -> 경기 단위 LOMO {len(ms)}폴드\n", flush=True)

    # 2) LOMO: 6경기로 조정값 학습 -> 빠진 1경기 예측
    P = {"M0": expit(z), "M1": np.zeros(len(y)), "M2": np.zeros(len(y))}
    for m in ms:
        tr, te = np.flatnonzero(g != m), np.flatnonzero(g == m)
        for key, track in (("M1", False), ("M2", True)):
            th = fit_adjust(z[tr], adj_inputs(z, T, tr, tr, track), y[tr])
            P[key][te] = expit(z[te] + adj_inputs(z, T, tr, te, track) @ th)
    P["REF"] = dfl.ref_xg.values

    desc = {"M0": "SB 모델 그대로 (외부 테스트)", "M1": "+ 리그 보정 (a, b)",
            "M2": "+ 리그 보정 + 트래킹 (a, b, γ)", "REF": "DFL 제공 xG (참고)"}
    rows = []
    for key, p in P.items():
        ci, dci = cluster_boot(y, p, g, P["M0"])
        rows.append(dict(model=key, desc=desc[key], auc=roc_auc_score(y, p), auc_lo=ci[0], auc_hi=ci[1],
                         d_vs_m0=roc_auc_score(y, p) - roc_auc_score(y, P["M0"]), d_lo=dci[0], d_hi=dci[1],
                         logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                         pred_goals=p.sum(), calib=p.sum() / y.sum()))
        r = rows[-1]
        print(f"  {key:3s} {r['desc']:28s} AUC {r['auc']:.3f} [{r['auc_lo']:.2f}, {r['auc_hi']:.2f}]  "
              f"vs M0 {r['d_vs_m0']:+.3f}  logloss {r['logloss']:.4f}  "
              f"예측 골 {r['pred_goals']:.1f} / 실제 {y.sum()}", flush=True)

    # 3) 해석용: 7경기 전부로 조정값을 다시 학습
    allr = np.arange(len(y))
    th = fit_adjust(z, adj_inputs(z, T, allr, allr, True), y)
    names = ["a", "b"] + [f"γ_{c}" for c in TRACK]
    print("\n최종 조정값 (7경기 전체, 로짓 단위 · 0이면 SB 모델 그대로)")
    print(f"  a  {th[0]:+.3f}   평균적인 슛의 골 odds ×{np.exp(th[0]):.2f}")
    print(f"  b  {th[1]:+.3f}   SB 로짓의 기울기 1 -> {1 + th[1]:.2f}")
    for c, v in zip(TRACK, th[2:]):
        print(f"  γ  {v:+.3f}   {c} +1 SD일 때 골 odds ×{np.exp(v):.2f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v2_results.csv"), index=False)
    pd.DataFrame({"shot_id": dfl.shot_id, "match_id": g, "goal": y, **P}).to_csv(
        os.path.join(OUT, "v2_oof.csv"), index=False)
    pd.DataFrame({"param": names, "value": th}).to_csv(os.path.join(OUT, "v2_params.csv"), index=False)
    print(f"\n-> out/v2_results.csv, v2_oof.csv, v2_params.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
