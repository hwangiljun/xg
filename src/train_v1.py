"""v1: StatsBomb La Liga 2015/16 이벤트(+프리즈프레임)로 xG 학습.

데이터 분할 (경기 단위, 시간순)
  dev   시즌 앞 80% 경기 (304경기) — 피처·모델 비교는 여기서만. 안에서 경기 단위 GroupKFold(5)
  test  시즌 마지막 20% 경기 (76경기) — dev 전체로 학습한 모델을 한 번만 평가 = 최종 성능

평가 원칙
  - 비페널티 슛만 (페널티는 상수 xG로 따로 처리하는 게 표준)
  - 하이퍼파라미터는 사전에 고정 (test로 튜닝하지 않음)
  - 최종 모델은 dev CV AUC로 고른다 (A/B/C/P 중. D는 주관 태그라 참고용)
  - AUC 신뢰구간은 '경기' 단위 부트스트랩 (슛 단위보다 보수적)
  - 같은 슛에 대해 statsbomb_xg와 짝지어 비교

피처 단계
  A  위치+부위        거리·각도·부위·기술·상황
  B  +어시스트·포제션  키패스 유형, 포제션 길이·직진성, 드리블 거리   <- 창의 이벤트 피처
  C  +프리즈프레임     open_angle 등 수비 배치 기하               <- 창의 기하 피처
  D  +분석관 태그      one_on_one, open_goal (주관 태그 — 참고용)
  P  공통 피처         위치·헤더/프리킥·프리즈프레임만. DFL에도 똑같이 있어 v2(train_v2.py)의 베이스
"""
import gzip, json, os, time, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))  # 물리 코어 탐지(wmic) 트레이스백 방지
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
SB_DIR = os.path.join(HERE, "..", "data", "statsbomb", "comp11_season27")
SEED = 0
TEST_FRAC = 0.2          # 시즌 마지막 20% 경기를 최종 테스트로

BASE_NUM = ["dist", "angle", "y_off", "inside_box", "under_pressure"]
BASE_CAT = ["body_part", "technique", "shot_type", "play_pattern"]
EVT_NUM = ["first_time", "aerial_won", "has_assist", "ast_length", "ast_cross", "ast_through",
           "ast_cutback", "ast_switch", "poss_duration", "poss_n_pass", "poss_start_x",
           "poss_directness", "carry_len"]
EVT_CAT = ["ast_height", "ast_setpiece", "prev_type"]
FF_NUM = ["open_angle", "open_frac", "open_frac_nogk", "gk_block_share", "n_def_cone", "n_def_3m",
          "n_def_5m", "d_def_min", "n_corridor", "d_corridor", "gk_dist_goal", "gk_dist_shot",
          "gk_in_cone", "gk_off_line", "n_att_box", "n_def_box", "n_opp_visible", "has_gk"]
TAG = ["one_on_one", "open_goal"]
# SB 프리즈프레임은 화면에 보인 선수만, DFL 트래킹은 22명 전원 -> 화면 밖 선수까지 세는 피처는 두 데이터에서 뜻이 달라 뺀다
FF_COMMON = [c for c in FF_NUM if c not in ("n_opp_visible", "has_gk", "n_att_box", "n_def_box")]
COMMON_NUM = ["dist", "angle", "y_off", "inside_box", "is_head", "is_fk"] + FF_COMMON

VARIANTS = {
    "A": ("위치+부위", BASE_NUM, BASE_CAT),
    "B": ("+어시스트·포제션", BASE_NUM + EVT_NUM, BASE_CAT + EVT_CAT),
    "C": ("+프리즈프레임", BASE_NUM + EVT_NUM + FF_NUM, BASE_CAT + EVT_CAT),
    "D": ("+분석관 태그", BASE_NUM + EVT_NUM + FF_NUM + TAG, BASE_CAT + EVT_CAT),
    "P": ("공통 피처", COMMON_NUM, []),
}
SELECTABLE = ["A", "B", "C", "P"]


def design(d, num, cat, fit=None, min_count=30):
    """숫자형 + 원핫. fit 행에서 30회 미만인 범주는 'Other'로 합침 (라벨 정보는 쓰지 않음)."""
    fit = np.ones(len(d), dtype=bool) if fit is None else fit
    X = d[num].astype(float).copy()
    X["log_dist"] = np.log(d["dist"].clip(lower=0.5))
    for c in cat:
        v = d[c].fillna("None").astype(str)
        v = v.where(v.map(v[fit].value_counts()).fillna(0) >= min_count, "Other")
        X = X.join(pd.get_dummies(v, prefix=c, dtype=float))
    return X


def models():
    return {
        "LogReg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=1.0, max_iter=3000)),
        "GBM": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=40, random_state=SEED),
    }


def split_by_date(match_ids, frac=TEST_FRAC):
    """시즌 마지막 frac 비율의 경기를 test로 표시 (날짜·킥오프 순). 반환: test 마스크, test 시작일."""
    with gzip.open(os.path.join(SB_DIR, "matches.json.gz"), "rb") as fh:
        ms = json.loads(fh.read().decode())
    order = sorted(ms, key=lambda m: (m["match_date"], m.get("kick_off") or "", m["match_id"]))
    start = len(order) - round(len(order) * frac)
    test_ids = [m["match_id"] for m in order[start:]]
    return np.isin(match_ids, test_ids), order[start]["match_date"]


def cluster_boot(y, preds, groups, ref=None, n=1000, seed=SEED):
    """경기 단위 부트스트랩: AUC 95% CI, 그리고 ref 대비 AUC 차이의 95% CI."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    a, dlt = [], []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if y[ii].min() == y[ii].max():
            continue
        s = roc_auc_score(y[ii], preds[ii]); a.append(s)
        if ref is not None:
            dlt.append(s - roc_auc_score(y[ii], ref[ii]))
    ci = np.percentile(a, [2.5, 97.5])
    dci = np.percentile(dlt, [2.5, 97.5]) if dlt else (np.nan, np.nan)
    return ci, dci


def scores(y, p, g, ref):
    """AUC(+경기 부트스트랩 CI), statsbomb_xg 대비 AUC 차이(+CI), logloss, Brier, 예측 골 합/실제 골."""
    ci, dci = cluster_boot(y, p, g, ref)
    return dict(auc=roc_auc_score(y, p), auc_lo=ci[0], auc_hi=ci[1],
                d_vs_sb=roc_auc_score(y, p) - roc_auc_score(y, ref), d_lo=dci[0], d_hi=dci[1],
                logloss=log_loss(y, p), brier=brier_score_loss(y, p), calib=p.sum() / y.sum())


def main():
    t0 = time.time()
    d = pd.read_csv(os.path.join(OUT, "ds_statsbomb.csv"))
    d = d[d.shot_type != "Penalty"].reset_index(drop=True)
    y, g, ref = d.goal.values, d.match_id.values, d.ref_xg.values
    test, cut = split_by_date(g)
    di, ti = np.flatnonzero(~test), np.flatnonzero(test)
    folds = [(di[a], di[b]) for a, b in GroupKFold(n_splits=5).split(di, groups=g[di])]
    print(f"비페널티 {len(d):,}슛 · {y.sum()}골")
    print(f"  dev  {len(np.unique(g[di]))}경기 {len(di):,}슛 {y[di].sum()}골  -> 경기 단위 GroupKFold(5)로 비교")
    print(f"  test {len(np.unique(g[ti]))}경기 {len(ti):,}슛 {y[ti].sum()}골  ({cut}부터, 마지막에 한 번만 평가)\n",
          flush=True)

    oof = pd.DataFrame({"shot_id": d.shot_id, "match_id": g, "goal": y, "ref_xg": ref,
                        "split": np.where(test, "test", "dev")})
    res = []
    for key, (lab, num, cat) in VARIANTS.items():
        X = design(d, num, cat, fit=~test).values
        for mname, mk in models().items():
            p = np.zeros(len(y))
            for tr, te in folds:                                       # dev 안에서 교차검증
                p[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            p[ti] = mk().fit(X[di], y[di]).predict_proba(X[ti])[:, 1]  # dev 전체로 학습 -> test 한 번
            oof[f"{key}_{mname}"] = p
            r = dict(variant=key, label=lab, model=mname, n_feat=X.shape[1],
                     **scores(y[di], p[di], g[di], ref[di]),
                     **{f"test_{k}": v for k, v in scores(y[ti], p[ti], g[ti], ref[ti]).items()})
            res.append(r)
            print(f"  {key} {lab:16s} {mname:6s} 피처{r['n_feat']:3d}  "
                  f"dev CV AUC {r['auc']:.4f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]  |  "
                  f"test AUC {r['test_auc']:.4f} [{r['test_auc_lo']:.3f}, {r['test_auc_hi']:.3f}]  "
                  f"vs SB {r['test_d_vs_sb']:+.4f}  Σp/Σgoal {r['test_calib']:.3f}", flush=True)

    res.append(dict(variant="REF", label="statsbomb_xg", model="-", n_feat=np.nan,
                    **scores(y[di], ref[di], g[di], ref[di]),
                    **{f"test_{k}": v for k, v in scores(y[ti], ref[ti], g[ti], ref[ti]).items()}))
    r = res[-1]
    print(f"\n  REF statsbomb_xg                      dev AUC {r['auc']:.4f}  |  "
          f"test AUC {r['test_auc']:.4f} [{r['test_auc_lo']:.3f}, {r['test_auc_hi']:.3f}]  "
          f"Σp/Σgoal {r['test_calib']:.3f}")

    R = pd.DataFrame(res)
    cand = R[R.variant.isin(SELECTABLE)]
    best = cand.loc[cand.auc.idxmax()]
    R["selected"] = R.index == best.name
    print(f"\n최종 모델 = dev CV AUC가 가장 높은 {best.variant} {best.model} ({'/'.join(SELECTABLE)} 중)")
    print(f"  dev CV AUC {best.auc:.4f}  ->  최종 test AUC {best.test_auc:.4f} "
          f"[{best.test_auc_lo:.3f}, {best.test_auc_hi:.3f}]  logloss {best.test_logloss:.4f}  "
          f"Σp/Σgoal {best.test_calib:.3f}")

    R.to_csv(os.path.join(OUT, "v1_results.csv"), index=False)
    oof.to_csv(os.path.join(OUT, "v1_oof.csv"), index=False)
    print(f"\n-> out/v1_results.csv, out/v1_oof.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
