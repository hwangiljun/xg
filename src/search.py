"""LOMO CV 하에서 정직하게 도달 가능한 최대 AUC를 탐욕적 전진선택으로 추정."""
import os,warnings,numpy as np,pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import LeaveOneGroupOut,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
F=pd.read_csv(f"{OUT}/features.csv"); P=pd.read_csv(f"{OUT}/features_plus.csv")
D=F.merge(P.drop(columns=["MatchId","goal","xG","ball_z"]),left_on="ev.EventId",right_on="shot_id")
y=D.goal.values; grp=D.MatchId.values; logo=LeaveOneGroupOut()
t=np.log(D.xG/(1-D.xG))

POOL=["DistanceToGoal","AngleToGoal","open_angle","open_frac","open_frac_nogk","gk_block_share",
      "n_def_cone","d_def_min","gk_dist_goal","gk_off_line","gk_dist_shot","shooter_speed",
      "ball_z","ball_speed","n_att_box","n_def_box","def_closing","orient","n_corridor",
      "d_corridor","y_off","score_diff","minute","Pressure","AmountOfDefenders",
      "GoalDistanceGoalkeeper","is_head","is_foot_strong","inside_box"]
D["is_head"]=(D.TypeOfShot=="head").astype(int)
D["is_foot_strong"]=(D.TakerBallControl.isin(["direct","volley"])).astype(int)
D["inside_box"]=D.InsideBox.astype(float)
X=D[POOL].astype(float); X=X.fillna(X.median())

def cv_auc(cols,mode):
    Z=X[cols].values
    if mode=="direct":
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
        p=cross_val_predict(m,Z,y,groups=grp,cv=logo,method="predict_proba")[:,1]
    else:
        m=make_pipeline(StandardScaler(),Ridge(alpha=2.0))
        p=1/(1+np.exp(-cross_val_predict(m,Z,t,groups=grp,cv=logo)))
    return roc_auc_score(y,p)

for mode in ["direct","distill"]:
    sel=[];best=0.5;hist=[]
    while len(sel)<8:
        cand=[(cv_auc(sel+[c],mode),c) for c in POOL if c not in sel]
        a,c=max(cand)
        if a<=best+1e-4: break
        sel.append(c); best=a; hist.append((c,a))
    lab="직접 학습 (타깃=골)" if mode=="direct" else "증류 (타깃=DFL xG)"
    print(f"\n=== {lab} · LOMO 전진선택 ===")
    for i,(c,a) in enumerate(hist,1): print(f"  {i}. +{c:24s} AUC={a:.4f}")
    print(f"  최종 {len(sel)}개 피처 → AUC={best:.4f}")

print(f"\n참조: DFL 제공 xG = {roc_auc_score(y,D.xG):.4f} · 목표 = 0.8300")
print("주의: 전진선택은 CV 점수 자체를 최적화하므로 위 숫자도 낙관 편향(선택 편향)이 있습니다.")
