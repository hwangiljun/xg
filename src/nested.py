"""중첩 CV: 바깥 LOMO, 안쪽에서만 피처 선택 -> 선택 편향이 제거된 정직한 AUC."""
import os,warnings,numpy as np,pandas as pd,time
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
F=pd.read_csv(f"{OUT}/features.csv"); P=pd.read_csv(f"{OUT}/features_plus.csv")
D=F.merge(P.drop(columns=["MatchId","goal","xG","ball_z"]),left_on="ev.EventId",right_on="shot_id")
D["is_head"]=(D.TypeOfShot=="head").astype(int)
D["inside_box"]=D.InsideBox.astype(float)
POOL=["DistanceToGoal","AngleToGoal","open_angle","open_frac","open_frac_nogk","gk_block_share",
      "n_def_cone","d_def_min","gk_dist_goal","gk_off_line","gk_dist_shot","shooter_speed",
      "ball_z","ball_speed","n_att_box","n_def_box","def_closing","orient","n_corridor",
      "d_corridor","y_off","score_diff","minute","Pressure","AmountOfDefenders",
      "GoalDistanceGoalkeeper","is_head","inside_box"]
X=D[POOL].astype(float); X=X.fillna(X.median())
y=D.goal.values; grp=D.MatchId.values; t=np.log(D.xG/(1-D.xG)).values
MAXF=5

def fit_pred(cols,tr,te,mode):
    Ztr,Zte=X.iloc[tr][cols].values,X.iloc[te][cols].values
    if mode=="direct":
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)).fit(Ztr,y[tr])
        return m.predict_proba(Zte)[:,1]
    m=make_pipeline(StandardScaler(),Ridge(alpha=2.0)).fit(Ztr,t[tr])
    return 1/(1+np.exp(-m.predict(Zte)))

def inner_select(tr,mode):
    gi=grp[tr]; sel=[];best=0.5
    inner=list(GroupKFold(n_splits=min(4,len(set(gi)))).split(X.iloc[tr],y[tr],gi))
    while len(sel)<MAXF:
        scores=[]
        for c in POOL:
            if c in sel: continue
            p=np.zeros(len(tr))
            for a,b in inner: p[b]=fit_pred(sel+[c],tr[a],tr[b],mode)
            try: scores.append((roc_auc_score(y[tr],p),c))
            except ValueError: pass
        if not scores: break
        a,c=max(scores)
        if a<=best+1e-4: break
        sel.append(c); best=a
    return sel

for mode in ["direct","distill"]:
    t0=time.time(); oof=np.zeros(len(y)); picks=[]
    for tr,te in LeaveOneGroupOut().split(X,y,grp):
        sel=inner_select(tr,mode); picks.append(sel)
        oof[te]=fit_pred(sel,tr,te,mode)
    lab="직접 학습 (타깃=골)" if mode=="direct" else "증류 (타깃=DFL xG)"
    print(f"\n=== {lab} · 중첩 CV ===  ({time.time()-t0:.0f}s)")
    print(f"  정직한 AUC = {roc_auc_score(y,oof):.4f}")
    from collections import Counter
    cnt=Counter([c for s in picks for c in s])
    print("  폴드별 선택 빈도:", dict(cnt.most_common(8)))

print(f"\nDFL 제공 xG = {roc_auc_score(y,D.xG):.4f}  ·  목표 = 0.8300")
