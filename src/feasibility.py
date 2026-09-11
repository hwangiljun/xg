import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, r2_score
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
F=pd.read_csv(os.path.join(OUT,"features.csv")); s=pd.read_csv(os.path.join(OUT,"shots.csv"))

print("="*78);print("6. CAN YOU TRAIN A BINARY xG MODEL ON THIS? (leave-one-match-out CV)");print("="*78)
X=F[["dist","angle"]].values; y=F.goal.values; grp=F.MatchId.values
logo=LeaveOneGroupOut()
pipe=make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
p=cross_val_predict(pipe,X,y,groups=grp,cv=logo,method="predict_proba")[:,1]
print(f"  own model (dist+angle) : AUC={roc_auc_score(y,p):.4f}  logloss={log_loss(y,p):.4f}  Brier={brier_score_loss(y,p):.4f}")
X2=F[["dist","angle","n_def_cone","d_def_min","gk_dist_goal","gk_off_line","shooter_speed","ball_z"]].fillna(0).values
p2=cross_val_predict(pipe,X2,y,groups=grp,cv=logo,method="predict_proba")[:,1]
print(f"  own model (+ tracking) : AUC={roc_auc_score(y,p2):.4f}  logloss={log_loss(y,p2):.4f}  Brier={brier_score_loss(y,p2):.4f}")
print(f"  DFL shipped xG         : AUC={roc_auc_score(y,F.xG):.4f}  logloss={log_loss(y,F.xG):.4f}  Brier={brier_score_loss(y,F.xG):.4f}")
print(f"  constant base rate     : AUC=0.5000  logloss={log_loss(y,np.full(len(y),y.mean())):.4f}  Brier={brier_score_loss(y,np.full(len(y),y.mean())):.4f}")

# stability of the fitted coefficients across folds = how unstable a 159-shot fit is
co=[]
for tr,te in logo.split(X,y,grp):
    m=make_pipeline(StandardScaler(),LogisticRegression()).fit(X[tr],y[tr])
    co.append(m[-1].coef_[0])
co=np.array(co)
print(f"\n  coef stability over 7 folds  dist: {co[:,0].mean():+.3f} +- {co[:,0].std():.3f}"
      f"   angle: {co[:,1].mean():+.3f} +- {co[:,1].std():.3f}")
print(f"  -> per-fold dist coef range [{co[:,0].min():+.3f}, {co[:,0].max():+.3f}]")

print("\n"+"="*78);print("7. ALTERNATIVE TARGET: regress DFL's xG (continuous -> 171 usable samples)");print("="*78)
num=["dist","angle","y_off","n_def_cone","n_def_3m","d_def_min","gk_dist_goal","gk_dist_shot",
     "gk_off_line","shooter_speed","ball_z","ball_speed","n_att_box","n_def_box"]
cat=["TypeOfShot","TakerBallControl","BuildUp","InsideBox","ShotCondition"]
D=pd.get_dummies(F[num+cat].fillna(0),columns=cat,drop_first=True).astype(float)
t=np.log(F.xG/(1-F.xG))                       # model xG on the logit scale
gkf=GroupKFold(n_splits=7)
for name,mdl in [("Ridge",make_pipeline(StandardScaler(),Ridge(alpha=1.0))),
                 ("GBM",GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0))]:
    pr=cross_val_predict(mdl,D.values,t,groups=grp,cv=gkf)
    xg_hat=1/(1+np.exp(-pr))
    print(f"  {name:6s} logit-xG R2={r2_score(t,pr):.3f}   recovered xG: corr={np.corrcoef(xg_hat,F.xG)[0,1]:.4f} MAE={np.abs(xg_hat-F.xG).mean():.4f}")
# geometry only
D2=F[["dist","angle"]].values
pr=cross_val_predict(make_pipeline(StandardScaler(),Ridge()),D2,t,groups=grp,cv=gkf)
print(f"  geometry-only Ridge  logit-xG R2={r2_score(t,pr):.3f}")

print("\n"+"="*78);print("8. LEAKAGE AUDIT - columns that encode the outcome");print("="*78)
suspect={"ChanceEvaluation":"post-hoc human grade of the chance","SignificanceEvaluation":"post-hoc grade",
         "outcome":"IS the target","out.*":"outcome-child attributes (GoalZone, SaveType, ...)",
         "ShotContribution":"only populated on notable chances","SitterContribution":"only on sitters",
         "GoalDistanceGoalkeeper":"OK - measured at shot instant","Pressure":"OK - at shot instant"}
for k,v in suspect.items(): print(f"  {k:26s} {v}")
print("\n  proof: ChanceEvaluation alone")
print(s.groupby("ChanceEvaluation").agg(n=("goal","size"),conv=("goal","mean")).round(3).to_string())
print(f"  AUC of ChanceEvaluation=='sitter' alone: {roc_auc_score(s.goal,(s.ChanceEvaluation=='sitter').astype(int)):.4f}")
