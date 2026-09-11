import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score, log_loss
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
F=pd.read_csv(os.path.join(OUT,"features.csv"))
num=["dist","angle","y_off","n_def_cone","n_def_3m","d_def_min","gk_dist_goal","gk_dist_shot",
     "gk_off_line","shooter_speed","ball_z","ball_speed","n_att_box","n_def_box"]
cat=["TypeOfShot","TakerBallControl","BuildUp","InsideBox","ShotCondition"]
D=pd.get_dummies(F[num+cat].fillna(0),columns=cat,drop_first=True).astype(float)
t=np.log(F.xG/(1-F.xG)); grp=F.MatchId.values
mdl=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(D.values,t)
print("="*78);print("9. WHAT DRIVES DFL's xG?  (standardised Ridge coefs on logit-xG, R2 in-sample %.3f)"%r2_score(t,mdl.predict(D.values)));print("="*78)
co=pd.Series(mdl[-1].coef_, index=D.columns).sort_values(key=abs, ascending=False)
print(co.head(16).round(3).to_string())

print("\n"+"="*78);print("10. HOW MUCH DATA WOULD A SELF-TRAINED BINARY MODEL NEED?");print("="*78)
print("  Rule of thumb: ~10-20 events per predictor (EPV) for a stable logistic fit.")
print(f"  You have {int(F.goal.sum())} goals in the tracking-linked sample ({len(F)} shots).")
for k in [2,5,10,20]:
    print(f"    {k:2d} predictors -> need {10*k:3d}-{20*k:3d} goals  ~= {int(10*k/0.111):4d}-{int(20*k/0.111):5d} shots "
          f"~= {10*k/0.111/24.4:5.0f}-{20*k/0.111/24.4:5.0f} matches")
print(f"\n  This dataset: {len(F)/7:.1f} shots/match, {F.goal.sum()/7:.1f} goals/match.")
print("  A public-standard xG model is fit on 10k-100k+ shots. You have 171.")

# bootstrap CI on the base rate to show how wide everything is
rng=np.random.default_rng(0); bs=[rng.choice(F.goal.values,len(F),replace=True).mean() for _ in range(5000)]
print(f"  95% CI on the overall conversion rate alone: [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}] (point {F.goal.mean():.3f})")

print("\n"+"="*78);print("11. DOES DISTILLATION BEAT DIRECT TRAINING ON THE ACTUAL GOALS?");print("="*78)
gkf=GroupKFold(n_splits=7)
pr=cross_val_predict(make_pipeline(StandardScaler(),Ridge()),D.values,t,groups=grp,cv=gkf)
xg_distil=1/(1+np.exp(-pr))
pr_dir=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=0.3)),D.values,F.goal.values,groups=grp,cv=gkf,method="predict_proba")[:,1]
y=F.goal.values
for nm,p in [("distilled (target=DFL xG)",xg_distil),("direct   (target=goal)",pr_dir),("DFL shipped xG",F.xG.values)]:
    print(f"  {nm:26s} AUC={roc_auc_score(y,p):.4f}  logloss={log_loss(y,p):.4f}  sum={p.sum():5.2f} vs {y.sum()} goals")
