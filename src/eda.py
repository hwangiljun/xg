import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
pd.set_option("display.width",200)
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
s=pd.read_csv(os.path.join(OUT,"shots.csv")); F=pd.read_csv(os.path.join(OUT,"features.csv"))

print("="*78); print("1. TARGET BALANCE")
print("="*78)
print(f"shots={len(s)}  goals={s.goal.sum()}  base rate={s.goal.mean():.4f}")
print(f"open play: n={(s.setpiece_parent=='openPlay').sum():3d} goals={s[s.setpiece_parent=='openPlay'].goal.sum()}")
print(f"free kick: n={(s.setpiece_parent=='FreeKick').sum():3d} goals={s[s.setpiece_parent=='FreeKick'].goal.sum()}")
print(f"penalty  : n={(s.setpiece_parent=='Penalty').sum():3d} goals={s[s.setpiece_parent=='Penalty'].goal.sum()}  (DFL xG fixed at {s[s.setpiece_parent=='Penalty'].xG.unique()})")
print("\nBlocked shots are in the sample:", (s.outcome=='BlockedShot').sum(),
      "-> decide whether to model them (they can never be goals here: goals among blocked =",s[s.outcome=='BlockedShot'].goal.sum(),")")

print("\n"+"="*78); print("2. HOW GOOD IS DFL's SHIPPED xG?  (it is a ready-made benchmark AND a soft label)")
print("="*78)
y=s.goal.values; p=s.xG.values
print(f"  sum(xG)={p.sum():.2f}  vs actual goals={y.sum()}   (bias {p.sum()-y.sum():+.2f})")
print(f"  ROC AUC   = {roc_auc_score(y,p):.4f}")
print(f"  log loss  = {log_loss(y,p):.4f}   (base-rate-only model = {log_loss(y,np.full_like(p,y.mean())):.4f})")
print(f"  Brier     = {brier_score_loss(y,p):.4f}   (base rate = {brier_score_loss(y,np.full_like(p,y.mean())):.4f})")
op=s[s.setpiece_parent=='openPlay']
print(f"  open-play only: AUC={roc_auc_score(op.goal,op.xG):.4f}  sum(xG)={op.xG.sum():.2f} vs {op.goal.sum()} goals")
print("\n  calibration by xG decile:")
q=pd.qcut(s.xG,5,duplicates='drop')
print(s.groupby(q,observed=True).agg(n=('goal','size'),goals=('goal','sum'),mean_xG=('xG','mean'),actual=('goal','mean')).round(3).to_string())

print("\n"+"="*78); print("3. SIGNAL IN THE GEOMETRY")
print("="*78)
b=pd.cut(s.DistanceToGoal,[0,6,11,16,22,30,60])
print(s.groupby(b,observed=True).agg(n=('goal','size'),goals=('goal','sum'),conv=('goal','mean'),mean_xG=('xG','mean')).round(3).to_string())
print()
b2=pd.cut(s.AngleToGoal,[0,10,20,30,40,90])
print(s.groupby(b2,observed=True).agg(n=('goal','size'),goals=('goal','sum'),conv=('goal','mean'),mean_xG=('xG','mean')).round(3).to_string())

print("\n"+"="*78); print("4. CATEGORICAL FEATURES: cell counts are the problem")
print("="*78)
for c in ["TypeOfShot","TakerBallControl","BuildUp","TakerSetup","ShotCondition","ChanceEvaluation","InsideBox","CounterAttack"]:
    g=s.groupby(c,observed=True).agg(n=('goal','size'),goals=('goal','sum'),conv=('goal','mean')).sort_values('n',ascending=False)
    print(f"\n[{c}]  levels={len(g)}  levels with 0 goals={(g.goals==0).sum()}  levels with n<5={(g.n<5).sum()}")
    print(g.head(6).round(3).to_string())

print("\n"+"="*78); print("5. TRACKING-DERIVED FEATURES (n=159) vs goal")
print("="*78)
cols=["dist","angle","y_off","n_def_cone","n_def_3m","n_def_5m","d_def_min","n_att_box","n_def_box",
      "shooter_speed","ball_z","ball_speed","gk_dist_goal","gk_dist_shot","gk_in_cone","gk_off_line"]
r=[]
for c in cols:
    v=F[c].astype(float)
    m=v.notna()
    r.append({"feature":c,"missing":int((~m).sum()),
              "AUC_alone":roc_auc_score(F.goal[m],v[m]) if F.goal[m].nunique()>1 else np.nan,
              "mean_goal":v[F.goal==1].mean(),"mean_nogoal":v[F.goal==0].mean(),
              "corr_DFLxG":v.corr(F.xG)})
R=pd.DataFrame(r); R["AUC_abs"]=(R.AUC_alone-0.5).abs()+0.5
print(R.sort_values("AUC_abs",ascending=False).round(3).to_string(index=False))
