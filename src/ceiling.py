import os,warnings,numpy as np,pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold,cross_val_predict,StratifiedKFold,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,log_loss
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out")
F=pd.read_csv(f"{OUT}/features.csv"); P=pd.read_csv(f"{OUT}/features_plus.csv")
S=pd.read_csv(f"{OUT}/shots.csv")
D=F.merge(P.drop(columns=["MatchId","goal","xG","ball_z"]),left_on="ev.EventId",right_on="shot_id",how="left")
D=D.merge(S[["ev.EventId","SetupOrigin","AssistTypeShotAtGoal"]].rename(
    columns={"SetupOrigin":"SO2","AssistTypeShotAtGoal":"AT2"}),on="ev.EventId",how="left")
y=D.goal.values; grp=D.MatchId.values

EV_NUM=["DistanceToGoal","AngleToGoal","score_diff","minute","ShotOrigin"]
EV_CAT=["TypeOfShot","TakerBallControl","BuildUp","InsideBox","CounterAttack","ShotCondition","SO2","AT2"]
TR_NUM=["open_angle","open_frac","open_frac_nogk","gk_block_share","n_def_cone","d_def_min",
        "gk_dist_goal","gk_off_line","gk_dist_shot","shooter_speed","ball_z","ball_speed",
        "n_att_box","n_def_box","def_closing","orient","n_corridor","d_corridor","y_off"]

def mat(num,cat):
    X=pd.get_dummies(D[num+cat],columns=cat,dummy_na=False,drop_first=True)
    return X.fillna(X.median(numeric_only=True)).fillna(0).astype(float).values

def ev_auc(p): return roc_auc_score(y,p)
def run(name,X,protocol):
    cvr=None
    if protocol=="LOMO": cv=LeaveOneGroupOut(); g=grp
    elif protocol=="Group5": cv=GroupKFold(n_splits=5); g=grp
    else: cv=StratifiedKFold(5,shuffle=True,random_state=0); cvr=KFold(5,shuffle=True,random_state=0); g=None
    lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.25,max_iter=2000))
    p=cross_val_predict(lr,X,y,groups=g,cv=cv,method="predict_proba")[:,1]
    # 증류: DFL xG를 logit으로 회귀 -> 골에 대한 AUC 평가
    t=np.log(D.xG/(1-D.xG))
    rg=make_pipeline(StandardScaler(),Ridge(alpha=3.0))
    pr=cross_val_predict(rg,X,t,groups=g,cv=(cvr or cv)); pd_=1/(1+np.exp(-pr))
    # 하이브리드: 직접 + 증류 평균(rank)
    from scipy.stats import rankdata
    hy=(rankdata(p)+rankdata(pd_))/2/len(y)
    print(f"  {name:26s} 직접={ev_auc(p):.4f}  증류={ev_auc(pd_):.4f}  하이브리드={ev_auc(hy):.4f}")
    return p,pd_,hy

print("="*76);print("A. 평가 프로토콜별 AUC — 같은 피처, 같은 모델");print("="*76)
Xev=mat(EV_NUM,EV_CAT); Xtr=mat(EV_NUM+TR_NUM,EV_CAT)
for proto in ["LOMO","Group5","Random5"]:
    print(f"\n[{proto}]  " + ("경기 누출 없음" if proto!="Random5" else "!! 같은 경기가 train/test 양쪽 -> 낙관 편향"))
    run("v1 이벤트만",Xev,proto)
    run("v2 +트래킹",Xtr,proto)

print("\n"+"="*76);print("B. In-sample (과적합 확인용 — 보고하면 안 되는 숫자)");print("="*76)
for nm,X in [("v1 이벤트만",Xev),("v2 +트래킹",Xtr)]:
    lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.25,max_iter=2000)).fit(X,y)
    rf=RandomForestClassifier(n_estimators=400,random_state=0).fit(X,y)
    print(f"  {nm:26s} LogReg={ev_auc(lr.predict_proba(X)[:,1]):.4f}  RandomForest={ev_auc(rf.predict_proba(X)[:,1]):.4f}")

print("\n"+"="*76);print("C. 참조점");print("="*76)
print(f"  DFL 제공 xG (해당 159슛)          AUC={ev_auc(D.xG.values):.4f}")
print(f"  목표                              AUC=0.8300")
