"""Build xG features: DFL event attributes + freeze-frame geometry from 25Hz tracking."""
import os
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","out")
GOAL_X, GOAL_Y, POST = 105.0, 34.0, 3.66     # event coords, attacking goal normalised to x=105

def load():
    s  = pd.read_csv(os.path.join(OUT,"shots.csv"))
    ff = pd.read_parquet(os.path.join(OUT,"freeze_frames.parquet"))
    pl = pd.read_csv(os.path.join(OUT,"players.csv"))
    ff["x"]=ff.X+52.5; ff["y"]=ff.Y+34.0          # tracking(centre-origin) -> event(corner-origin)
    return s, ff, pl

def subtended_angle(x,y):
    dx=GOAL_X-x; dy=y-GOAL_Y
    return np.degrees(np.abs(np.arctan2(POST-dy,dx)-np.arctan2(-POST-dy,dx)))

def in_cone(px,py,sx,sy):
    """point (px,py) inside triangle shooter(sx,sy)-post(105,34+-3.66)"""
    def sign(ax,ay,bx,by,cx,cy): return (ax-cx)*(by-cy)-(bx-cx)*(ay-cy)
    ax,ay=sx,sy; bx,by=GOAL_X,GOAL_Y+POST; cx,cy=GOAL_X,GOAL_Y-POST
    d1=sign(px,py,ax,ay,bx,by); d2=sign(px,py,bx,by,cx,cy); d3=sign(px,py,cx,cy,ax,ay)
    neg=(d1<0)|(d2<0)|(d3<0); pos=(d1>0)|(d2>0)|(d3>0)
    return ~(neg&pos)

def build():
    s, ff, pl = load()
    gks = set(pl.loc[pl.PlayingPosition=="TW","PersonId"])
    s = s[s["ev.CalculatedFrame"].notna()].copy()
    s["N"]=s["ev.CalculatedFrame"].astype(int)
    ffi = {(m,n): g for (m,n),g in ff.groupby(["MatchId","N"])}
    rows=[]
    for _,sh in s.iterrows():
        g = ffi.get((sh["ev.MatchId"], sh["N"]))
        if g is None: continue
        sx0,sy0 = sh["ev.X-PositionFromTracking"], sh["ev.Y-PositionFromTracking"]
        flip = abs(sx0-105) > abs(sx0-0)          # attacking the x=0 goal -> mirror
        def nx(v): return 105-v if flip else v
        def ny(v): return 68-v if flip else v
        sx, sy = nx(sx0), ny(sy0)
        pls = g[(g.TeamId!="BALL")&(g.TeamId!="referee")].copy()
        pls["px"]=nx(pls.x); pls["py"]=ny(pls.y)
        att = pls[pls.TeamId==sh["Team"]]
        dfd = pls[(pls.TeamId!=sh["Team"])]
        dfd = dfd[dfd.PersonId!=sh["Player"]]
        gk  = dfd[dfd.PersonId.isin(gks)]
        out_ = dfd[~dfd.PersonId.isin(gks)]        # outfield defenders
        ball= g[g.TeamId=="BALL"]

        d_def = np.hypot(out_.px-sx, out_.py-sy)
        cone  = in_cone(out_.px.values, out_.py.values, sx, sy)
        # defenders in cone AND goal-side of the shooter
        goalside = out_.px.values > sx
        r={"shot_id":sh["ev.EventId"], "MatchId":sh["ev.MatchId"], "goal":sh["goal"],
           "x":sx, "y":sy, "dist":np.hypot(GOAL_X-sx, sy-GOAL_Y), "angle":subtended_angle(sx,sy),
           "y_off":abs(sy-GOAL_Y),
           "n_def_cone":int((cone&goalside).sum()),
           "n_def_3m":int((d_def<3).sum()), "n_def_5m":int((d_def<5).sum()),
           "d_def_min":float(d_def.min()) if len(d_def) else np.nan,
           "n_att_box":int(((att.px>88.5)&(att.py.between(13.84,54.16))).sum()),
           "n_def_box":int(((out_.px>88.5)&(out_.py.between(13.84,54.16))).sum()),
           "shooter_speed":float(pls.loc[pls.PersonId==sh["Player"],"S"].iloc[0]) if (pls.PersonId==sh["Player"]).any() else np.nan,
           "ball_z":float(ball.Z.iloc[0]) if len(ball) else np.nan,
           "ball_speed":float(ball.S.iloc[0]) if len(ball) else np.nan,
           }
        if len(gk):
            gx,gy=float(gk.px.iloc[0]),float(gk.py.iloc[0])
            r.update({"gk_dist_goal":np.hypot(GOAL_X-gx, gy-GOAL_Y),
                      "gk_dist_shot":np.hypot(gx-sx, gy-sy),
                      "gk_in_cone":int(in_cone(np.array([gx]),np.array([gy]),sx,sy)[0]),
                      # perpendicular offset of GK from the shooter->goal-centre line
                      "gk_off_line":abs((GOAL_X-sx)*(sy-gy)-(sx-gx)*(GOAL_Y-sy))/max(np.hypot(GOAL_X-sx,GOAL_Y-sy),1e-9)})
        else:
            r.update({"gk_dist_goal":np.nan,"gk_dist_shot":np.nan,"gk_in_cone":np.nan,"gk_off_line":np.nan})
        rows.append(r)
    F=pd.DataFrame(rows)
    keep=["ev.EventId","ev.MatchId","setpiece_parent","outcome","xG","DistanceToGoal","AngleToGoal",
          "Pressure","GoalDistanceGoalkeeper","AmountOfDefenders","PlayerSpeed","TypeOfShot","TakerBallControl",
          "TakerSetup","BuildUp","ShotOrigin","InsideBox","CounterAttack","ChanceEvaluation","ShotCondition",
          "AssistTypeShotAtGoal","SetupOrigin","ExtendedTypeOfShot","Player","Team"]
    F=F.merge(s[[c for c in keep if c in s.columns]], left_on="shot_id", right_on="ev.EventId", how="left")
    return F

if __name__=="__main__":
    F=build(); F.to_csv(os.path.join(OUT,"features.csv"),index=False)
    print("shots with freeze-frame features:",len(F),"goals:",F.goal.sum())
    print("\n--- validation vs DFL's own columns ---")
    print(f"  dist  vs DistanceToGoal        corr={F.dist.corr(F.DistanceToGoal):.5f}  MAE={(F.dist-F.DistanceToGoal).abs().mean():.4f}")
    print(f"  angle vs AngleToGoal           corr={F.angle.corr(F.AngleToGoal):.5f}  MAE={(F.angle-F.AngleToGoal).abs().mean():.4f}")
    print(f"  gk_dist_goal vs GoalDistanceGoalkeeper corr={F.gk_dist_goal.corr(F.GoalDistanceGoalkeeper):.4f} MAE={(F.gk_dist_goal-F.GoalDistanceGoalkeeper).abs().mean():.3f}")
    print(f"  n_def_cone vs AmountOfDefenders corr={F.n_def_cone.corr(F.AmountOfDefenders):.4f}")
    print(f"  shooter_speed vs PlayerSpeed    corr={F.shooter_speed.corr(F.PlayerSpeed):.4f} MAE={(F.shooter_speed-F.PlayerSpeed).abs().mean():.3f}")
    print(f"  d_def_min vs Pressure           corr={F.d_def_min.corr(F.Pressure):.4f}")
