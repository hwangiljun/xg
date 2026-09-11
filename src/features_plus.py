"""창의적 피처: 골문 가시율(ray-casting), 수비수 접근속도, 슈터 방향, 시퀀스/스코어 상태."""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","out")
GX,GY,POST=105.0,34.0,3.66

def open_goal_fraction(sx,sy,bx,by,r=0.42,n=121):
    """슈터에서 골문 폭으로 121개 광선을 쏴, 선수 반경 r 안을 지나면 차단된 것으로 본다."""
    if len(bx)==0: return 1.0
    ty=np.linspace(GY-POST+0.15, GY+POST-0.15, n)
    dx=np.full_like(ty, GX-sx); dy=ty-sy      # (n,)
    wx=bx-sx; wy=by-sy                        # (m,)
    dd=dx*dx+dy*dy                            # (n,)
    t=(wx[None,:]*dx[:,None]+wy[None,:]*dy[:,None])/dd[:,None]
    t=np.clip(t,0,1)
    cx=sx+t*dx[:,None]; cy=sy+t*dy[:,None]    # (n,m)
    d=np.hypot(bx[None,:]-cx, by[None,:]-cy)
    return float((d.min(axis=1)>r).mean())

def build():
    ff=pd.read_parquet(os.path.join(OUT,"freeze_frames.parquet"))
    s =pd.read_csv(os.path.join(OUT,"shots.csv"))
    pl=pd.read_csv(os.path.join(OUT,"players.csv"))
    gks=set(pl.loc[pl.PlayingPosition=="TW","PersonId"])
    ff["x"]=ff.X+52.5; ff["y"]=ff.Y+34.0
    s=s[s["ev.CalculatedFrame"].notna()].copy(); s["N"]=s["ev.CalculatedFrame"].astype(int)

    # --- 스코어 상태 (골 이벤트로 러닝 스코어 복원) ---
    s["ev.EventTime"]=pd.to_datetime(s["ev.EventTime"],format="ISO8601",utc=True)
    s=s.sort_values(["ev.MatchId","ev.EventTime"])
    s["goal_f"]=(s.outcome=="SuccessfulShot").astype(int)
    gfor=[];gag=[]
    for mid,g in s.groupby("ev.MatchId",sort=False):
        tally={}
        for _,r in g.iterrows():
            t=r["Team"]; opp=[k for k in tally if k!=t]
            gfor.append(tally.get(t,0)); gag.append(tally.get(opp[0],0) if opp else 0)
            tally[t]=tally.get(t,0)+r["goal_f"]
    s["score_for"]=gfor; s["score_against"]=gag
    s["score_diff"]=s.score_for-s.score_against
    k=s.groupby("ev.MatchId")["ev.EventTime"].transform("min")
    s["minute"]=(s["ev.EventTime"]-k).dt.total_seconds()/60.0

    idx={(m,n):g for (m,n),g in ff.groupby(["MatchId","N"])}
    rows=[]
    for _,sh in s.iterrows():
        N=sh["N"]; mid=sh["ev.MatchId"]
        g=idx.get((mid,N)); gp=idx.get((mid,N-25))       # 1초 전
        if g is None: continue
        sx0,sy0=sh["ev.X-PositionFromTracking"],sh["ev.Y-PositionFromTracking"]
        flip=abs(sx0-105)>abs(sx0-0)
        nx=lambda v:105-v if flip else v; ny=lambda v:68-v if flip else v
        sx,sy=nx(sx0),ny(sy0)
        P=g[(g.TeamId!="BALL")&(g.TeamId!="referee")].copy()
        P["px"]=nx(P.x); P["py"]=ny(P.y)
        D=P[(P.TeamId!=sh["Team"])&(P.PersonId!=sh["Player"])]
        gk=D[D.PersonId.isin(gks)]; out=D[~D.PersonId.isin(gks)]
        ball=g[g.TeamId=="BALL"]

        # 1) 골문 가시율 — 수비수+GK 전원을 차단체로
        of_all=open_goal_fraction(sx,sy,D.px.values,D.py.values)
        of_nogk=open_goal_fraction(sx,sy,out.px.values,out.py.values)
        sub=np.degrees(abs(np.arctan2(POST-(sy-GY),GX-sx)-np.arctan2(-POST-(sy-GY),GX-sx)))

        # 2) 수비수 접근 속도 (1초간 슈터까지 거리 변화)
        clos=np.nan
        if gp is not None:
            Q=gp[(gp.TeamId!="BALL")&(gp.TeamId!="referee")].copy()
            Q["px"]=nx(Q.x); Q["py"]=ny(Q.y)
            sp=Q[Q.PersonId==sh["Player"]]
            if len(sp):
                spx,spy=float(sp.px.iloc[0]),float(sp.py.iloc[0])
                Qd=Q[(Q.TeamId!=sh["Team"])&(Q.PersonId!=sh["Player"])&(~Q.PersonId.isin(gks))]
                m=Qd.merge(out[["PersonId","px","py"]],on="PersonId",suffixes=("_prev",""))
                if len(m):
                    d_prev=np.hypot(m.px_prev-spx,m.py_prev-spy)
                    d_now =np.hypot(m.px-sx,m.py-sy)
                    o=np.argsort(d_now.values)[:3]
                    clos=float((d_prev.values[o]-d_now.values[o]).mean())   # +면 접근중

        # 3) 슈터 진행방향과 골 방향의 각도차
        orient=np.nan
        if gp is not None:
            sp=gp[gp.PersonId==sh["Player"]]
            if len(sp):
                vx=sx-nx(float(sp.x.iloc[0])); vy=sy-ny(float(sp.y.iloc[0]))
                if np.hypot(vx,vy)>0.3:
                    orient=float(abs(np.degrees(np.arctan2(GY-sy,GX-sx)-np.arctan2(vy,vx))))
                    orient=min(orient,360-orient)

        # 4) 슛 방향 전방 통로의 최근접 수비수
        v=np.array([GX-sx,GY-sy]); v/=np.linalg.norm(v)
        rel=np.c_[out.px-sx,out.py-sy]
        along=rel@v; perp=np.abs(rel[:,0]*(-v[1])+rel[:,1]*v[0])
        corr=(along>0)&(perp<2.0)
        rows.append({
          "shot_id":sh["ev.EventId"],"MatchId":mid,"goal":int(sh["goal_f"]),"xG":sh["xG"],
          "open_frac":of_all,"open_frac_nogk":of_nogk,"open_angle":of_all*sub,
          "gk_block_share":of_nogk-of_all,
          "def_closing":clos,"orient":orient,
          "n_corridor":int(corr.sum()),
          "d_corridor":float(along[corr].min()) if corr.any() else 40.0,
          "score_diff":sh["score_diff"],"minute":sh["minute"],
          "ball_z":float(ball.Z.iloc[0]) if len(ball) else np.nan,
        })
    return pd.DataFrame(rows)

if __name__=="__main__":
    P=build(); P.to_csv(os.path.join(OUT,"features_plus.csv"),index=False)
    from sklearn.metrics import roc_auc_score
    print(P.shape)
    for c in ["open_frac","open_frac_nogk","open_angle","gk_block_share","def_closing",
              "orient","n_corridor","d_corridor","score_diff","minute"]:
        v=P[c].astype(float); m=v.notna()
        print(f"  {c:16s} AUC={roc_auc_score(P.goal[m],v[m]):.3f}  "
              f"골={v[P.goal==1].mean():7.2f}  노골={v[P.goal==0].mean():7.2f}  결측={(~m).sum()}")
