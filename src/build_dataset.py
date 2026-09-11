"""DFL과 StatsBomb을 같은 피처 정의로 변환한다.

두 어댑터 모두 geometry.shot_features()를 통과하므로 피처 정의가 구조적으로 일치한다.
범주 이름이 다른 부위·세트피스는 공통 이진 피처 is_head / is_fk로 맞춘다.
DFL 쪽에만 트래킹 전용 피처(속도/가속/공 높이/접근속도/몸 방향)가 추가로 붙는다.

    python src/build_dataset.py statsbomb   -> out/ds_statsbomb.csv
    python src/build_dataset.py dfl         -> out/ds_dfl.csv
    python src/build_dataset.py both
"""
import glob, gzip, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
SB_DIR = os.path.join(HERE, "..", "data", "statsbomb", "comp11_season27")


# ============================================================= StatsBomb
# 누출 방지: key pass의 goal_assist / shot_assist 는 '그 슛이 골이 됐는지'를 뜻하므로 절대 쓰지 않는다.
#           shot의 deflected / redirect / end_location / saved_* 도 슛 이후 정보라 제외.
def _ts(e):
    h, m, sec = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def sb_context(ev):
    """슛마다 이벤트 맥락 피처: 슛 태그, 키패스 유형, 포제션 흐름."""
    byid = {e["id"]: e for e in ev}
    poss = {}
    for e in ev:
        poss.setdefault(e["possession"], []).append(e)
    ctx = {}
    for e in ev:
        if e["type"]["name"] != "Shot":
            continue
        s = e["shot"]
        c = {k: int(bool(s.get(k))) for k in ("first_time", "aerial_won", "one_on_one", "open_goal")}

        kp = byid.get(s.get("key_pass_id"))
        p = kp.get("pass", {}) if kp else {}
        if p:
            tech = p.get("technique", {}).get("name")
            c.update(has_assist=1, ast_length=p.get("length"),
                     ast_height=p.get("height", {}).get("name", "None"),
                     ast_cross=int(bool(p.get("cross"))),
                     ast_through=int(bool(p.get("through_ball")) or tech == "Through Ball"),
                     ast_cutback=int(bool(p.get("cut_back"))),
                     ast_switch=int(bool(p.get("switch"))),
                     ast_setpiece=p.get("type", {}).get("name", "Open"))
        else:
            c.update(has_assist=0, ast_length=np.nan, ast_height="None", ast_cross=0,
                     ast_through=0, ast_cutback=0, ast_switch=0, ast_setpiece="None")

        seq = [x for x in poss[e["possession"]] if x["index"] < e["index"] and x["period"] == e["period"]]
        own = [x for x in seq if x.get("team", {}).get("id") == e["team"]["id"]]
        dur = _ts(e) - _ts(seq[0]) if seq else 0.0
        start = next((x for x in own if x.get("location")), None)
        x0 = start["location"][0] * G.SB_SX if start else np.nan
        c.update(poss_duration=dur,
                 poss_n_pass=sum(1 for x in own if x["type"]["name"] == "Pass"),
                 poss_start_x=x0,
                 poss_directness=(e["location"][0] * G.SB_SX - x0) / max(dur, 1.0) if start else np.nan)

        acts = [x for x in own if x["type"]["name"] not in ("Ball Receipt*", "Pressure", "Duel")]
        c["prev_type"] = acts[-1]["type"]["name"] if acts else "None"
        last = acts[-1] if acts else None
        if last is not None and last["type"]["name"] == "Carry" and last.get("player", {}).get("id") == e["player"]["id"]:
            a, b = last["location"], last["carry"]["end_location"]
            c["carry_len"] = float(np.hypot((b[0] - a[0]) * G.SB_SX, (b[1] - a[1]) * G.SB_SY))
        else:
            c["carry_len"] = 0.0
        ctx[e["id"]] = c
    return ctx


def build_statsbomb():
    rows = []
    files = sorted(glob.glob(os.path.join(SB_DIR, "events_*.json.gz")))
    for k, fp in enumerate(files, 1):
        with gzip.open(fp, "rb") as fh:
            ev = json.loads(fh.read().decode())
        mid = int(os.path.basename(fp)[7:-8])
        ctx = sb_context(ev)
        for e in ev:
            if e["type"]["name"] != "Shot":
                continue
            s = e["shot"]
            ff = s.get("freeze_frame") or []
            sx, sy = G.from_statsbomb(e["location"])[0]

            opp, gkflag, mate = [], [], []
            for p in ff:
                xy = G.from_statsbomb(p["location"])[0]
                if p.get("teammate"):
                    mate.append(xy)
                else:
                    opp.append(xy)
                    gkflag.append(p.get("position", {}).get("name") == "Goalkeeper")

            f = G.shot_features(sx, sy, np.array(opp).reshape(-1, 2),
                                np.array(gkflag, dtype=bool), np.array(mate).reshape(-1, 2),
                                n_visible=len(ff))
            f.update(
                source="statsbomb", match_id=mid, shot_id=e["id"],
                goal=int(s["outcome"]["name"] == "Goal"),
                ref_xg=s.get("statsbomb_xg"),
                shot_type=s["type"]["name"],
                body_part=s["body_part"]["name"],
                is_head=int(s["body_part"]["name"] == "Head"),
                is_fk=int(s["type"]["name"] == "Free Kick"),
                technique=s.get("technique", {}).get("name"),
                play_pattern=e.get("play_pattern", {}).get("name"),
                under_pressure=int(bool(e.get("under_pressure"))),
                minute=e.get("minute"), period=e.get("period"),
                x_raw=e["location"][0], y_raw=e["location"][1],
            )
            f.update(ctx.get(e["id"], {}))
            rows.append(f)
        if k % 80 == 0:
            print(f"  statsbomb {k}/{len(files)}  누적 슛 {len(rows):,}", flush=True)
    return pd.DataFrame(rows)


# ============================================================= DFL
def build_dfl():
    ff = pd.read_parquet(os.path.join(OUT, "freeze_frames.parquet"))
    sh = pd.read_csv(os.path.join(OUT, "shots.csv"))
    pl = pd.read_csv(os.path.join(OUT, "players.csv"))
    gks = set(pl.loc[pl.PlayingPosition == "TW", "PersonId"])

    sh = sh[sh["ev.CalculatedFrame"].notna()].copy()
    sh["N"] = sh["ev.CalculatedFrame"].astype(int)
    sh["ev.EventTime"] = pd.to_datetime(sh["ev.EventTime"], format="ISO8601", utc=True)
    sh = sh.sort_values(["ev.MatchId", "ev.EventTime"])

    # 러닝 스코어
    sh["goal"] = (sh.outcome == "SuccessfulShot").astype(int)
    sf, sa = [], []
    for _, g in sh.groupby("ev.MatchId", sort=False):
        tal = {}
        for _, r in g.iterrows():
            t = r["Team"]; opp = [k for k in tal if k != t]
            sf.append(tal.get(t, 0)); sa.append(tal.get(opp[0], 0) if opp else 0)
            tal[t] = tal.get(t, 0) + r["goal"]
    sh["score_for"], sh["score_against"] = sf, sa
    k0 = sh.groupby("ev.MatchId")["ev.EventTime"].transform("min")
    sh["minute"] = (sh["ev.EventTime"] - k0).dt.total_seconds() / 60.0

    idx = {key: g for key, g in ff.groupby(["MatchId", "N"])}
    rows = []
    for _, s in sh.iterrows():
        fr = idx.get((s["ev.MatchId"], s["N"]))
        if fr is None:
            continue
        prev = idx.get((s["ev.MatchId"], s["N"] - 25))          # 1초 전
        flip = abs(s["ev.X-PositionFromTracking"] - 105) > abs(s["ev.X-PositionFromTracking"])
        sx, sy = G.from_dfl_tracking([s["ev.X-PositionFromTracking"] - 52.5,
                                      s["ev.Y-PositionFromTracking"] - 34.0], flip)[0]

        P = fr[(fr.TeamId != "BALL") & (fr.TeamId != "referee")]
        xy = G.from_dfl_tracking(np.c_[P.X.values, P.Y.values], flip)
        is_opp = (P.TeamId != s["Team"]).values & (P.PersonId != s["Player"]).values
        is_mate = (P.TeamId == s["Team"]).values & (P.PersonId != s["Player"]).values
        gkflag = P.PersonId.isin(gks).values[is_opp]

        f = G.shot_features(sx, sy, xy[is_opp], gkflag, xy[is_mate], n_visible=len(P))

        # --- 트래킹 전용 (StatsBomb으로는 만들 수 없는 것) ---
        ball = fr[fr.TeamId == "BALL"]
        me = P[P.PersonId == s["Player"]]
        f["shooter_speed"] = float(me.S.iloc[0]) if len(me) else np.nan
        f["shooter_accel"] = float(me.A.iloc[0]) if len(me) else np.nan
        f["ball_z"] = float(ball.Z.iloc[0]) if len(ball) else np.nan
        f["ball_speed"] = float(ball.S.iloc[0]) if len(ball) else np.nan
        f["def_closing"], f["orient"] = np.nan, np.nan
        if prev is not None:
            Q = prev[(prev.TeamId != "BALL") & (prev.TeamId != "referee")]
            qxy = G.from_dfl_tracking(np.c_[Q.X.values, Q.Y.values], flip)
            qme = Q.PersonId.values == s["Player"]
            if qme.any():
                px, py = qxy[qme][0]
                vx, vy = sx - px, sy - py
                if np.hypot(vx, vy) > 0.3:                     # 정지 상태면 방향 무의미
                    a = abs(np.degrees(np.arctan2(G.GOAL_Y - sy, G.GOAL_X - sx) - np.arctan2(vy, vx)))
                    f["orient"] = min(a, 360 - a)
                qopp = (Q.TeamId != s["Team"]).values & (~Q.PersonId.isin(gks).values)
                ids_prev = Q.PersonId.values[qopp]
                ids_now = P.PersonId.values[is_opp][~gkflag]
                if len(ids_prev) and len(ids_now):
                    mp = {pid: qxy[qopp][i] for i, pid in enumerate(ids_prev)}
                    now = xy[is_opp][~gkflag]
                    dn, dp = [], []
                    for i, pid in enumerate(ids_now):
                        if pid in mp:
                            dn.append(np.hypot(*(now[i] - [sx, sy])))
                            dp.append(np.hypot(*(mp[pid] - [px, py])))
                    if dn:
                        o = np.argsort(dn)[:3]                  # 가장 가까운 3명
                        f["def_closing"] = float(np.mean(np.array(dp)[o] - np.array(dn)[o]))

        f.update(source="dfl", match_id=s["ev.MatchId"], shot_id=s["ev.EventId"],
                 goal=int(s["goal"]), ref_xg=s["xG"],
                 shot_type=s["setpiece_parent"], body_part=s["TypeOfShot"],
                 is_head=int(s["TypeOfShot"] == "head"), is_fk=int(s["setpiece_parent"] == "FreeKick"),
                 technique=s["TakerBallControl"], play_pattern=s["BuildUp"],
                 under_pressure=np.nan, minute=s["minute"], period=np.nan,
                 score_diff=s["score_for"] - s["score_against"],
                 dfl_dist=s["DistanceToGoal"], dfl_angle=s["AngleToGoal"])
        rows.append(f)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(OUT, exist_ok=True)
    if what in ("statsbomb", "both"):
        t = time.time(); d = build_statsbomb()
        d.to_csv(os.path.join(OUT, "ds_statsbomb.csv"), index=False)
        print(f"[statsbomb] {len(d):,}슛 {d.goal.sum():,}골  {time.time()-t:.0f}s -> out/ds_statsbomb.csv")
    if what in ("dfl", "both"):
        t = time.time(); d = build_dfl()
        d.to_csv(os.path.join(OUT, "ds_dfl.csv"), index=False)
        print(f"[dfl]       {len(d):,}슛 {d.goal.sum():,}골  {time.time()-t:.0f}s -> out/ds_dfl.csv")
