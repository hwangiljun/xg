"""Parse DFL raw event XML -> tidy shot table. Finds ShotAtGoal at any nesting depth."""
import glob, os
import pandas as pd
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, "..", "Dfl")
OUT  = os.path.join(HERE, "..", "out")

OUTCOMES = {"SuccessfulShot","SavedShot","BlockedShot","ShotWide","ShotWoodWork","OtherShot"}

def shots_from_file(path):
    root = ET.parse(path).getroot()
    rows = []
    for ev in root.findall("Event"):
        # walk to find ShotAtGoal at any depth, remembering its parent chain
        stack = [(ev, [])]
        while stack:
            node, chain = stack.pop()
            for k in node:
                if k.tag == "ShotAtGoal":
                    rec = {f"ev.{a}": v for a, v in ev.attrib.items()}
                    rec["setpiece_parent"] = chain[-1] if chain else "openPlay"
                    for c in chain:                      # FreeKick / Penalty attrs
                        rec.update({f"{c}.{a}": v for a, v in node.attrib.items()})
                    rec.update({a: v for a, v in k.attrib.items()})
                    for o in k:
                        if o.tag in OUTCOMES:
                            rec["outcome"] = o.tag
                            rec.update({f"out.{a}": v for a, v in o.attrib.items()})
                    rows.append(rec)
                else:
                    stack.append((k, chain + [k.tag]))
    return pd.DataFrame(rows)

NUM = ["ev.X-Position","ev.Y-Position","ev.X-Source-Position","ev.Y-Source-Position",
       "ev.X-PositionFromTracking","ev.Y-PositionFromTracking","ev.CalculatedFrame",
       "Pressure","GoalDistanceGoalkeeper","AngleToGoal","PlayerSpeed","DistanceToGoal",
       "xG","AmountOfDefenders","ShotOrigin","BallPossessionPhase"]

def load():
    dfs=[]
    for f in sorted(glob.glob(os.path.join(RAW, "DFL_03_02_events_raw_*.xml"))):
        d = shots_from_file(f); d["src_file"]=os.path.basename(f); dfs.append(d)
    s = pd.concat(dfs, ignore_index=True)
    for c in NUM:
        if c in s: s[c] = pd.to_numeric(s[c], errors="coerce")
    for c in ["CounterAttack","InsideBox","AfterFreeKick"]:
        if c in s: s[c] = s[c].map({"true":1,"false":0})
    s["goal"] = (s["outcome"] == "SuccessfulShot").astype(int)
    s["ev.EventTime"] = pd.to_datetime(s["ev.EventTime"], format="ISO8601")
    return s.sort_values(["ev.MatchId","ev.EventTime"]).reset_index(drop=True)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    s = load()
    s.to_csv(os.path.join(OUT,"shots.csv"), index=False)
    print(f"shots={len(s)} matches={s['ev.MatchId'].nunique()} goals={s.goal.sum()} "
          f"conv={s.goal.mean():.3f}")
    print("\nsetpiece_parent:\n", s.setpiece_parent.value_counts())
    print("\noutcome:\n", s.outcome.value_counts())
    print("\nper match:\n", s.groupby("ev.MatchId").agg(shots=("goal","size"), goals=("goal","sum")))
