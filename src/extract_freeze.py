"""Stream DFL positions XML and pull all-object freeze frames in a window around each shot."""
import re, os, sys, glob, time
import pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"..","Dfl"); OUT=os.path.join(HERE,"..","out")
FS=re.compile(r'<FrameSet ([^>]*)>')
ATTR=re.compile(r'(\w[\w-]*)="([^"]*)"')
PRE, POST = 25, 25          # +-1s around the shot frame

def extract(posfile, frames_needed):
    """frames_needed: set of int N. returns DataFrame of every tracked object at those frames."""
    want=frames_needed; rows=[]; meta=None
    with open(posfile, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<Frame "):
                q=line.index('"',10); n=int(line[10:q])
                if n in want:
                    d=dict(ATTR.findall(line))
                    d["TeamId"]=meta[0]; d["PersonId"]=meta[1]; d["GameSection"]=meta[2]
                    rows.append(d)
            elif line.startswith("<FrameSet"):
                m=FS.search(line); d=dict(ATTR.findall(m.group(1)))
                meta=(d["TeamId"], d["PersonId"], d["GameSection"])
    return pd.DataFrame(rows)

if __name__=="__main__":
    shots=pd.read_csv(os.path.join(OUT,"shots.csv"))
    shots=shots[shots["ev.CalculatedFrame"].notna()]
    allf=[]
    for pf in sorted(glob.glob(os.path.join(RAW,"DFL_04_03_positions_raw_observed_*.xml"))):
        mid=re.search(r'(DFL-MAT-\w+)\.xml',pf).group(1)
        sh=shots[shots["ev.MatchId"]==mid]
        need=set()
        for n in sh["ev.CalculatedFrame"].astype(int):
            need.update(range(n-PRE, n+POST+1))
        t=time.time(); df=extract(pf, need); el=time.time()-t
        df["MatchId"]=mid
        print(f"{mid}  shots={len(sh):3d} frames_wanted={len(need):6d} rows={len(df):7d}  {el:5.1f}s", flush=True)
        allf.append(df)
    out=pd.concat(allf,ignore_index=True)
    for c in ["N","M","BallPossession","BallStatus"]:
        if c in out: out[c]=pd.to_numeric(out[c],errors="coerce")
    for c in ["X","Y","Z","S","A","D"]:
        if c in out: out[c]=pd.to_numeric(out[c],errors="coerce")
    out.to_parquet(os.path.join(OUT,"freeze_frames.parquet"),index=False)
    print("\nTOTAL rows:",len(out)); print(out.dtypes)
