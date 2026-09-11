"""Parse match information XML -> match meta + player/lineup table."""
import glob, os
import pandas as pd
import xml.etree.ElementTree as ET
HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"..","Dfl"); OUT=os.path.join(HERE,"..","out")

def load():
    matches, players = [], []
    for f in sorted(glob.glob(os.path.join(RAW,"DFL_02_01_matchinformation_*.xml"))):
        mi=ET.parse(f).getroot().find("MatchInformation")
        g=mi.find("General").attrib; e=mi.find("Environment").attrib
        matches.append({**g, **{f"env.{k}":v for k,v in e.items()}})
        for team in mi.find("Teams"):
            for p in team.find("Players"):
                players.append({"MatchId":g["MatchId"],"TeamId":team.attrib["TeamId"],
                                "TeamName":team.attrib["TeamName"],"Role":team.attrib["Role"],
                                "LineUp":team.attrib.get("LineUp"), **p.attrib})
    return pd.DataFrame(matches), pd.DataFrame(players)

if __name__=="__main__":
    m,p=load()
    m.to_csv(os.path.join(OUT,"matches.csv"),index=False); p.to_csv(os.path.join(OUT,"players.csv"),index=False)
    print(m[["MatchId","MatchDay","Season","MatchTitle","Result","env.PitchX","env.PitchY","env.NumberOfSpectators"]].to_string(index=False))
    print("\nplayers rows:",len(p),"| GKs:",(p.PlayingPosition=="TW").sum())
    print(p.PlayingPosition.value_counts().to_dict())
