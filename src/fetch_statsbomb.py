"""StatsBomb 오픈데이터 이벤트 JSON 다운로더.

- 재실행 가능(resumable): 이미 받은 경기는 건너뜀
- 부분 저장 방지: 임시파일에 쓰고 rename
- 실패 시 지수 백오프 재시도, 마지막에 실패 목록 보고
- gzip 저장으로 용량 약 1/5

사용:
    python src/fetch_statsbomb.py                 # La Liga 2015/16 (11/27)
    python src/fetch_statsbomb.py --comp 11 --season 27
    python src/fetch_statsbomb.py --limit 5       # 테스트
"""
import argparse, gzip, json, os, random, sys, time, urllib.error, urllib.request

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "statsbomb")


def fetch(url, tries=4, timeout=90):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xg-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                      # 네트워크/HTTP 모두
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            time.sleep((2 ** i) + random.random())
    raise last


def save_gz(path, data):
    tmp = path + ".part"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(data)
    os.replace(tmp, path)                           # 원자적 교체


def ok(path):
    """이미 받은 파일이 온전한지 (gzip이 끝까지 읽히고 JSON으로 파싱되는지)."""
    if not os.path.exists(path):
        return False
    try:
        with gzip.open(path, "rb") as f:
            json.loads(f.read().decode())
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=int, default=11)      # La Liga
    ap.add_argument("--season", type=int, default=27)    # 2015/2016
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default=ROOT)
    a = ap.parse_args()

    d = os.path.join(a.outdir, f"comp{a.comp}_season{a.season}")
    os.makedirs(d, exist_ok=True)

    mpath = os.path.join(d, "matches.json.gz")
    if not ok(mpath):
        save_gz(mpath, fetch(f"{BASE}/matches/{a.comp}/{a.season}.json"))
    with gzip.open(mpath, "rb") as f:
        matches = json.loads(f.read().decode())
    matches.sort(key=lambda m: m["match_id"])
    if a.limit:
        matches = matches[: a.limit]

    print(f"[start] comp={a.comp} season={a.season}  경기 {len(matches)}개 -> {d}", flush=True)
    t0 = time.time()
    done = skipped = 0
    failed = []
    nbytes = 0

    for i, m in enumerate(matches, 1):
        mid = m["match_id"]
        p = os.path.join(d, f"events_{mid}.json.gz")
        if ok(p):
            skipped += 1
        else:
            try:
                b = fetch(f"{BASE}/events/{mid}.json")
                save_gz(p, b)
                done += 1
            except Exception as e:
                failed.append((mid, f"{type(e).__name__}: {e}"))
                print(f"  [fail] {mid}: {type(e).__name__}", flush=True)
                continue
        nbytes += os.path.getsize(p)
        if i % 20 == 0 or i == len(matches):
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(matches) - i) / rate if rate else 0
            print(f"  {i:4d}/{len(matches)}  받음={done} 건너뜀={skipped} 실패={len(failed)}"
                  f"  디스크={nbytes/1e6:6.0f}MB  경과={el/60:4.1f}분  남음~{eta/60:4.1f}분", flush=True)

    el = time.time() - t0
    print(f"\n[done] {len(matches)}경기  받음={done} 건너뜀={skipped} 실패={len(failed)}"
          f"  {nbytes/1e6:.0f}MB  {el/60:.1f}분", flush=True)
    if failed:
        print("[failed] 아래는 스크립트를 다시 실행하면 재시도됩니다:", flush=True)
        for mid, err in failed[:20]:
            print(f"   {mid}  {err}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
