"""xG 파이프라인을 순서대로 한 번에 실행한다.

    python run_all.py                  전체 (1~7단계)
    python run_all.py --from dataset   5단계부터 이어서
    python run_all.py --from train     v1 학습 + v2 파인튜닝만
    python run_all.py --only finetune  7단계(v2)만
    python run_all.py --analysis       전체 + DFL 단독 분석 스크립트
    python run_all.py --list           단계 목록
    python run_all.py -q               각 단계의 마지막 줄만 표시

실패하면 그 단계에서 멈추고 에러 끝부분을 보여준다.
고친 뒤 `--from <단계>`로 이어서 돌리면 된다.
"""
import argparse, collections, glob, os, subprocess, sys, time

# 한국어 Windows 콘솔(cp949)에서 출력이 리다이렉트돼도 죽지 않게
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    ("matchinfo", "src/parse_matchinfo.py", [], "DFL 경기정보 -> out/matches.csv, players.csv"),
    ("events",    "src/parse_events.py",    [], "DFL 이벤트 -> out/shots.csv"),
    ("freeze",    "src/extract_freeze.py",  [], "DFL 트래킹 2.5GB -> out/freeze_frames.parquet"),
    ("statsbomb", "src/fetch_statsbomb.py", [], "StatsBomb La Liga 15/16 다운로드 (있으면 건너뜀)"),
    ("dataset",   "src/build_dataset.py",   ["both"], "공용 피처 -> out/ds_statsbomb.csv, ds_dfl.csv"),
    ("train",     "src/train_v1.py",        [], "v1 학습·평가 -> out/v1_results.csv, v1_oof.csv"),
    ("finetune",  "src/train_v2.py",        [], "v2 DFL 파인튜닝 -> out/v2_results.csv, v2_oof.csv"),
    ("dfl_only",  "src/train_dfl.py",       [], "DFL 단독 모델 (비교용) -> out/dfl_results.csv"),
]
ANALYSIS = [
    ("features",      "src/features.py",      [], "DFL 트래킹 피처 -> out/features.csv"),
    ("features_plus", "src/features_plus.py", [], "DFL 창의 피처 -> out/features_plus.csv"),
    ("eda",           "src/eda.py",           [], "DFL 탐색 분석"),
    ("feasibility",   "src/feasibility.py",   [], "DFL 직접학습 vs 증류"),
    ("whatdrives",    "src/whatdrives.py",    [], "DFL xG 드라이버·표본 크기"),
    ("ceiling",       "src/ceiling.py",       [], "DFL 평가 프로토콜별 AUC"),
    ("search",        "src/search.py",        [], "DFL 전진선택 (선택 편향 있음)"),
    ("nested",        "src/nested.py",        [], "DFL 중첩 CV (정직한 천장)"),
]
NEEDS_DFL = {"matchinfo", "events", "freeze"}


def check_dfl():
    n = len(glob.glob(os.path.join(ROOT, "Dfl", "DFL_0[234]_*.xml")))
    if n < 21:
        print(f"!! Dfl/ 폴더에 DFL XML이 {n}개뿐입니다 (21개 필요: 7경기 x 3종).")
        print("   DFL 오픈데이터를 C:\\xg\\Dfl\\ 에 넣은 뒤 다시 실행하세요.")
        sys.exit(1)


def run(script, args, quiet):
    cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, script), *args]
    # wmic이 없는 Windows 11에선 joblib이 물리 코어를 못 세고 트레이스백을 찍는다.
    # 논리 코어의 절반(≈물리 코어)을 지정하면 그 탐지를 건너뛴다.
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               LOKY_MAX_CPU_COUNT=str(max(1, (os.cpu_count() or 2) // 2)))
    tail = collections.deque(maxlen=25)
    t = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if not quiet:
            print("    " + line, flush=True)
    return p.wait(), time.time() - t, tail


def main():
    ap = argparse.ArgumentParser(description="xG 파이프라인 실행")
    names = [s[0] for s in PIPELINE + ANALYSIS]
    ap.add_argument("--from", dest="start", choices=names, help="이 단계부터 실행")
    ap.add_argument("--only", choices=names, help="이 단계만 실행")
    ap.add_argument("--analysis", action="store_true", help="DFL 단독 분석 스크립트까지 실행")
    ap.add_argument("--list", action="store_true", help="단계 목록만 출력")
    ap.add_argument("-q", "--quiet", action="store_true", help="각 단계의 마지막 줄만 표시")
    a = ap.parse_args()

    analysis_names = {s[0] for s in ANALYSIS}
    want_analysis = a.analysis or a.only in analysis_names or a.start in analysis_names
    steps = PIPELINE + (ANALYSIS if want_analysis else [])
    if a.list:
        for i, (n, s, _, desc) in enumerate(PIPELINE + ANALYSIS, 1):
            tag = "" if i <= len(PIPELINE) else "  (--analysis)"
            print(f"  {i:2d}. {n:14s} {desc}{tag}")
        return
    if a.only:
        steps = [s for s in steps if s[0] == a.only]
    elif a.start:
        steps = steps[[s[0] for s in steps].index(a.start):]

    if any(s[0] in NEEDS_DFL for s in steps):
        check_dfl()

    print(f"실행할 단계 {len(steps)}개: {' -> '.join(s[0] for s in steps)}\n", flush=True)
    done = []
    t0 = time.time()
    for i, (name, script, args, desc) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}  - {desc}", flush=True)
        rc, el, tail = run(script, args, a.quiet)
        if rc != 0:
            print(f"\n!! '{name}' 단계 실패 (exit {rc}, {el:.0f}초). 마지막 출력:")
            for line in tail:
                print("    " + line)
            print(f"\n고친 뒤 이어서 실행:  python run_all.py --from {name}")
            sys.exit(rc)
        if a.quiet and tail:
            print("    " + tail[-1])
        print(f"    완료 ({el:.0f}초)\n", flush=True)
        done.append((name, el))

    print("=" * 50)
    for name, el in done:
        print(f"  {name:14s} {el:6.0f}초")
    print(f"  {'합계':14s} {time.time()-t0:6.0f}초")
    if any(n in ("train", "finetune") for n, _ in done):
        print("\n결과는 explore.ipynb에서 그림으로 보거나, 표로 보기:")
        for f in ("v1_results", "v2_results"):
            print(f"  python -X utf8 -c \"import pandas as pd; "
                  f"print(pd.read_csv('out/{f}.csv').round(4).to_string())\"")


if __name__ == "__main__":
    main()
