"""DFL과 StatsBomb 양쪽에서 '같은 정의'의 xG 피처를 뽑기 위한 공용 기하 모듈.

정규 좌표계 (canonical):
    미터 단위, 105 x 68 피치, 공격 골대는 항상 (105, 34).
    골포스트는 실제 규격 y = 34 +- 3.66.

양쪽 원본의 차이:
    DFL       트래킹 중앙원점(-52.5..52.5, -34..34), 팀마다 공격 방향이 다름 -> 미러링 필요
    StatsBomb 120 x 80 정규화 좌표, 공격 방향은 항상 +x -> 축척만 변환

주의: StatsBomb의 y축 축척(0.850)으로 골문 폭(8단위)을 환산하면 6.80 m가 나와
      실제 7.32 m와 어긋난다. 선수 위치는 피치 축척으로 변환하되 골문은 실제
      규격(POST=3.66)을 쓴다 — 두 데이터의 각도 정의를 일치시키기 위한 선택.
"""
import numpy as np

PITCH_X, PITCH_Y = 105.0, 68.0
GOAL_X, GOAL_Y = 105.0, 34.0
POST = 3.66                                  # 골문 반폭 (7.32 m / 2)
BOX_X, BOX_Y0, BOX_Y1 = 88.5, 13.84, 54.16   # 페널티 에어리어
SB_SX, SB_SY = PITCH_X / 120.0, PITCH_Y / 80.0   # 0.875, 0.850


# ---------------------------------------------------------------- 좌표 변환
def from_statsbomb(xy):
    """StatsBomb [x,y] (120x80) -> 미터. 공격 방향은 이미 +x로 정규화돼 있다."""
    a = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.c_[a[:, 0] * SB_SX, a[:, 1] * SB_SY]


def from_dfl_tracking(xy, flip):
    """DFL 트래킹 중앙원점 -> 미터 코너원점. flip=True면 x=0 골대를 공격 중이라 미러링."""
    a = np.asarray(xy, dtype=float).reshape(-1, 2)
    x, y = a[:, 0] + PITCH_X / 2, a[:, 1] + PITCH_Y / 2
    if flip:
        x, y = PITCH_X - x, PITCH_Y - y
    return np.c_[x, y]


# ---------------------------------------------------------------- 기하 primitives
def subtended_angle(sx, sy):
    """골문 7.32 m가 슈터에게 벌어져 보이는 각(도). DFL AngleToGoal과 MAE 0.0024도로 일치."""
    dx, dy = GOAL_X - sx, sy - GOAL_Y
    return float(np.degrees(abs(np.arctan2(POST - dy, dx) - np.arctan2(-POST - dy, dx))))


def in_cone(px, py, sx, sy):
    """슈터-양 골포스트 삼각형 내부 판정 (벡터화)."""
    px, py = np.atleast_1d(px), np.atleast_1d(py)
    ax, ay = sx, sy
    bx, by = GOAL_X, GOAL_Y + POST
    cx, cy = GOAL_X, GOAL_Y - POST
    s = lambda X, Y, x1, y1, x2, y2: (x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)
    d1, d2, d3 = s(px, py, ax, ay, bx, by), s(px, py, bx, by, cx, cy), s(px, py, cx, cy, ax, ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def open_goal_fraction(sx, sy, bx, by, r=0.42, n=121):
    """슈터에서 골문 폭으로 광선 n개를 쏴, 차단체 반경 r 안을 지나지 않는 비율.

    선수를 반지름 r의 원으로 보고 광선-점 최단거리로 차단을 판정한다.
    """
    bx, by = np.atleast_1d(bx), np.atleast_1d(by)
    if bx.size == 0:
        return 1.0
    ty = np.linspace(GOAL_Y - POST + 0.15, GOAL_Y + POST - 0.15, n)
    dx = np.full_like(ty, GOAL_X - sx)
    dy = ty - sy
    dd = dx * dx + dy * dy
    wx, wy = bx - sx, by - sy
    t = np.clip((wx[None, :] * dx[:, None] + wy[None, :] * dy[:, None]) / dd[:, None], 0, 1)
    cx = sx + t * dx[:, None]
    cy = sy + t * dy[:, None]
    d = np.hypot(bx[None, :] - cx, by[None, :] - cy)
    return float((d.min(axis=1) > r).mean())


# ---------------------------------------------------------------- 공용 피처
def shot_features(sx, sy, opp, opp_is_gk, mate, n_visible=None):
    """정규 좌표 입력 -> 두 데이터 공통으로 계산 가능한 피처.

    sx, sy      슈터 위치 (m)
    opp         (n,2) 상대팀 선수 위치 (슈터 제외)
    opp_is_gk   (n,) bool, 상대 골키퍼 여부
    mate        (m,2) 같은 팀 선수 위치
    n_visible   프레임에 실제로 잡힌 선수 수 (StatsBomb은 가변, DFL은 22)
    """
    opp = np.asarray(opp, dtype=float).reshape(-1, 2)
    mate = np.asarray(mate, dtype=float).reshape(-1, 2)
    opp_is_gk = np.asarray(opp_is_gk, dtype=bool).reshape(-1)

    out = opp[~opp_is_gk] if opp.size else opp        # 필드 수비수
    gk = opp[opp_is_gk] if opp.size else opp

    dist = float(np.hypot(GOAL_X - sx, sy - GOAL_Y))
    ang = subtended_angle(sx, sy)

    of_all = open_goal_fraction(sx, sy, opp[:, 0], opp[:, 1])
    of_nogk = open_goal_fraction(sx, sy, out[:, 0], out[:, 1])

    f = {
        "dist": dist,
        "angle": ang,
        "y_off": float(abs(sy - GOAL_Y)),
        "open_frac": of_all,
        "open_frac_nogk": of_nogk,
        "open_angle": of_all * ang,
        "gk_block_share": of_nogk - of_all,
        "n_visible": int(len(opp) + len(mate)) if n_visible is None else int(n_visible),
        "n_opp_visible": int(len(opp)),
        "has_gk": int(len(gk) > 0),
    }

    # 수비수 배치
    if len(out):
        d = np.hypot(out[:, 0] - sx, out[:, 1] - sy)
        cone = in_cone(out[:, 0], out[:, 1], sx, sy) & (out[:, 0] > sx)
        f.update(n_def_cone=int(cone.sum()), n_def_3m=int((d < 3).sum()),
                 n_def_5m=int((d < 5).sum()), d_def_min=float(d.min()))
        # 슛 방향 +-2 m 통로
        v = np.array([GOAL_X - sx, GOAL_Y - sy], dtype=float)
        v /= max(np.linalg.norm(v), 1e-9)
        rel = out - np.array([sx, sy])
        along = rel @ v
        perp = np.abs(rel[:, 0] * -v[1] + rel[:, 1] * v[0])
        corr = (along > 0) & (perp < 2.0)
        f.update(n_corridor=int(corr.sum()),
                 d_corridor=float(along[corr].min()) if corr.any() else 40.0)
    else:
        f.update(n_def_cone=0, n_def_3m=0, n_def_5m=0, d_def_min=np.nan,
                 n_corridor=0, d_corridor=40.0)

    # 골키퍼
    if len(gk):
        gx, gy = float(gk[0, 0]), float(gk[0, 1])
        seg = max(np.hypot(GOAL_X - sx, GOAL_Y - sy), 1e-9)
        f.update(gk_dist_goal=float(np.hypot(GOAL_X - gx, gy - GOAL_Y)),
                 gk_dist_shot=float(np.hypot(gx - sx, gy - sy)),
                 gk_in_cone=int(in_cone(gx, gy, sx, sy)[0]),
                 gk_off_line=float(abs((GOAL_X - sx) * (sy - gy) - (sx - gx) * (GOAL_Y - sy)) / seg))
    else:
        f.update(gk_dist_goal=np.nan, gk_dist_shot=np.nan, gk_in_cone=np.nan, gk_off_line=np.nan)

    inbox = lambda a: ((a[:, 0] > BOX_X) & (a[:, 1] > BOX_Y0) & (a[:, 1] < BOX_Y1)).sum() if len(a) else 0
    f.update(n_att_box=int(inbox(mate)), n_def_box=int(inbox(out)),
             inside_box=int(sx > BOX_X and BOX_Y0 < sy < BOX_Y1))
    return f
