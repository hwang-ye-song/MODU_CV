"""YOLO 없이 관측값만 넣어 판정 규칙을 확인한다.  python -m pytest mvp -q"""

from room_logic import RoomState, RoomTracker, Rules

R = Rules(limit_sec=30 * 60, on_sec=2, off_sec=10)


def run(tr, person, start, end, step=1.0):
    t = start
    while t <= end:
        tr.update(person, t)
        t += step
    return tr.state


def test_studying_stays_red():
    tr = RoomTracker("A", R)
    assert run(tr, True, 0, 3600) is RoomState.OCCUPIED


def test_short_occlusion_does_not_flip():
    tr = RoomTracker("A", R)
    run(tr, True, 0, 60)
    assert run(tr, False, 61, 68) is RoomState.OCCUPIED   # 8초 가림
    assert run(tr, True, 69, 80) is RoomState.OCCUPIED


def test_leave_yellow_then_green_after_30min():
    tr = RoomTracker("A", R)
    run(tr, True, 0, 60)
    assert run(tr, False, 61, 61 + 29 * 60) is RoomState.AWAY       # 29분: 노랑
    assert run(tr, False, 61 + 29 * 60 + 1, 61 + 31 * 60) is RoomState.FREE   # 30분 넘음: 초록


def test_return_resets_timer():
    tr = RoomTracker("A", R)
    run(tr, True, 0, 60)
    run(tr, False, 61, 61 + 20 * 60)
    run(tr, True, 61 + 20 * 60 + 1, 61 + 20 * 60 + 10)               # 돌아옴
    assert tr.state is RoomState.OCCUPIED
    t = 61 + 20 * 60 + 11
    assert run(tr, False, t, t + 25 * 60) is RoomState.AWAY          # 다시 비워도 처음부터


def test_empty_at_start_counts_from_start():
    tr = RoomTracker("A", R)
    assert run(tr, False, 0, 10 * 60) is RoomState.AWAY
    assert run(tr, False, 10 * 60 + 1, 31 * 60) is RoomState.FREE


def test_time_scale_demo():
    fast = Rules(limit_sec=30, on_sec=2, off_sec=3)   # 30분 → 30초 시연
    tr = RoomTracker("A", fast)
    run(tr, True, 0, 10)
    assert run(tr, False, 10.5, 30) is RoomState.AWAY
    assert run(tr, False, 30.5, 45) is RoomState.FREE
