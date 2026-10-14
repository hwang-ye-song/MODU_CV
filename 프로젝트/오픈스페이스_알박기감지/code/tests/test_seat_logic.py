"""판정 규칙 시나리오 테스트 (YOLO 없이 관측값만 넣어 본다)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seat_logic import Rules, SeatState, SeatTracker  # noqa: E402

MIN = 60
S = SeatState


def run(tr: SeatTracker, start: float, minutes: float, person: bool, items: bool, dt: float = 1.0):
    """start초부터 minutes분 동안 같은 관측값을 dt초 간격으로 넣는다. 끝 시각을 돌려준다."""
    t = start
    end = start + minutes * MIN
    while t < end:
        tr.update(person, items, t)
        t += dt
    return t


def new() -> SeatTracker:
    return SeatTracker("S1", Rules())


def test_empty_seat_stays_empty():
    tr = new()
    run(tr, 0, 10, False, False)
    assert tr.state is S.EMPTY


def test_studying_person_is_occupied():
    tr = new()
    run(tr, 0, 30, True, True)
    assert tr.state is S.OCCUPIED


def test_short_occlusion_keeps_occupied():
    # 옆 사람·칸막이에 20초 가려져도 사용중 유지 (person_off_sec=30)
    tr = new()
    t = run(tr, 0, 10, True, True)
    run(tr, t, 20 / MIN, False, True)
    assert tr.state is S.OCCUPIED


def test_bathroom_break_is_away_not_warning():
    # 1시간 50분 공부 후 10분 화장실 → 재석 비율이 높으므로 자리비움
    tr = new()
    t = run(tr, 0, 110, True, True)
    run(tr, t, 10, False, True)
    assert tr.state is S.AWAY


def test_bag_left_alone_goes_warning_then_reclaim():
    tr = new()
    t = run(tr, 0, 2, True, True)          # 잠깐 앉았다가
    t = run(tr, t, 115, False, True)       # 가방만 두고 떠남
    assert tr.state is S.AWAY              # 아직 2시간이 안 지남
    t = run(tr, t, 5, False, True)
    assert tr.state is S.WARNING
    assert 0 < tr.grace_left(t) <= 15 * MIN
    run(tr, t, 15, False, True)
    assert tr.state is S.RECLAIM
    assert tr.state.available


def test_return_during_grace_keeps_seat():
    tr = new()
    t = run(tr, 0, 2, True, True)
    t = run(tr, t, 125, False, True)
    assert tr.state is S.WARNING
    t = run(tr, t, 10, True, True)         # 유예 중에 돌아옴
    assert tr.state is S.OCCUPIED


def test_drop_in_trick_is_caught():
    # 기준 시간 직전에 5분씩 들렀다 나가는 꼼수: 연속 부재 시간으로는 못 잡지만 비율로 잡힌다
    # (부재는 매번 100분이라 '2시간 연속 부재' 규칙이었다면 한 번도 안 걸린다)
    tr = new()
    seen = set()
    t = 0
    for _ in range(2):
        t = run(tr, t, 5, True, True)
        for _ in range(100):
            t = run(tr, t, 1, False, True)
            seen.add(tr.state)
    assert S.WARNING in seen and S.RECLAIM in seen


def test_items_removed_becomes_empty():
    tr = new()
    t = run(tr, 0, 30, True, True)
    run(tr, t, 2, False, False)            # 짐 챙겨 나감 (item_off_sec=60)
    assert tr.state is S.EMPTY
    assert tr.session_start is None


def test_reclaimed_seat_new_user_starts_fresh_session():
    tr = new()
    t = run(tr, 0, 140, False, True)
    assert tr.state is S.RECLAIM
    t = run(tr, t, 1, True, True)          # 새 학생이 앉음
    assert tr.state is S.OCCUPIED
    assert tr.presence_ratio(t) > 0.9      # 이전 사람의 부재 기록은 지워짐


def test_time_scale_compresses_rules():
    r = Rules().scaled(60)
    assert r.window_sec == 2 * MIN and r.grace_sec == 15 and r.min_ratio == 0.10
