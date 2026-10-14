"""좌석 상태 판정 로직 (YOLO·OpenCV와 무관한 순수 파이썬).

매 프레임마다 좌석별로 '사람 있음 / 물건 있음' 두 가지 관측값만 받아서
아래 다섯 가지 상태 중 하나로 판정한다.

    빈자리   EMPTY     사람도 물건도 없음                    → 이용 가능
    사용중   OCCUPIED  사람이 있음
    자리비움 AWAY      물건만 있음, 아직 재석 비율이 기준 이상
    반납예정 WARNING   물건만 있음, 최근 window 동안 재석 비율 < min_ratio → 유예시간 카운트다운
    반납대상 RECLAIM   유예시간까지 지남                     → 이용 가능(관리자 조치)

판정 기준은 PR/FAQ 문서의 규칙을 따른다.
- 연속 부재 시간이 아니라 "최근 2시간 중 재석 시간 비율"로 판정 (잠깐 들렀다 나가는 꼼수 방지)
- 비율 10% 미만이면 반납예정 → 15분 유예 → 반납대상
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class SeatState(str, Enum):
    EMPTY = "빈자리"
    OCCUPIED = "사용중"
    AWAY = "자리비움"
    WARNING = "반납예정"
    RECLAIM = "반납대상"

    @property
    def available(self) -> bool:
        """학생 입장에서 앉아도 되는 자리인지."""
        return self in (SeatState.EMPTY, SeatState.RECLAIM)


@dataclass
class Rules:
    window_sec: float = 2 * 60 * 60   # 재석 비율을 계산하는 기간 (2시간)
    min_ratio: float = 0.10           # 이 비율 미만이면 반납예정
    grace_sec: float = 15 * 60        # 반납예정 → 반납대상 유예시간 (15분)
    person_on_sec: float = 3          # 사람이 이만큼 연속으로 보여야 '있음' (오검출 방지)
    person_off_sec: float = 30        # 사람이 이만큼 연속으로 안 보여야 '없음' (가림·놓침 방지)
    item_on_sec: float = 5            # 물건이 이만큼 연속으로 보여야 '있음'
    item_off_sec: float = 60          # 물건이 이만큼 연속으로 안 보여야 '없음' → 빈자리

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        d = d or {}
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def scaled(self, time_scale: float) -> "Rules":
        """시연용: 모든 시간을 time_scale배 줄인다 (예: 60 → 2시간이 2분)."""
        if time_scale == 1:
            return self
        return Rules(**{
            k: (v if k == "min_ratio" else v / time_scale)
            for k, v in self.__dict__.items()
        })


class Debounce:
    """on_sec 동안 연속 True면 켜지고, off_sec 동안 연속 False면 꺼지는 스위치."""

    def __init__(self, on_sec: float, off_sec: float):
        self.on_sec, self.off_sec = on_sec, off_sec
        self.value = False
        self._since: float | None = None   # 현재 값과 다른 관측이 시작된 시각

    def update(self, raw: bool, t: float) -> bool:
        if raw == self.value:
            self._since = None
        else:
            if self._since is None:
                self._since = t
            if t - self._since >= (self.on_sec if raw else self.off_sec):
                self.value = raw
                self._since = None
        return self.value


@dataclass
class SeatTracker:
    seat_id: str
    rules: Rules
    state: SeatState = SeatState.EMPTY
    session_start: float | None = None      # 물건/사람이 처음 자리에 생긴 시각
    warning_start: float | None = None
    last_person_time: float | None = None
    # (시각, 사람 있음) 기록. 이 구간들의 시간 가중 평균이 재석 비율이다.
    _history: deque = field(default_factory=deque)

    def __post_init__(self):
        r = self.rules
        self.person = Debounce(r.person_on_sec, r.person_off_sec)
        self.items = Debounce(r.item_on_sec, r.item_off_sec)

    # ------------------------------------------------------------------ 비율
    def _record(self, t: float, present: bool):
        # 값이 바뀐 시점만 저장한다 (매 프레임 저장하면 2시간에 수만 개가 쌓임)
        if not self._history or self._history[-1][1] != present:
            self._history.append((t, present))
        # window보다 오래된 기록은 버리되, 경계를 계산하려고 하나는 남긴다
        cutoff = t - self.rules.window_sec
        while len(self._history) >= 2 and self._history[1][0] <= cutoff:
            self._history.popleft()

    def presence_ratio(self, now: float) -> float:
        """최근 window(세션 시작 이후로 제한) 동안 사람이 있었던 시간의 비율."""
        if not self._history:
            return 0.0
        start = now - self.rules.window_sec
        if self.session_start is not None:
            start = max(start, self.session_start)
        total = present = 0.0
        pts = list(self._history)
        for (t0, p), (t1, _) in zip(pts, pts[1:] + [(now, None)]):
            a, b = max(t0, start), min(t1, now)
            if b > a:
                total += b - a
                if p:
                    present += b - a
        return present / total if total > 0 else 0.0

    def observed_sec(self, now: float) -> float:
        return 0.0 if self.session_start is None else now - self.session_start

    def grace_left(self, now: float) -> float | None:
        if self.state is not SeatState.WARNING or self.warning_start is None:
            return None
        return max(0.0, self.rules.grace_sec - (now - self.warning_start))

    # ------------------------------------------------------------------ 갱신
    def update(self, raw_person: bool, raw_items: bool, t: float) -> SeatState:
        person = self.person.update(raw_person, t)
        items = self.items.update(raw_items, t)

        if person:
            self.last_person_time = t
            if self.state is SeatState.RECLAIM:
                self._reset_session()          # 반납대상 자리에 앉은 사람 = 새 이용자
        if self.session_start is None and (person or items):
            self.session_start = t
        if self.session_start is not None:
            self._record(t, person)

        r = self.rules
        if person:
            new = SeatState.OCCUPIED
        elif not items:
            new = SeatState.EMPTY
        else:
            # 물건만 있음. 판정에 필요한 만큼(window) 관찰한 뒤에만 반납예정을 띄운다.
            # 세션이 window보다 짧으면 "방금 짐 두고 나간 사람"일 수 있으므로 자리비움.
            enough = self.observed_sec(t) >= r.window_sec
            low = self.presence_ratio(t) < r.min_ratio
            if self.state is SeatState.RECLAIM:
                new = SeatState.RECLAIM            # 사람이 오기 전까지 유지
            elif enough and low:
                if self.state is not SeatState.WARNING:
                    self.warning_start = t
                new = SeatState.WARNING
                if t - self.warning_start >= r.grace_sec:
                    new = SeatState.RECLAIM
            else:
                new = SeatState.AWAY

        if new is not SeatState.WARNING and new is not SeatState.RECLAIM:
            self.warning_start = None
        if new is SeatState.EMPTY:
            self._reset_session()
        self.state = new
        return new

    def _reset_session(self):
        self.session_start = None
        self.warning_start = None
        self._history.clear()
