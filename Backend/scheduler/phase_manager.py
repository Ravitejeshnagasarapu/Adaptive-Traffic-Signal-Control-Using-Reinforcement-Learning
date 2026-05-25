from collections import deque

DIRECTIONS    = ['N', 'E', 'S', 'W']
MIN_GREEN     = 5
MAX_GREEN     = 20
BASE_GREEN    = 6
DYN_K         = 0.6
HISTORY_LEN   = 1
YELLOW_STEPS  = 2
ALL_RED_STEPS = 2


class PhaseManager:
    def __init__(self):
        self.current_dir       = None
        self.phase_timer       = 0
        self.transition        = None
        self.trans_timer       = 0
        self.YELLOW_STEPS      = YELLOW_STEPS
        self.ALL_RED_STEPS     = ALL_RED_STEPS
        self._pending_dir      = None
        self._pending_duration = MIN_GREEN
        self._rotation         = deque(DIRECTIONS)
        self._history          = deque(maxlen=HISTORY_LEN)
        self.total_phases      = 0
        self.phase_counts      = {d: 0 for d in DIRECTIONS}

    def is_locked(self) -> bool:
        return self.phase_timer > 0 or self.transition is not None

    def step_timer(self) -> str:
        if self.transition == 'YELLOW':
            self.trans_timer -= 1
            if self.trans_timer <= 0:
                self.transition  = 'ALL_RED'
                self.trans_timer = self.ALL_RED_STEPS
            return 'YELLOW'
        elif self.transition == 'ALL_RED':
            self.trans_timer -= 1
            if self.trans_timer <= 0:
                self.transition  = None
                self.trans_timer = 0
            return 'ALL_RED'
        elif self.phase_timer > 0:
            self.phase_timer -= 1
            return 'GREEN'
        return 'IDLE'

    def can_select(self) -> bool:
        return self.phase_timer == 0 and self.transition is None

    def set_phase(self, direction: str, duration: int):
        duration = max(MIN_GREEN, min(MAX_GREEN, duration))
        if self.current_dir is not None and direction != self.current_dir:
            self._pending_dir      = direction
            self._pending_duration = duration
            self.transition        = 'YELLOW'
            self.trans_timer       = self.YELLOW_STEPS
        else:
            self._commit(direction, duration)

    def commit_pending(self):
        if self._pending_dir is not None:
            self._commit(self._pending_dir, self._pending_duration)
            self._pending_dir = None

    def _commit(self, direction: str, duration: int):
        self.current_dir = direction
        self.phase_timer = duration
        if direction in self._rotation:
            self._rotation.remove(direction)
        self._rotation.append(direction)
        self._history.append(direction)
        self.total_phases            += 1
        self.phase_counts[direction] += 1

    def get_next_direction(self, q_values, queues, emergency_dir=None) -> str:
        if emergency_dir and emergency_dir in DIRECTIONS:
            return emergency_dir

        recently   = set(self._history)
        candidates = [d for d in DIRECTIONS if d not in recently]
        if not candidates:
            candidates = list(DIRECTIONS)

        with_queue = [d for d in candidates if queues.get(d, 0) > 0]
        if with_queue:
            candidates = with_queue

        if q_values:
            return max(candidates, key=lambda d: q_values.get(d, 0.0))
        else:
            rotation_order = list(self._rotation)
            return next((d for d in rotation_order if d in candidates), candidates[0])

    def get_dynamic_duration(self, queue_len: int) -> int:
        raw = BASE_GREEN + DYN_K * queue_len
        return max(MIN_GREEN, min(MAX_GREEN, int(raw)))

    def fairness_score(self) -> float:
        counts = list(self.phase_counts.values())
        total  = sum(counts)
        if total == 0: return 1.0
        mean = total / len(counts)
        std  = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
        cv   = std / mean if mean > 0 else 0.0
        return max(0.0, 1.0 - cv)