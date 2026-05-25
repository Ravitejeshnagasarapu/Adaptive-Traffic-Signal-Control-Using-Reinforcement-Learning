import random
import numpy as np
from collections import deque

from scheduler.phase_manager import PhaseManager, MIN_GREEN, MAX_GREEN

DIRECTIONS = ['N', 'E', 'S', 'W']
DIR_INDEX  = {d: i for i, d in enumerate(DIRECTIONS)}

ACTIONS     = [("N", 8), ("E", 8), ("S", 8), ("W", 8)]
NUM_ACTIONS = 4

# -- Traffic parameters ------------------------------------------------
SPAWN_STRAIGHT = 0.10
SPAWN_RIGHT    = 0.07
EMERGENCY_PROB = 0.012

# -- State normalisation bounds ----------------------------------------
MAX_Q          = 15.0
MAX_W          = 90.0      # wider wait range
MAX_STARVATION = 40

# -- REWARD WEIGHTS ----------------------
W_THROUGHPUT  = 5.0
W_WAIT        = 2.5
W_STARVATION  = 3.0
W_EMERGENCY   = 8.0
W_FAST_SWITCH = 1.5

FAST_SWITCH_TP_THRESH = 1
MAX_STEPS             = 200


class TrafficEnv:
    def __init__(self):
        self.pm = PhaseManager()
        self.reset()

    def reset(self):
        self.sq     = {d: 0   for d in DIRECTIONS}
        self.rq     = {d: 0   for d in DIRECTIONS}
        self.waits  = {d: 0.0 for d in DIRECTIONS}
        self.emerg  = {d: 0   for d in DIRECTIONS}
        self.served = {d: 0   for d in DIRECTIONS}

        self.pm.__init__()

        self.total_tp  = 0
        self.step_no   = 0
        self.ep_reward = 0.0
        self._last_dur = 8
        self._phase_tp = 0

        self._wait_buf = deque(maxlen=30)
        self.dir_tp    = {d: 0 for d in DIRECTIONS}

        self.pm.set_phase('N', 8)
        return self._state()

    # -- Valid actions API ---------------------------------------------

    def get_initial_valid_actions(self) -> list:
        """Valid actions for the first decision point after reset."""
        return self._get_next_valid_actions()

    def _get_next_valid_actions(self) -> list:
        recently = set(self.pm._history)
        valid    = [i for i, d in enumerate(DIRECTIONS) if d not in recently]
        return valid if valid else list(range(4))

    # -- Properties ---------------------------------------------------
    @property
    def phase_name(self) -> str:
        if self.pm.transition == 'YELLOW':  return 'YELLOW'
        if self.pm.transition == 'ALL_RED': return 'ALL_RED'
        cd = self.pm.current_dir
        return f"{cd}_GREEN" if cd else "IDLE"

    @property
    def state_size(self) -> int: return 17

    @property
    def action_size(self) -> int: return NUM_ACTIONS

    def _cq(self, d: str) -> int:
        return self.sq[d] + self.rq[d]

    def _state(self) -> np.ndarray:
        """
        17-feature state vector:
          [0:4]  — normalised queue lengths (N, E, S, W)
          [4:8]  — normalised wait times
          [8]    — avg normalised wait
          [9]    — emergency flag (binary)
          [10]   — current phase (0..1)
          [11]   — time remaining in phase (0..1)
          [12:16]— starvation counters
          [16]   — last phase duration
        """
        q = [min(self._cq(d),    MAX_Q)         / MAX_Q          for d in DIRECTIONS]
        w = [min(self.waits[d],  MAX_W)          / MAX_W          for d in DIRECTIONS]
        s = [min(self.served[d], MAX_STARVATION) / MAX_STARVATION for d in DIRECTIONS]

        avg_w = sum(w) / 4.0
        emerg = 1.0 if any(v > 0 for v in self.emerg.values()) else 0.0
        cd    = self.pm.current_dir
        phase = DIR_INDEX.get(cd, 0) / 3.0 if cd else 0.0
        t_f   = min(self.pm.phase_timer, MAX_GREEN) / MAX_GREEN
        dur_f = self._last_dur / MAX_GREEN

        return np.array(q + w + [avg_w, emerg, phase, t_f] + s + [dur_f],
                        dtype=np.float32)

    def step(self, action: int):
        self.step_no += 1
        requested_dir = DIRECTIONS[action]

        for d in DIRECTIONS:
            if random.random() < SPAWN_STRAIGHT: self.sq[d] += 1
            if random.random() < SPAWN_RIGHT:    self.rq[d] += 1
            if random.random() < EMERGENCY_PROB: self.emerg[d] = 1

        # Update starvation counters
        cd = self.pm.current_dir
        for d in DIRECTIONS:
            if d == cd and self.pm.transition is None:
                self.served[d] = 0
            else:
                self.served[d] = min(self.served[d] + 1, MAX_STARVATION * 2)

        self.pm.step_timer()
        if self.pm.transition is None and self.pm._pending_dir is not None:
            self.pm.commit_pending()

        decision_step = False
        switched      = False

        if self.pm.can_select():
            decision_step = True

            # EMERGENCY: hard safety override — not reward-based
            emerg_dir = next((d for d in DIRECTIONS if self.emerg[d] == 1), None)
            chosen_dir = emerg_dir if emerg_dir else requested_dir

            chosen_queue    = self._cq(chosen_dir)
            final_dur       = self.pm.get_dynamic_duration(chosen_queue)
            self._last_dur  = final_dur
            self._phase_tp  = 0

            old_dir = self.pm.current_dir
            self.pm.set_phase(chosen_dir, final_dur)
            switched = (old_dir is not None and chosen_dir != old_dir)

        # Clear vehicles during green phase
        cd = self.pm.current_dir
        tp, em_served = 0, 0

        if cd and self.pm.transition is None and self.pm.phase_timer > 0:
            pass_s = min(self.sq[cd], 4)
            pass_r = min(self.rq[cd], 3)
            self.sq[cd] = max(0, self.sq[cd] - pass_s)
            self.rq[cd] = max(0, self.rq[cd] - pass_r)
            tp = pass_s + pass_r
            if self.emerg[cd] and tp > 0:
                self.emerg[cd] = 0
                em_served      = 1
            self.dir_tp[cd] = self.dir_tp.get(cd, 0) + tp

        self._phase_tp += tp
        self.total_tp  += tp

        for d in DIRECTIONS:
            if self._cq(d) > 0:
                self.waits[d] = min(self.waits[d] + 1, MAX_W * 2)
            else:
                self.waits[d] = max(0.0, self.waits[d] - 5)

        # Normalizations
        avg_w_norm = min(sum(self.waits.values()) / (4 * MAX_W), 1.0)
        max_w_norm = min(max(self.waits.values()) / MAX_W, 1.0)
        tp_norm    = min(tp / 7.0, 1.0)

        max_q_norm = min(max(self._cq(d) for d in DIRECTIONS) / MAX_Q, 1.0)

        # Starvation
        max_starv = max(
            self.served[d] / MAX_STARVATION
            for d in DIRECTIONS if d != (cd or 'N')
        )
        starvation = min(max_starv, 1.0)

        fast_switch = 1.0 if (switched and self._phase_tp < FAST_SWITCH_TP_THRESH) else 0.0

        raw = (
            + 4.0 * tp_norm
            - 3.5 * max_w_norm
            - 1.5 * avg_w_norm
            - 2.0 * max_q_norm
            - 2.5 * starvation
            + 6.0 * em_served
            - 1.2 * fast_switch
        )

        reward = float(np.clip(raw / 10.0, -1.5, 1.5))

        self._wait_buf.append(avg_w_norm * MAX_W)
        self.ep_reward += reward
        done = self.step_no >= MAX_STEPS

        info = {
            "phase":              self.phase_name,
            "queue_length":       sum(self._cq(d) for d in DIRECTIONS),
            "waiting_time":       round(sum(self._wait_buf) / max(len(self._wait_buf), 1), 2),
            "throughput":         self.total_tp,
            "time_in_phase":      MAX_GREEN - self.pm.phase_timer,
            "vehicles":           {d: self._cq(d) for d in DIRECTIONS},
            "dir_queues":         {d: self._cq(d) for d in DIRECTIONS},
            "starvation":         {d: self.served[d] for d in DIRECTIONS},
            "fairness":           round(self.pm.fairness_score(), 3),
            "phase_counts":       dict(self.pm.phase_counts),
            "decision_step":      decision_step,
            "next_valid_actions": self._get_next_valid_actions() if decision_step else None,
        }
        return self._state(), reward, done, info