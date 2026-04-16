"""
State Key Design (128 states = 4x4x4x2):
   ------------------------------------------------------------------
  | max_valid_q_dir (0-3)  argmax queue among NON-BLOCKED dirs       |
  | max_wait_level  (0-3)  bucket of worst single-direction wait     |
  | current_phase   (0-3)  direction currently green (= blocked)     |
  | emerg           (0-1)  any emergency vehicle present             |
   ------------------------------------------------------------------
  Why exclude the blocked direction from max_valid_q_dir?
    The fairness history (HISTORY_LEN=1) forbids repeating the last green.
    If the state key included the blocked direction in argmax, many states
    would recommend an action the agent cannot take. Q-values for those
    states never converge correctly.
"""

import numpy as np
import random
import os
import pickle
from collections import defaultdict

DIRECTIONS  = ['N', 'E', 'S', 'W']
DIR_INDEX   = {d: i for i, d in enumerate(DIRECTIONS)}
NUM_ACTIONS = 4
EFF_ACTIONS = 4

ACTIONS = [(d, 8) for d in DIRECTIONS]

# -- Hyperparameters ---------------------------------------------------
ALPHA_INIT  = 0.15
ALPHA_MIN   = 0.02
ALPHA_DECAY = 0.9995
GAMMA       = 0.95
EPS_START   = 1.0
EPS_MIN     = 0.08
EPS_DECAY   = 0.985


# ═══════════════════════════════════════
#  STATE DISCRETISATION — 128 states
# ═══════════════════════════════════════

def _bucket4(x: float) -> int:
    return min(3, int(x * 4)) # Map [0,1] → {0,1,2,3} uniformly.


def _disc(state: np.ndarray) -> tuple:
    qs = state[0:4]
    ws = state[4:8]

    current_phase = int(round(float(state[10]) * 3)) % 4

    valid_qs    = [(float(qs[i]), i) for i in range(4) if i != current_phase]
    max_valid_q = max(valid_qs, key=lambda x: x[0])[1]
    max_wait_lvl = _bucket4(float(np.max(ws)))
    emerg = 1 if float(state[9]) > 0.5 else 0

    return (max_valid_q, max_wait_lvl, current_phase, emerg)


def _explore_weights(state: np.ndarray, valid_actions: list) -> np.ndarray:
    qs = np.array(state[0:4], dtype=np.float64)
    ss = np.array(
        state[12:16] if len(state) >= 16 else np.zeros(4),
        dtype=np.float64
    )
    w       = qs * 0.6 + ss * 0.4 + 0.05   # always positive
    valid_w = np.array([w[a] for a in valid_actions], dtype=np.float64)
    total   = valid_w.sum()
    return valid_w / total if total > 0 else np.ones(len(valid_actions)) / len(valid_actions)


# ═══════════════════════════════════════
#  BASE AGENT
# ═══════════════════════════════════════

class BaseAgent:
    name = "BaseAgent"

    def __init__(self, state_size: int, action_size: int):
        self.state_size  = state_size
        self.action_size = EFF_ACTIONS
        self.epsilon     = EPS_START
        self.alpha       = ALPHA_INIT
        self._ep         = 0
        self.ep_aligned  = 0
        self.ep_steps    = 0

    def select_action(self, state, valid_actions=None) -> int:
        raise NotImplementedError

    def learn(self, s, a, r, s2, done): raise NotImplementedError

    def _record_alignment(self, state: np.ndarray, action: int):
        ds          = _disc(state)
        self.ep_steps   += 1
        self.ep_aligned += int(action == ds[0])

    def preference_rate(self) -> float:
        if self.ep_steps == 0: return 0.0
        return round(self.ep_aligned / self.ep_steps * 100, 1)

    def reset_ep_stats(self):
        self.ep_aligned = 0
        self.ep_steps   = 0

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def decay_alpha(self):
        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)
        self._ep  += 1

    def q_table_size(self) -> int:
        return len(getattr(self, '_q', {}))

    def save(self, path: str = ""): pass
    def load(self, path: str = ""): pass


# ═══════════════════════════════════════
#  FIXED-TIME BASELINE
# ═══════════════════════════════════════

class FixedTimeAgent(BaseAgent):
    name = "FixedTime"

    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self.epsilon    = 0.0
        self._phase_idx = 0

    def select_action(self, state, valid_actions=None) -> int:
        if not valid_actions: valid_actions = list(range(4))
        action = valid_actions[self._phase_idx % len(valid_actions)]
        self._phase_idx += 1
        self._record_alignment(state, action)
        return action

    def learn(self, s, a, r, s2, done): pass
    def decay_epsilon(self): pass
    def decay_alpha(self):   pass

    def save(self, path: str = "models/fixedtime.pkl"):
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"type": "FixedTime"}, f)

    def load(self, path: str = "models/fixedtime.pkl"): pass


# ═══════════════════════════════════════
#  Q-LEARNING  (off-policy TD(0))
# ═══════════════════════════════════════

class QLearningAgent(BaseAgent):
    """
    Off-policy TD(0) Q-Learning.
    Update rule:
      Q[s][a] ← Q[s][a] + α x (r + γ x max_{a'} Q[s'] - Q[s][a])
    """
    name    = "Q-Learning"
    BONUS_C = 0.08

    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self._q:      dict = {}
        self._visits: dict = defaultdict(lambda: np.zeros(EFF_ACTIONS))

    def _qv(self, ds: tuple) -> np.ndarray:
        if ds not in self._q:
            q = np.zeros(EFF_ACTIONS, dtype=np.float64)
            q[ds[0]] = 0.15   # optimistic init → explore max-queue direction first
            self._q[ds] = q
        return self._q[ds]

    def _augmented_q(self, ds: tuple) -> np.ndarray:
        return self._qv(ds) + self.BONUS_C / np.sqrt(self._visits[ds] + 1)

    def select_action(self, state, valid_actions=None) -> int:
        if not valid_actions: valid_actions = list(range(EFF_ACTIONS))
        if random.random() < self.epsilon:
            p      = _explore_weights(state, valid_actions)
            action = int(np.random.choice(valid_actions, p=p))
        else:
            ds     = _disc(state)
            q      = self._augmented_q(ds)
            action = max(valid_actions, key=lambda a: q[a])
        self._record_alignment(state, action)
        return action

    def learn(self, s, a, r, s2, done):
        ds, ds2 = _disc(s), _disc(s2)
        self._visits[ds][a] += 1
        q      = self._qv(ds)
        target = r + (0.0 if done else GAMMA * np.max(self._qv(ds2)))
        q[a]  += self.alpha * (target - q[a])

    def decay_epsilon(self):
        super().decay_epsilon()
        self.decay_alpha()

    def save(self, path: str = "models/qlearning.pkl"):
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q": self._q, "v": dict(self._visits),
                         "epsilon": self.epsilon, "alpha": self.alpha}, f)

    def load(self, path: str = "models/qlearning.pkl"):
        if not os.path.exists(path): return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "q" in obj:
                self._q      = obj["q"]
                self._visits = defaultdict(lambda: np.zeros(EFF_ACTIONS), obj.get("v", {}))
            else:
                self._q = obj
            self.epsilon = EPS_MIN
            self.alpha   = ALPHA_MIN
        except Exception as e:
            pass


# ═══════════════════════════════════════
#  SARSA  (on-policy TD(0))
# ═══════════════════════════════════════

class SARSAAgent(BaseAgent):
    """
    On-policy TD(0) SARSA.
    Update rule:
      Q[s][a] ← Q[s][a] + α x (r + γ x Q[s'][a'] - Q[s][a])
    """
    name    = "SARSA"
    BONUS_C = 0.08

    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self._q:      dict = {}
        self._visits: dict = defaultdict(lambda: np.zeros(EFF_ACTIONS))
        self._next_a: int  = None

    def _qv(self, ds: tuple) -> np.ndarray:
        if ds not in self._q:
            q = np.zeros(EFF_ACTIONS, dtype=np.float64)
            q[ds[0]] = 0.15
            self._q[ds] = q
        return self._q[ds]

    def _augmented_q(self, ds: tuple) -> np.ndarray:
        return self._qv(ds) + self.BONUS_C / np.sqrt(self._visits[ds] + 1)

    def _policy(self, state, valid_actions: list) -> int:
        if not valid_actions: valid_actions = list(range(EFF_ACTIONS))
        if random.random() < self.epsilon:
            p = _explore_weights(state, valid_actions)
            return int(np.random.choice(valid_actions, p=p))
        ds = _disc(state)
        q  = self._augmented_q(ds)
        return max(valid_actions, key=lambda a: q[a])

    def select_action(self, state, valid_actions=None) -> int:
        if not valid_actions: valid_actions = list(range(EFF_ACTIONS))
        if self._next_a is not None and self._next_a in valid_actions:
            a, self._next_a = self._next_a, None
        else:
            self._next_a = None
            a = self._policy(state, valid_actions)
        self._record_alignment(state, a)
        return a

    def learn(self, s, a, r, s2, done, valid_actions_s2=None):
        if not valid_actions_s2: valid_actions_s2 = list(range(EFF_ACTIONS))
        ds, ds2 = _disc(s), _disc(s2)
        self._visits[ds][a] += 1
        a2           = self._policy(s2, valid_actions_s2)
        self._next_a = a2
        q            = self._qv(ds)
        q2_val       = 0.0 if done else self._qv(ds2)[a2]
        q[a]        += self.alpha * (r + GAMMA * q2_val - q[a])

    def decay_epsilon(self):
        super().decay_epsilon()
        self.decay_alpha()

    def save(self, path: str = "models/sarsa.pkl"):
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q": self._q, "v": dict(self._visits),
                         "epsilon": self.epsilon, "alpha": self.alpha}, f)

    def load(self, path: str = "models/sarsa.pkl"):
        if not os.path.exists(path): return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "q" in obj:
                self._q      = obj["q"]
                self._visits = defaultdict(lambda: np.zeros(EFF_ACTIONS), obj.get("v", {}))
            else:
                self._q = obj
            self.epsilon = EPS_MIN
            self.alpha   = ALPHA_MIN
            self._next_a = None
        except Exception as e:
            pass


# ═══════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════

_MAP = {
    "qlearning":  QLearningAgent,
    "q-learning": QLearningAgent,
    "q_learning": QLearningAgent,
    "sarsa":      SARSAAgent,
    "fixedtime":  FixedTimeAgent,
    "fixed":      FixedTimeAgent,
    "baseline":   FixedTimeAgent,
}


def make_agent(name: str, state_size: int, action_size: int) -> BaseAgent:
    cls = _MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown agent '{name}'. Choose from: {list(_MAP.keys())}")
    return cls(state_size, action_size)