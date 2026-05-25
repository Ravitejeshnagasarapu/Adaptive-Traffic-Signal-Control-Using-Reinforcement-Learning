import os
import json
import math
from collections import deque


class MetricsLogger:
    def __init__(self, algo: str, output_root: str = "outputs"):
        self.algo     = algo
        self.save_dir = os.path.join(output_root, algo)
        self.data: dict = {
            "algo": algo,
            "episodes": [], "reward": [], "wait": [],
            "queue": [], "epsilon": [], "throughput": [],
            "preference_rate": [], "action_dist": [],
            "reward_smooth": [], "wait_smooth": [],
        }
        self._r_buf      = deque(maxlen=20)
        self._w_buf      = deque(maxlen=20)
        self._best_wait   = float("inf")
        self._best_reward = float("-inf")
        self._best_ep     = 0

    def log(self, episode, reward, wait, queue, epsilon,
            throughput=0, preference_rate=0.0, action_dist=None,
            dir_queues=None, phase_counts=None):
        self._r_buf.append(reward)
        self._w_buf.append(wait)
        rs = round(sum(self._r_buf) / len(self._r_buf), 4)
        ws = round(sum(self._w_buf) / len(self._w_buf), 2)

        self.data["episodes"].append(episode)
        self.data["reward"].append(round(reward, 4))
        self.data["wait"].append(round(wait, 2))
        self.data["queue"].append(queue)
        self.data["epsilon"].append(round(epsilon, 4))
        self.data["throughput"].append(throughput)
        self.data["preference_rate"].append(round(preference_rate, 1))
        self.data["action_dist"].append(action_dist or [25, 25, 25, 25])
        self.data["reward_smooth"].append(rs)
        self.data["wait_smooth"].append(ws)

        if dir_queues:
            for d, v in dir_queues.items():
                key = f"dir_queue_{d}"
                if key not in self.data: self.data[key] = []
                self.data[key].append(v)

        if phase_counts:
            for d, v in phase_counts.items():
                key = f"phase_{d}"
                if key not in self.data: self.data[key] = []
                self.data[key].append(v)

        if ws < self._best_wait or (ws == self._best_wait and rs > self._best_reward):
            self._best_wait   = ws
            self._best_reward = rs
            self._best_ep     = episode

    def save(self, save_dir=None):
        d = save_dir or self.save_dir
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        return path

    def summary(self) -> dict:
        n = min(30, len(self.data["reward_smooth"]))
        if n == 0: return {}
        waits = self.data["wait"]
        prefs = self.data["preference_rate"]

        last_w = waits[-30:]
        mw     = sum(last_w) / len(last_w)
        std_w  = math.sqrt(sum((v - mw)**2 for v in last_w) / len(last_w))

        p_first = sum(prefs[:50]) / 50  if len(prefs) >= 50 else 0
        p_last  = sum(prefs[-50:]) / 50 if len(prefs) >= 50 else 0

        return {
            "algo":              self.algo,
            "total_episodes":    len(self.data["episodes"]),
            "best_ep":           self._best_ep,
            "best_wait":         self._best_wait,
            "best_reward":       self._best_reward,
            "last30_avg_wait":   round(mw, 2),
            "last30_std_wait":   round(std_w, 2),
            "last30_avg_reward": round(sum(self.data["reward_smooth"][-n:]) / n, 4),
            "pref_first50":      round(p_first, 1),
            "pref_last50":       round(p_last, 1),
            "pref_improvement":  round(p_last - p_first, 1),
        }