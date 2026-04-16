# 🚦 Adaptive Traffic Signal Control Using Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)](https://numpy.org/)
[![WebSockets](https://img.shields.io/badge/WebSockets-asyncio-green)](https://websockets.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-orange)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

> Reinforcement Learning Project — Department of Computer Science and Engineering, IIITDM Kancheepuram

---

## 📖 Description

This project implements an **adaptive traffic signal control system** at a four-way intersection using model-free Reinforcement Learning. The problem is formulated as a **Markov Decision Process (MDP)**, where an intelligent agent observes real-time traffic state (queue lengths, wait times, starvation, emergency flags), selects a signal phase (N/E/S/W), and receives a shaped reward that incentivises throughput, minimises waiting time, and prevents lane starvation.

Two tabular RL algorithms — **Q-Learning** (off-policy) and **SARSA** (on-policy) — are implemented and compared against a **Fixed-Time baseline**. A live WebSocket server streams simulation state to a browser frontend, and a full metrics/plotting pipeline records every episode for post-hoc analysis.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Three control strategies** | Q-Learning, SARSA, Fixed-Time baseline |
| **128-state discrete MDP** | Compact state key `(max_valid_q_dir, max_wait_level, current_phase, emerg)` — 4×4×4×2 |
| **Dynamic green duration** | Phase length computed from queue depth: `BASE_GREEN + DYN_K × queue_len`, clamped to `[5, 20]` steps |
| **Emergency vehicle override** | Hard safety override (not reward-weighted) — emergency direction always served first |
| **Starvation prevention** | Per-direction starvation counters penalise neglected lanes in the reward |
| **Fairness tracking** | `fairness_score()` based on phase-count coefficient of variation, logged every episode |
| **Fairness history constraint** | `HISTORY_LEN = 1` blocks consecutive repeats of the same phase |
| **UCB-style exploration bonus** | Augmented Q-value `Q[s][a] + BONUS_C / √(visits[s][a] + 1)` for count-based exploration |
| **Weighted ε-greedy exploration** | Exploration probability weighted by queue length and starvation counters |
| **Three operating modes** | `train`, `test`, `compare` |
| **WebSocket live broadcast** | Async server pushes state to all connected browser clients at configurable FPS |
| **Metrics logging** | Per-episode reward, wait time, queue length, throughput, ε, preference rate, action distribution |
| **Automated plotting** | Per-algorithm dashboards + multi-algorithm comparison curves and bar charts |
| **Model persistence** | Best model (wait-time threshold), final model, and interrupt checkpoint saved separately |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Core language | Python 3.11 |
| Numerical computation | NumPy |
| Async server / WebSocket | `asyncio`, `websockets` |
| Visualisation | Matplotlib |
| Serialisation | `pickle` (model), `json` (metrics) |
| Frontend communication | WebSocket (JSON frames) |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        server.py                            │
│                                                             │
│   ┌──────────┐   action    ┌──────────────────────────┐    │
│   │  Agent   │ ─────────► │     TrafficEnv           │    │
│   │          │             │  ┌────────────────────┐  │    │
│   │ Q-Learn  │ ◄─────────  │  │   PhaseManager     │  │    │
│   │  SARSA   │  state,     │  │  (yellow/all-red/  │  │    │
│   │ FixedTime│  reward,    │  │   green FSM)       │  │    │
│   └──────────┘  done,info  │  └────────────────────┘  │    │
│         │                  └──────────────────────────┘    │
│         │ learn()                                           │
│         ▼                                                   │
│   ┌──────────┐   JSON frame   ┌────────────────────────┐   │
│   │ Q-Table  │               │   WebSocket Clients    │   │
│   │ (pickle) │ ─────────────► │   (browser frontend)  │   │
│   └──────────┘                └────────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  MetricsLogger  →  metrics.json  →  Plotter          │  │
│   │  (reward, wait, queue, throughput, fairness, ε, …)   │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Decision flow per step:**
1. Environment spawns vehicles stochastically (straight: 10%, right-turn: 7%, emergency: 1.2% per direction per step).  
2. Agent selects action (direction to give green) via ε-greedy or greedy policy.  
3. `PhaseManager` triggers YELLOW (2 steps) → ALL_RED (2 steps) → GREEN transition.  
4. Vehicles are cleared during GREEN; wait counters and starvation counters are updated.  
5. Shaped reward is computed and the agent's Q-values are updated.  
6. State, reward, and info dict are broadcast over WebSocket.

---

## 🌍 Environment Design

### State Vector — 17 features

```python
state = [
    # [0:4]   Normalised queue lengths  (N, E, S, W)  → each ∈ [0, 1], MAX_Q = 15
    # [4:8]   Normalised wait times     (N, E, S, W)  → each ∈ [0, 1], MAX_W = 90
    # [8]     Average normalised wait across all directions
    # [9]     Emergency flag            (binary 0/1)
    # [10]    Current phase             (0..1 encoded, 4 directions)
    # [11]    Time remaining in phase   (0..1, MAX_GREEN = 20)
    # [12:16] Starvation counters       (N, E, S, W)  → each ∈ [0, 1], MAX_STARVATION = 40
    # [16]    Last phase duration       (normalised)
]
```

### Discrete State Key — 128 states

The full 17-feature vector is compressed to a 4-tuple for the Q-table:

```
(max_valid_q_dir, max_wait_level, current_phase, emerg)
 └── 0-3           └── 0-3          └── 0-3        └── 0-1
```

`max_valid_q_dir` is the argmax queue **excluding the currently blocked direction** — this prevents the agent from learning Q-values for actions it is forbidden to take by the fairness history constraint.

### Actions

| Index | Direction |
|---|---|
| 0 | North |
| 1 | East |
| 2 | South |
| 3 | West |

Valid actions at each decision point exclude any direction in `PhaseManager._history` (length 1), preventing back-to-back repeats.

### Reward Function

```python
raw = (
    + 4.0 * tp_norm        # throughput incentive  (tp / 7.0, clipped to 1)
    - 3.5 * max_w_norm     # worst-direction wait penalty
    - 1.5 * avg_w_norm     # average wait penalty
    - 2.0 * max_q_norm     # worst-direction queue penalty
    - 2.5 * starvation     # starvation penalty (non-green directions)
    + 6.0 * em_served      # emergency vehicle cleared
    - 1.2 * fast_switch    # penalty for switching before clearing any vehicles
)

reward = clip(raw / 10.0, -1.5, 1.5)
```

`fast_switch` fires when the agent switches phase while `_phase_tp < 1` (no vehicles cleared during current phase).

### Phase Manager

- **Minimum green:** 5 steps  
- **Maximum green:** 20 steps  
- **Dynamic duration:** `max(5, min(20, int(6 + 0.6 × queue_len)))`  
- **Transition:** YELLOW (2 steps) → ALL_RED (2 steps) → GREEN  
- **Fairness score:** `max(0, 1 − CV)` where CV = std/mean of per-direction phase counts

---

## 🤖 Algorithms

### Q-Learning (Off-Policy TD(0))

```
Q[s][a] ← Q[s][a] + α × (r + γ × max_{a'} Q[s'] − Q[s][a])
```

- Learns the **optimal** value function independent of the behaviour policy.  
- Uses the **greedy** bootstrap target even during ε-greedy exploration.  
- UCB bonus added at action selection: `Q_aug[s][a] = Q[s][a] + 0.08 / √(visits[s][a] + 1)`  
- Optimistic initialisation: `Q[s][max_valid_q_dir] = 0.15` on first visit.

### SARSA (On-Policy TD(0))

```
Q[s][a] ← Q[s][a] + α × (r + γ × Q[s'][a'] − Q[s][a])
```

- Updates using the **action actually taken** by the current ε-greedy policy.  
- `a'` is selected by `_policy()` immediately after learning and cached as `_next_a` so `select_action` returns it on the next call — ensuring true on-policy consistency.  
- Same UCB bonus and optimistic initialisation as Q-Learning.

### Fixed-Time Baseline

- Cycles through valid directions in round-robin order.  
- No learning; epsilon fixed at 0.  
- Used purely as a performance lower-bound reference.

---

## 📁 Project Structure

```
.
├── server.py               # Main entry point — training loop, WebSocket server, modes
├── rl_agents.py            # QLearningAgent, SARSAAgent, FixedTimeAgent, make_agent()
├── traffic_env.py          # TrafficEnv — step(), reset(), state vector, reward
├── plotter.py              # Plotter, AdvancedPlotter — all matplotlib graphs
├── scheduler/
│   └── phase_manager.py    # PhaseManager — FSM, dynamic duration, fairness score
├── analytics/              # MetricsLogger (referenced in server.py)
│
├── models/                 # Saved model checkpoints (auto-created)
│   ├── qlearning.pkl       # Final Q-Learning model
│   ├── sarsa.pkl           # Final SARSA model
│   ├── best_qlearning.pkl  # Best Q-Learning model (by avg wait time)
│   ├── best_sarsa.pkl      # Best SARSA model
│   └── interrupted_*.pkl   # Auto-saved on Ctrl-C
│
├── outputs/                # Training artefacts (auto-created)
│   ├── qlearning/
│   │   ├── metrics.json
│   │   ├── reward.png
│   │   ├── wait.png
│   │   ├── queue.png
│   │   ├── throughput.png
│   │   └── dashboard.png
│   ├── sarsa/              # (same structure)
│   ├── fixedtime/          # (same structure)
│   └── compare/
│       ├── reward_compare.png
│       ├── wait_compare.png
│       ├── throughput_compare.png
│       ├── queue_compare.png
│       ├── comparison_bar.png
│       └── comparison_curves.png
│
└── index.html              # Browser frontend (connects via WebSocket)
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11
- pip

### Install dependencies

```bash
pip install numpy matplotlib websockets
```

---

## 🚀 Running the Project

### Training Mode

Train a specific algorithm for N episodes:

```bash
# Train Q-Learning
python server.py --mode train --algo qlearning --episodes 1200

# Train SARSA
python server.py --mode train --algo sarsa --episodes 1200

# Run Fixed-Time baseline
python server.py --mode train --algo fixedtime --episodes 1200
```

### Testing Mode

Evaluate a saved model (loads `models/best_<algo>.pkl` or `models/<algo>.pkl`):

```bash
python server.py --mode test --algo qlearning --episodes 50 --fps 25.0
python server.py --mode test --algo sarsa     --episodes 50 --fps 25.0
```

### Comparison Mode

Run all three algorithms sequentially and generate comparison graphs:

```bash
python server.py --mode compare --episodes 1200
```

Results and graphs are written to `outputs/compare/`.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `localhost` | WebSocket host |
| `--port` | `8765` | WebSocket port |
| `--fps` | `25.0` | Playback speed in test/compare mode |

### Frontend (live visualisation)

```bash
# In a separate terminal, serve the frontend
python -m http.server 5000
```

Open `http://localhost:5000` in a browser. The page connects to `ws://localhost:8765` and displays live signal state, metrics, and rolling graphs.

---

## 📊 Outputs & Metrics

### `metrics.json` (per algorithm)

Saved to `outputs/<algo>/metrics.json` every 100 episodes and at run end.

```json
{
  "algo": "qlearning",
  "episodes":        [...],
  "reward":          [...],
  "reward_smooth":   [...],
  "wait":            [...],
  "wait_smooth":     [...],
  "queue":           [...],
  "throughput":      [...],
  "epsilon":         [...],
  "preference_rate": [...],
  "action_dist":     [[N%, E%, S%, W%], ...]
}
```

### Generated plots

| Plot | Description |
|---|---|
| `reward.png` | Raw + smoothed (15-ep MA) episode reward |
| `wait.png` | Raw + smoothed average wait time per episode |
| `queue.png` | Raw + smoothed queue length |
| `throughput.png` | Vehicles cleared per episode |
| `dashboard.png` | 4-panel dashboard: reward, wait, queue, ε |
| `reward_compare.png` | Smoothed reward — all three algorithms |
| `wait_compare.png` | Smoothed wait time — all three algorithms |
| `throughput_compare.png` | Throughput — all three algorithms |
| `queue_compare.png` | Queue length — all three algorithms |
| `comparison_bar.png` | Bar chart: avg wait + throughput with % vs FixedTime |
| `comparison_curves.png` | 3-panel overview from `metrics.json` |

---

## 🎛 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ALPHA_INIT` | 0.15 | Initial learning rate |
| `ALPHA_MIN` | 0.02 | Minimum learning rate |
| `ALPHA_DECAY` | 0.9995 | Per-episode multiplicative decay |
| `GAMMA` | 0.95 | Discount factor |
| `EPS_START` | 1.0 | Initial exploration rate |
| `EPS_MIN` | 0.08 | Minimum exploration rate |
| `EPS_DECAY` | 0.985 | Per-episode multiplicative decay |
| `BONUS_C` | 0.08 | UCB exploration bonus coefficient |
| `MIN_GREEN` | 5 | Minimum green phase duration (steps) |
| `MAX_GREEN` | 20 | Maximum green phase duration (steps) |
| `BASE_GREEN` | 6 | Base green duration before queue scaling |
| `DYN_K` | 0.6 | Queue scaling factor for dynamic duration |
| `YELLOW_STEPS` | 2 | Yellow transition length |
| `ALL_RED_STEPS` | 2 | All-red clearance length |
| `HISTORY_LEN` | 1 | Fairness history — blocks last N phases from repeating |
| `MAX_STEPS` | 200 | Steps per episode |
| `SPAWN_STRAIGHT` | 0.10 | Per-direction straight-vehicle spawn probability |
| `SPAWN_RIGHT` | 0.07 | Per-direction right-turn spawn probability |
| `EMERGENCY_PROB` | 0.012 | Per-direction emergency vehicle probability |

---

## 📈 Evaluation Metrics

Models are evaluated on four primary metrics recorded at the end of every episode:

| Metric | Direction | Description |
|---|---|---|
| **Average Wait Time** (s) | ↓ lower is better | 30-step rolling mean of `avg_w_norm × MAX_W` |
| **Queue Length** (vehicles) | ↓ lower is better | Total vehicles queued across all four directions |
| **Throughput** (vehicles/ep) | ↑ higher is better | Cumulative vehicles cleared per episode |
| **Episode Reward** | ↑ higher is better | Sum of clipped per-step rewards |
| **Fairness Score** | ↑ closer to 1 is better | `max(0, 1 − CV)` of per-direction phase counts |
| **Preference Rate** (%) | — | % of decisions that matched the highest-queue non-blocked direction |

**Best model saving:** A new checkpoint (`models/best_<algo>.pkl`) is written only when the 20-episode smoothed wait time improves by ≥ 0.5 s, preventing noisy saves.

---

## 🖼 Results Summary

| Algorithm | Avg Reward | Avg Wait Time | Avg Throughput | Stability |
|---|---|---|---|---|
| Fixed-Time | -124.75 | 43.78 s | 100–120 veh/ep | Low |
| Q-Learning | -108.11 | 38.83 s | 90–120 veh/ep | Medium |
| **SARSA** | **-104.26** | **33.73 s** | **100–135 veh/ep** | **High** |

*Values from the project report (1200 training episodes, final 20-episode averages).*

**Key findings:**
- Both RL methods significantly outperform Fixed-Time control.
- SARSA's on-policy updates produce more stable convergence than Q-Learning's off-policy bootstrap.
- All three methods maintain a Jain Fairness Index consistently above 0.90.
- Q-Learning's preference alignment improves from 44% → 46% over training; SARSA's declines slightly (45% → 37%) but overall performance is superior — indicating SARSA diversifies beyond the greedy heuristic.

---

## 🖼 Visual Outputs

| Dashboard | Description |
|---|---|
| ![SARSA Dashboard](outputs/sarsa/dashboard.png) | SARSA training summary — reward, wait, queue, ε |
| ![Q-Learning Dashboard](outputs/qlearning/dashboard.png) | Q-Learning training summary |
| ![Comparison Bar](outputs/compare/comparison_bar.png) | Side-by-side wait time and throughput |
| ![Comparison Curves](outputs/compare/comparison_curves.png) | Smoothed training curves — all algorithms |

---

## 🔮 Future Improvements

- **Multi-intersection coordination** — extend the MDP to a network of intersections with shared state.
- **Deep Q-Network (DQN)** — replace the 128-state tabular Q-table with a neural network to handle continuous, high-dimensional state spaces.
- **Real sensor integration** — connect to live CCTV/IoT feeds instead of the stochastic simulation spawner.
- **Pedestrian phase modelling** — add pedestrian crossing requests as additional state features and actions.
- **Prioritised experience replay** — for DQN variant, weight updates towards high-TD-error transitions.
- **Multi-agent RL** — one independent agent per intersection with shared reward shaping for network-level optimisation.
- **Adaptive reward tuning** — use meta-learning or Bayesian optimisation to auto-tune the seven reward coefficients.

---

## 👥 Project Team

| Name | Roll Number |
|---|---|
| Y. Vamsi | CS23B2027 |
| P. Srikala | CS23B2049 |
| N. Ravi Tejesh | CS23B2051 |
| N. Durgalakshmi Prasad | CS23B2052 |

**Faculty Advisor:** Dr. Rahul Raman  
**Institution:** IIITDM Kancheepuram — Department of Computer Science and Engineering

---

## 📄 Reference

> - [Traffic Signal Control based on Markov Decision Process](https://www.sciencedirect.com/science/article/pii/S2405896316302075)

> - [Adaptive Traffic Control System using Reinforcement Learning](https://www.researchgate.net/publication/341876548_Adaptive_Traffic_Control_System_using_Reinforcement_Learning)
