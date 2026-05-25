import asyncio
import json
import argparse
import logging
import sys
import os
import warnings
from collections import deque

import websockets

from traffic_env import TrafficEnv, ACTIONS, NUM_ACTIONS
from rl_agents   import make_agent, EPS_MIN, DIRECTIONS, SARSAAgent
from analytics   import MetricsLogger, Plotter, AdvancedPlotter

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

CLIENTS: set       = set()
LATEST_FRAME: dict = {}

# -- Directory structure -----------------------------------------------
OUTPUT_ROOT  = "outputs"
MODEL_DIR    = "models"
COMPARE_DIR  = os.path.join(OUTPUT_ROOT, "compare")

METRICS_SAVE_EVERY = 100
MIN_SAVE_EPISODE   = 80

_active_agent = None
_active_algo  = ""


# --------------------------------------------
#  PATH HELPERS
# --------------------------------------------
def best_path(a):      return os.path.join(MODEL_DIR, f"best_{a}.pkl")
def default_path(a):   return os.path.join(MODEL_DIR, f"{a}.pkl")
def interrupt_path(a): return os.path.join(MODEL_DIR, f"interrupted_{a}.pkl")
def out_dir(a):        return os.path.join(OUTPUT_ROOT, a)


def _ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)
    for algo in ["qlearning", "sarsa", "fixedtime"]:
        os.makedirs(out_dir(algo), exist_ok=True)


# --------------------------------------------
#  WEBSOCKET
# --------------------------------------------
async def _broadcast(data: dict):
    global LATEST_FRAME
    LATEST_FRAME = data
    if not CLIENTS: return
    msg = json.dumps(data)
    await asyncio.gather(*[c.send(msg) for c in list(CLIENTS)],
                         return_exceptions=True)

async def _ws_handler(ws):
    CLIENTS.add(ws)
    log.info(f"Client connected ({len(CLIENTS)} active)")
    try:
        if LATEST_FRAME:
            await ws.send(json.dumps(LATEST_FRAME))
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        CLIENTS.discard(ws)
        log.info(f"Client disconnected ({len(CLIENTS)} active)")


# --------------------------------------------
#  HELPERS
# --------------------------------------------
def _fav_dir(ep_actions: list) -> str:
    return DIRECTIONS[ep_actions.index(max(ep_actions))]

def _ep_dist_str(ep_actions: list) -> str:
    tot = max(sum(ep_actions), 1)
    return "/".join(f"{ep_actions[i]/tot*100:.0f}" for i in range(4))

def _dir_dist_str(counts: list) -> str:
    tot = max(sum(counts), 1)
    return " | ".join(f"{DIRECTIONS[i]}:{counts[i]/tot*100:.1f}%" for i in range(4))


# --------------------------------------------
#  BEST MODEL TRACKER
# --------------------------------------------
class BestModelTracker:
    """
    Only saves a new best model when avg_wait improves by ≥ 0.5s.
    This prevents noisy episode-to-episode fluctuations from being
    saved as "best" when the underlying policy hasn't actually improved.
    """
    def __init__(self, algo, logger, plotter):
        self.algo        = algo
        self.logger      = logger
        self.plotter     = plotter
        self.best_wait   = float("inf")
        self.best_reward = float("-inf")
        self.best_ep     = 0
        self._loaded     = False

    def _try_load(self):
        if self._loaded: return
        self._loaded = True
        mf = os.path.join(out_dir(self.algo), "metrics.json")
        bp = best_path(self.algo)
        if os.path.exists(mf) and os.path.exists(bp):
            try:
                with open(mf) as f: data = json.load(f)
                ws = data.get("wait_smooth", [])
                rs = data.get("reward_smooth", [])
                if ws and rs:
                    self.best_wait   = min(ws)
                    self.best_reward = max(rs)
                    log.info(f" --> Resumed best: wait={self.best_wait:.2f}s  ← {bp}")
            except Exception as e:
                log.error(f"  Resume load failed: {e}")

    def update(self, agent, ep, avg_wait, avg_reward):
        if ep == MIN_SAVE_EPISODE: self._try_load()
        if ep < MIN_SAVE_EPISODE:
            return False

        is_best = avg_wait < self.best_wait - 0.5

        if is_best:
            self.best_wait   = avg_wait
            self.best_reward = avg_reward
            self.best_ep     = ep
            try:
                agent.save(best_path(self.algo))
                log.info(f"  🔥 NEW BEST  ep={ep}  wait={avg_wait:.2f}s  "
                         f"reward={avg_reward:.4f}  → {best_path(self.algo)}")
            except Exception as e:
                log.error(f"  Model save failed: {e}")
        return is_best

    def summary(self):
        if self.best_ep:
            log.info(f"  Best model: ep={self.best_ep}  wait={self.best_wait:.2f}s"
                     f"  --> {best_path(self.algo)}")
        else:
            log.info("  No best model saved (threshold not reached)")


def load_for_test(agent, algo):
    for path in [best_path(algo), default_path(algo)]:
        if os.path.exists(path):
            try:
                agent.load(path)
                log.info(f"  ✅ Loaded model ← {path}")
                return
            except Exception as e:
                log.error(f"  Load failed ({path}): {e}")
    log.warning(f"  ⚠️  No model found for '{algo}' — running untrained")


def on_interrupt():
    global _active_agent, _active_algo
    if _active_agent is None: return
    try:
        _active_agent.save(interrupt_path(_active_algo))
        log.info(f"  ⚠️  Interrupted → {interrupt_path(_active_algo)}")
    except Exception as e:
        log.error(f"  Interrupt save failed: {e}")


def _periodic_save(logger, plotter, algo, agent, ep):
    try:
        mp = logger.save(out_dir(algo))
        plotter.plot_algo(mp, out_dir(algo), agent.name)
        log.info(f"  --> ep {ep}: metrics saved  |  Q-table: {agent.q_table_size()} states")
    except Exception as e:
        log.error(f"  Periodic save failed: {e}")
    return os.path.join(out_dir(algo), "metrics.json")


# ═══════════════════════════════════════
#  CORE TRAINING LOOP
# ═══════════════════════════════════════
async def run_loop(algo, episodes, mode, fps=25.0,
                   reward_history=None, wait_history=None):
    global _active_agent, _active_algo

    rh = reward_history or []
    wh = wait_history   or []

    env     = TrafficEnv()
    agent   = make_agent(algo, env.state_size, env.action_size)
    logger  = MetricsLogger(algo, OUTPUT_ROOT)
    plotter = Plotter()

    _active_agent = agent
    _active_algo  = algo

    if mode == "test":
        agent.epsilon = EPS_MIN
        load_for_test(agent, algo)

    delay = 0.0 if mode == "train" else (1.0 / fps)

    tracker      = BestModelTracker(algo, logger, plotter)
    smooth_r     = deque(maxlen=20)
    smooth_w     = deque(maxlen=20)
    action_counts = [0] * NUM_ACTIONS

    log.info("═" * 68)
    log.info(f"  {mode.upper():8s}  |  {agent.name:12s}  |  {episodes} episodes")
    log.info(f"  Speed    : {'FULL (0ms delay)' if mode=='train' else f'{1000/fps:.0f}ms/step'}")
    log.info(f"  Models → {MODEL_DIR}/   |   Outputs → {out_dir(algo)}/")
    log.info(f"  ε: {agent.epsilon:.2f}→{EPS_MIN}  α: 0.15→0.02  γ=0.95  States=128")
    log.info("═" * 68)

    for ep in range(1, episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        done      = False
        last_info: dict = {}
        ep_actions      = [0] * NUM_ACTIONS
        agent.reset_ep_stats()

        initial_valid = env.get_initial_valid_actions()
        action        = agent.select_action(state, valid_actions=initial_valid)
        option_state  = state.copy()
        option_action = action
        option_reward = 0.0
        option_steps  = 0
        decisions     = 1
        current_valid = initial_valid

        while not done:
            s2, reward, done, info = env.step(action)

            option_reward += reward
            option_steps  += 1
            ep_reward     += reward
            last_info      = info
            state          = s2

            ep_actions[option_action]    += 1
            action_counts[option_action] += 1

            if info.get("decision_step", False):
                next_valid = info.get("next_valid_actions") or list(range(NUM_ACTIONS))

                if mode == "train" and option_state is not None:
                    norm_r = option_reward / max(option_steps, 1)
                    if isinstance(agent, SARSAAgent):
                        agent.learn(option_state, option_action, norm_r,
                                    s2, done, valid_actions_s2=next_valid)
                    else:
                        agent.learn(option_state, option_action, norm_r, s2, done)

                if not done:
                    action        = agent.select_action(s2, valid_actions=next_valid)
                    option_state  = s2.copy()
                    option_action = action
                    option_reward = 0.0
                    option_steps  = 0
                    current_valid = next_valid
                    decisions    += 1

            if CLIENTS and (mode == "test" or ep % 5 == 0):
                await _broadcast({
                    "phase":           info["phase"],
                    "reward":          round(reward, 4),
                    "waiting_time":    info["waiting_time"],
                    "queue_length":    info["queue_length"],
                    "throughput":      info["throughput"],
                    "epsilon":         round(agent.epsilon, 4),
                    "algorithm":       agent.name,
                    "mode":            mode,
                    "episode":         ep,
                    "max_episodes":    episodes,
                    "time_in_phase":   info["time_in_phase"],
                    "vehicles":        info["vehicles"],
                    "fairness":        info.get("fairness", 0.0),
                    "phase_counts":    info.get("phase_counts", {}),
                    "preference_rate": agent.preference_rate(),
                    "decisions":       decisions,
                    "graph":           {"reward": rh[-100:], "wait": wh[-100:]},
                })
            if delay > 0:
                await asyncio.sleep(delay)

        if mode == "train" and option_state is not None:
            norm_r = option_reward / max(option_steps, 1)
            if isinstance(agent, SARSAAgent):
                agent.learn(option_state, option_action, norm_r,
                            state, True, valid_actions_s2=current_valid)
            else:
                agent.learn(option_state, option_action, norm_r, state, True)

        smooth_r.append(ep_reward)
        smooth_w.append(last_info.get("waiting_time", 0.0))
        sr = round(sum(smooth_r) / len(smooth_r), 4)
        sw = round(sum(smooth_w) / len(smooth_w), 2)
        rh.append(sr)
        wh.append(sw)

        logger.log(
            episode         = ep,
            reward          = ep_reward,
            wait            = last_info.get("waiting_time", 0.0),
            queue           = last_info.get("queue_length", 0),
            epsilon         = agent.epsilon,
            throughput      = last_info.get("throughput", 0),
            preference_rate = agent.preference_rate(),
            action_dist     = [
                round(ep_actions[i] / max(sum(ep_actions), 1) * 100, 1)
                for i in range(4)
            ],
        )

        if mode in ("train", "compare"):
            agent.decay_epsilon()
            tracker.update(agent, ep, avg_wait=sw, avg_reward=sr)
            if ep % METRICS_SAVE_EVERY == 0:
                _periodic_save(logger, plotter, algo, agent, ep)

        if ep == 1:
            if abs(ep_reward) > 500:
                log.warning(f"  ⚠️  ep1 reward={ep_reward:.1f} — magnitude >500, check clipping")
            else:
                log.info(f"  ✅ Reward scale OK: ep1={ep_reward:.1f}")

        trend = ""
        if len(rh) >= 20:
            d     = rh[-1] - rh[-20]
            trend = f"  Δ20={d:+.4f}" + (" ✅" if d > 0 else "")

        log.info(
            f"  Ep {ep:4d}/{episodes}  "
            f"r={ep_reward:7.1f}  sm={sr:7.1f}  "
            f"wait={sw:5.1f}s  q={last_info.get('queue_length',0):3d}  "
            f"ε={agent.epsilon:.4f}  pref={agent.preference_rate():5.1f}%  "
            f"fair={last_info.get('fairness',0):.2f}  "
            f"dec={decisions:3d}  fav={_fav_dir(ep_actions)}  "
            f"act=[{_ep_dist_str(ep_actions)}]%"
            f"{trend}"
        )

    # -- Post-training -------------------------
    if mode == "train":
        try:
            agent.save(default_path(algo))
            log.info(f"  Final model → {default_path(algo)}")
        except Exception as e:
            log.error(f"  Final save failed: {e}")

        tracker.summary()

        try:
            mp = logger.save(out_dir(algo))
            plotter.plot_algo(mp, out_dir(algo), agent.name)
            ap = AdvancedPlotter()
            ap.plot_all(mp, out_dir(algo), agent.name)
        except Exception as e:
            log.error(f"  Plot generation failed: {e}")

        log.info(f"  Action dist: {_dir_dist_str(action_counts)}")
        summary = logger.summary()

        log.info("\n" + "═" * 60)
        log.info("  FINAL TRAINING SUMMARY")
        log.info("═" * 60)

        log.info(f"  Algorithm        : {summary['algo'].upper()}")
        log.info(f"  Total Episodes   : {summary['total_episodes']}")

        log.info("\n  BEST PERFORMANCE")
        log.info(f"    Episode        : {summary['best_ep']}")
        log.info(f"    Avg Wait       : {summary['best_wait']:.2f} sec")
        log.info(f"    Reward         : {summary['best_reward']:.2f}")

        log.info("\n  LAST 30 EPISODES (STABILITY)")
        log.info(f"    Avg Wait       : {summary['last30_avg_wait']:.2f} sec")
        log.info(f"    Std Deviation  : {summary['last30_std_wait']:.2f}")
        log.info(f"    Avg Reward     : {summary['last30_avg_reward']:.2f}")

        log.info("\n  LEARNING QUALITY")
        log.info(f"    Pref (first 50): {summary['pref_first50']:.1f}%")
        log.info(f"    Pref (last 50) : {summary['pref_last50']:.1f}%")
        log.info(f"    Improvement    : {summary['pref_improvement']:+.1f}%")
        
        wait = summary['last30_avg_wait']

        if 25 <= wait <= 35:
            log.info("\n  ✅ Performance: OPTIMAL (25-35 sec range achieved)")
        elif wait < 25:
            log.info("\n  ⚠️ Performance: Too low (may indicate unrealistic behavior)")
        else:
            log.info("\n  ⚠️ Performance: High wait time (needs improvement)")

        log.info("═" * 60 + "\n")
        log.info(f"  Graphs  → {out_dir(algo)}/")
        log.info(f"  Models  → {MODEL_DIR}/")

    return agent, rh, wh


# ═══════════════════════════════════════
#  COMPARE MODE  -  FixedTime → Q-Learning → SARSA
# ═══════════════════════════════════════
async def compare_loop(episodes, fps=35.0):
    algos   = ["fixedtime", "qlearning", "sarsa"]
    results = []
    plotter = Plotter()

    multi_graph = {"reward": {}, "wait": {}, "throughput": {}, "queue": {}}

    log.info("═" * 65)
    log.info("  COMPARE MODE: FixedTime vs Q-Learning vs SARSA")
    log.info(f"  {episodes} episodes per algorithm")
    log.info(f"  Graphs → {COMPARE_DIR}/")
    log.info("═" * 65)

    for algo in algos:
        env   = TrafficEnv()
        agent = make_agent(algo, env.state_size, env.action_size)
        if algo not in ("fixedtime", "fixed", "baseline"):
            load_for_test(agent, algo)
        agent.epsilon = EPS_MIN

        ep_rewards, ep_waits, ep_tputs, ep_queues = [], [], [], []
        all_rh = []
        smooth_r = deque(maxlen=10)
        smooth_w = deque(maxlen=10)

        log.info(f"\n  -- {agent.name}  ({episodes} eps) --")

        for ep in range(1, episodes + 1):
            state         = env.reset()
            ep_reward     = 0.0
            done          = False
            last_info     = {}
            initial_valid = env.get_initial_valid_actions()
            action        = agent.select_action(state, valid_actions=initial_valid)

            while not done:
                s2, reward, done, info = env.step(action)
                state     = s2
                ep_reward += reward
                last_info  = info
                if info.get("decision_step", False) and not done:
                    nv     = info.get("next_valid_actions") or list(range(NUM_ACTIONS))
                    action = agent.select_action(s2, valid_actions=nv)

            ep_rewards.append(ep_reward)
            ep_waits.append(last_info.get("waiting_time", 0.0))
            ep_tputs.append(last_info.get("throughput", 0))
            ep_queues.append(last_info.get("queue_length", 0))
            smooth_r.append(ep_reward)
            smooth_w.append(last_info.get("waiting_time", 0.0))
            all_rh.append(round(sum(smooth_r) / len(smooth_r), 4))

            if ep % 10 == 0 or ep <= 3:
                await _broadcast({
                    "phase": last_info.get("phase", "IDLE"),
                    "reward": round(ep_reward, 4),
                    "waiting_time": last_info.get("waiting_time", 0),
                    "queue_length": last_info.get("queue_length", 0),
                    "throughput": last_info.get("throughput", 0),
                    "epsilon": round(agent.epsilon, 4),
                    "algorithm": agent.name, "mode": "compare",
                    "episode": ep, "max_episodes": episodes,
                    "time_in_phase": 0,
                    "vehicles": last_info.get("vehicles", {}),
                    "graph": {"reward": all_rh[-100:], "wait": ep_waits[-100:]},
                })
                await asyncio.sleep(0)

                log.info(
                    f"  [{agent.name:12s}]  ep {ep:4d}/{episodes}  "
                    f"r={ep_reward:7.1f}  "
                    f"wait={last_info.get('waiting_time',0):5.1f}s  "
                    f"q={last_info.get('queue_length',0):3d}  "
                    f"tp={last_info.get('throughput',0):4d}"
                )

        n     = min(20, len(ep_rewards))
        avg_r = round(sum(ep_rewards[-n:]) / n, 2)
        avg_w = round(sum(ep_waits[-n:])   / n, 2)
        avg_t = round(sum(ep_tputs[-n:])   / n, 1)
        avg_q = round(sum(ep_queues[-n:])  / n, 1)

        results.append({
            "algorithm": agent.name,
            "avg_wait":  avg_w,
            "throughput": avg_t,
            "reward":    avg_r,
            "avg_queue": avg_q,
        })

        multi_graph["reward"][agent.name]     = ep_rewards
        multi_graph["wait"][agent.name]       = ep_waits
        multi_graph["throughput"][agent.name] = ep_tputs
        multi_graph["queue"][agent.name]      = ep_queues

        log.info(f"  ✔ {agent.name:12s}  wait={avg_w}s  tp={avg_t}  r={avg_r}")

    log.info(f"\n  Generating comparison graphs → {COMPARE_DIR}/")
    try:
        algo_dirs = [out_dir(a) for a in ["qlearning", "sarsa", "fixedtime"]
                     if os.path.exists(os.path.join(out_dir(a), "metrics.json"))]
        plotter.plot_comparison(
            algo_dirs       = algo_dirs,
            comparison_data = results,
            multi_graph     = multi_graph,
            compare_dir     = COMPARE_DIR,
        )
    except Exception as e:
        log.error(f"  Comparison plot failed: {e}")

    # -- Results table ------------------------
    baseline_w = next((r["avg_wait"] for r in results if r["algorithm"] == "FixedTime"), None)

    log.info("\n" + "═" * 65)
    log.info("  COMPARISON RESULTS")
    log.info("═" * 65)
    log.info(f"  {'Algorithm':14s}  {'Avg Wait':10s}  {'Throughput':12s}  {'Reward':10s}")
    log.info("  " + "-" * 55)
    for r in results:
        pct = ""
        if baseline_w and r["algorithm"] != "FixedTime":
            impv = (baseline_w - r["avg_wait"]) / baseline_w * 100
            pct  = f"   ({impv:+.1f}% vs FixedTime)"
        log.info(f"  {r['algorithm']:14s}  {r['avg_wait']:8.2f}s  "
                 f"{r['throughput']:10.1f}  {r['reward']:10.2f}{pct}")

    # -- FINAL RESULT box --------------------
    rl_results = [r for r in results if r["algorithm"] != "FixedTime"]
    if rl_results:
        winner = min(rl_results, key=lambda x: x["avg_wait"])
        runner = max(rl_results, key=lambda x: x["avg_wait"])
        impv   = round((baseline_w - winner["avg_wait"]) / baseline_w * 100, 1) if baseline_w else 0

        log.info("\n" + "═" * 50)
        log.info("  FINAL RESULT")
        log.info("═" * 50)
        log.info(f"  Best Model     : {winner['algorithm']}")
        log.info(f"  Avg Wait       : {winner['avg_wait']} sec")
        log.info(f"  Reward         : {winner['reward']}")
        log.info(f"  Throughput     : {winner['throughput']}")
        log.info(f"  Improvement    : +{impv}% vs FixedTime")
        log.info(f"  Runner-up      : {runner['algorithm']} ({runner['avg_wait']}s)")
        log.info(f"  Model path     : {best_path(winner['algorithm'].lower().replace('-','').replace(' ',''))}")
        log.info("═" * 50)
        log.info(f"\n  All graphs → {COMPARE_DIR}/")

    await _broadcast({
        "phase": "ALL_RED", "reward": 0.0, "waiting_time": 0.0,
        "queue_length": 0, "throughput": 0, "epsilon": 0.0,
        "algorithm": "COMPARE COMPLETE", "mode": "compare",
        "episode": episodes, "max_episodes": episodes, "time_in_phase": 0,
        "vehicles": {d: 0 for d in DIRECTIONS},
        "graph": {"reward": [], "wait": []},
        "comparison": results,
    })
    log.info("Comparison complete.")


# ═══════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════
def _parse():
    p = argparse.ArgumentParser(description="RL Traffic Signal Control — Final Submission")
    p.add_argument("--mode",     choices=["train","test","compare"], default="train")
    p.add_argument("--algo",     choices=["qlearning","sarsa","fixedtime"], default="qlearning")
    p.add_argument("--episodes", type=int,   default=600)
    p.add_argument("--host",     default="localhost")
    p.add_argument("--port",     type=int,   default=8765)
    p.add_argument("--fps",      type=float, default=25.0)
    return p.parse_args()


async def _main():
    args = _parse()
    _ensure_dirs()

    log.info("═" * 65)
    log.info(f"  RL Traffic Signal Control — Final Submission")
    log.info(f"  WebSocket  ws://{args.host}:{args.port}")
    log.info(f"  Mode       {args.mode.upper()}")
    if args.mode != "compare":
        log.info(f"  Algorithm  {args.algo.upper()}")
    log.info(f"  Episodes   {args.episodes}   FPS={args.fps}")
    log.info(f"  Models  → {MODEL_DIR}/   |   Outputs → {OUTPUT_ROOT}/")
    log.info("═" * 65)

    async with websockets.serve(_ws_handler, args.host, args.port):
        if args.mode == "compare":
            await compare_loop(episodes=args.episodes, fps=args.fps)
        else:
            await run_loop(algo=args.algo, episodes=args.episodes,
                          mode=args.mode, fps=args.fps)
        log.info("Done. Server listening — Ctrl+C to stop.")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        on_interrupt()
        log.info("Server stopped.")
        sys.exit(0)