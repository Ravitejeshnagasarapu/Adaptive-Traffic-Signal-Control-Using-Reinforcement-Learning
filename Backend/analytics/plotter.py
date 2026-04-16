import os
import json
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False

ALGO_COLORS = {
    "Q-Learning": "#2196F3",
    "SARSA":      "#FF9800",
    "FixedTime":  "#9E9E9E",
}
DEFAULT_COLOR = "#9C27B0"
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"
TEXT_COL = "#e0e0e0"


def _ma(data, k=15):
    if not data: return []
    return [sum(data[max(0,i-k):i+1]) / (i-max(0,i-k)+1) for i in range(len(data))]


def _load(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=11, pad=6)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)


class Plotter:
    def __init__(self, dpi=120):
        self.dpi = dpi

    # ---------------------------
    #  PER-ALGORITHM PLOTS
    # ---------------------------

    def plot_algo(self, metrics_path, save_dir, algo_name) -> bool:
        if not _MPL: return False
        data = _load(metrics_path)
        if not data: return False
        os.makedirs(save_dir, exist_ok=True)

        eps   = data.get("episodes", list(range(1, len(data["reward"]) + 1)))
        color = ALGO_COLORS.get(algo_name, DEFAULT_COLOR)

        # 1. Reward
        self._single(eps, data["reward"],
                     data.get("reward_smooth") or _ma(data["reward"]),
                     color, f"{algo_name} — Reward vs Episode",
                     "Episode Reward", os.path.join(save_dir, "reward.png"), False)

        # 2. Wait time
        self._single(eps, data["wait"],
                     data.get("wait_smooth") or _ma(data["wait"]),
                     "#ef4444", f"{algo_name} — Avg Wait Time vs Episode",
                     "Wait Time (s)", os.path.join(save_dir, "wait.png"), True)

        # 3. Queue length
        if data.get("queue"):
            self._single(eps, data["queue"], _ma(data["queue"]),
                         "#FF9800", f"{algo_name} — Queue Length vs Episode",
                         "Queue (vehicles)", os.path.join(save_dir, "queue.png"), True)

        # 4. Throughput
        if data.get("throughput"):
            self._single(eps, data["throughput"], _ma(data["throughput"]),
                         "#00BCD4", f"{algo_name} — Throughput vs Episode",
                         "Vehicles Cleared", os.path.join(save_dir, "throughput.png"), False)

        # 5. Dashboard (4-panel: reward, wait, queue, epsilon)
        self._dashboard(data, eps, algo_name, color, save_dir)
        return True

    # ---------------------------
    #  COMPARISON PLOTS
    # ---------------------------

    def plot_comparison(self,
                        algo_dirs: List[str],
                        comparison_data: Optional[list] = None,
                        multi_graph: Optional[dict] = None,
                        compare_dir: str = "outputs/compare") -> bool:
        if not _MPL: return False
        os.makedirs(compare_dir, exist_ok=True)

        # 4 individual metric comparison graphs
        if multi_graph and any(multi_graph.values()):
            self._individual_comparison_curves(multi_graph, compare_dir)

        # Bar chart with % improvement
        if comparison_data:
            self._bar_comparison(comparison_data, compare_dir)

        # 3-panel overview from metrics.json
        datasets = []
        for d in algo_dirs:
            mf   = os.path.join(d, "metrics.json")
            data = _load(mf)
            if data: datasets.append(data)
        if len(datasets) >= 2:
            self._curve_comparison(datasets, compare_dir)

        return True

    def _individual_comparison_curves(self, multi_graph: dict, compare_dir: str):
        """One graph per metric — one line per algorithm."""
        metrics_config = [
            ("reward",     "Reward Comparison",      "Episode Reward",      False, "reward_compare.png"),
            ("wait",       "Wait Time Comparison",   "Avg Wait Time (s)",   True,  "wait_compare.png"),
            ("throughput", "Throughput Comparison",  "Vehicles Cleared/Ep", False, "throughput_compare.png"),
            ("queue",      "Queue Length Comparison","Queue (vehicles)",    True,  "queue_compare.png"),
        ]

        for metric, title, ylabel, lower_better, filename in metrics_config:
            if metric not in multi_graph or not multi_graph[metric]:
                continue

            fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)

            for algo_name, values in multi_graph[metric].items():
                color    = ALGO_COLORS.get(algo_name, DEFAULT_COLOR)
                x        = list(range(1, len(values) + 1))
                smoothed = _ma(values, k=20)
                ax.plot(x, values, color=color, alpha=0.15, linewidth=0.8)
                ax.plot(x[:len(smoothed)], smoothed, color=color,
                        linewidth=2.2, label=algo_name)

            note = "(lower = better ↓)" if lower_better else "(higher = better ↑)"
            ax.text(0.99, 0.99, note, transform=ax.transAxes,
                    color="#888888", fontsize=8, ha="right", va="top")

            _style(ax, title, "Episode", ylabel)
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, fontsize=10)
            plt.tight_layout()
            path = os.path.join(compare_dir, filename)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)
            print(f"[Plotter] → {path}")

    def _bar_comparison(self, comparison_data: list, compare_dir: str):
        """Bar chart: wait + throughput side-by-side, winner highlighted."""
        algos  = [r["algorithm"] for r in comparison_data]
        waits  = [r["avg_wait"]  for r in comparison_data]
        tps    = [r["throughput"] for r in comparison_data]
        colors = [ALGO_COLORS.get(a, DEFAULT_COLOR) for a in algos]

        baseline_w = next((r["avg_wait"]   for r in comparison_data if r["algorithm"] == "FixedTime"), None)
        baseline_t = next((r["throughput"] for r in comparison_data if r["algorithm"] == "FixedTime"), None)

        rl_results = [r for r in comparison_data if r["algorithm"] != "FixedTime"]
        winner     = min(rl_results, key=lambda x: x["avg_wait"]) if rl_results else None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=DARK_BG)

        win_str = f"  Best: {winner['algorithm']} ({winner['avg_wait']:.1f}s)" if winner else ""
        fig.suptitle(f"Algorithm Comparison — Summary | {win_str}", color=TEXT_COL, fontsize=12)

        # Wait bars
        bars = ax1.bar(algos, waits, color=colors, edgecolor=GRID_COL, width=0.5)
        for bar, algo, val in zip(bars, algos, waits):
            pct = ""
            if baseline_w and algo != "FixedTime":
                impv = (baseline_w - val) / baseline_w * 100
                pct  = f"\n({impv:+.1f}%)"
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}s{pct}", ha="center",
                     color=TEXT_COL, fontsize=9, fontweight="bold")
        if winner and winner["algorithm"] in algos:
            idx = algos.index(winner["algorithm"])
            bars[idx].set_edgecolor("#FFD700")
            bars[idx].set_linewidth(2.5)
        _style(ax1, "Avg Wait Time (lower = better ↓)", "Algorithm", "Seconds")
        ax1.set_facecolor(PANEL_BG)

        # Throughput bars
        bars2 = ax2.bar(algos, tps, color=colors, edgecolor=GRID_COL, width=0.5)
        for bar, algo, val in zip(bars2, algos, tps):
            pct = ""
            if baseline_t and algo != "FixedTime":
                impv = (val - baseline_t) / max(baseline_t, 1) * 100
                pct  = f"\n({impv:+.1f}%)"
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.0f}{pct}", ha="center",
                     color=TEXT_COL, fontsize=9, fontweight="bold")
        _style(ax2, "Throughput (higher = better ↑)", "Algorithm", "Vehicles/ep")
        ax2.set_facecolor(PANEL_BG)

        plt.tight_layout()
        path = os.path.join(compare_dir, "comparison_bar.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"[Plotter] → {path}")

    def _curve_comparison(self, datasets: list, compare_dir: str):
        """3-panel smoothed overview from metrics.json files."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
        fig.suptitle("Q-Learning vs SARSA vs FixedTime — Training Curves",
                     color=TEXT_COL, fontsize=12, y=1.01)

        for data in datasets:
            algo  = data.get("algo", "unknown")
            name  = ("Q-Learning" if "q" in algo.lower()
                     else ("SARSA" if "sarsa" in algo.lower() else "FixedTime"))
            color = ALGO_COLORS.get(name, DEFAULT_COLOR)
            eps   = data.get("episodes", list(range(len(data["reward"]))))
            rs    = data.get("reward_smooth") or _ma(data["reward"])
            ws    = data.get("wait_smooth")   or _ma(data["wait"])
            tp    = _ma(data.get("throughput", []))

            axes[0].plot(eps[:len(rs)], rs, label=name, color=color, linewidth=2.0)
            axes[1].plot(eps[:len(ws)], ws, label=name, color=color, linewidth=2.0)
            if tp: axes[2].plot(eps[:len(tp)], tp, label=name, color=color, linewidth=2.0)

        _style(axes[0], "Smoothed Reward", "Episode", "Reward")
        _style(axes[1], "Smoothed Wait Time", "Episode", "Wait (s)")
        _style(axes[2], "Throughput", "Episode", "Vehicles Cleared")

        for ax in axes:
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, fontsize=9)
        plt.tight_layout()
        path = os.path.join(compare_dir, "comparison_curves.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"[Plotter] → {path}")

    # ---------------------------
    #  HELPERS
    # ---------------------------

    def _single(self, x, raw, smooth, color, title, ylabel, save, invert=False):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
        ax.plot(x, raw,    color=color, alpha=0.22, linewidth=0.8, label="Raw")
        ax.plot(x, smooth, color=color, linewidth=2.0, label="Smoothed (15-ep)")
        if invert and len(smooth) > 1:
            base = smooth[0]
            ax.fill_between(x, smooth, base,
                            where=[v < base for v in smooth],
                            alpha=0.15, color=color, label="Improvement")
        _style(ax, title, "Episode", ylabel)
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout()
        fig.savefig(save, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"[Plotter] → {save}")

    def _dashboard(self, data, eps, algo_name, color, save_dir):
        """4-panel: reward, wait, queue, epsilon."""
        fig = plt.figure(figsize=(13, 8), facecolor=DARK_BG)
        fig.suptitle(f"{algo_name} — Training Dashboard", color=TEXT_COL, fontsize=14)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        panels = [
            (data.get("reward_smooth") or _ma(data["reward"]),
             "Reward",     "Episode Reward", "#2196F3"),
            (data.get("wait_smooth") or _ma(data["wait"]),
             "Wait Time",  "Wait Time (s)", "#ef4444"),
            (_ma(data.get("queue", [])),
             "Queue",      "Queue (veh)",    "#FF9800"),
            (data.get("epsilon", []),
             "Epsilon",    "ε",              "#9C27B0"),
        ]
        for idx, (series, label, ylabel, c) in enumerate(panels):
            if not series: continue
            ax = fig.add_subplot(gs[idx//2, idx%2])
            ax.plot(eps[:len(series)], series, color=c, linewidth=1.8)
            _style(ax, label, "Episode", ylabel)
            if label == "Epsilon":
                ax.axhline(y=0.08, color="#ef4444", linewidth=1,
                           linestyle="--", label="ε_min=0.08")
                ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                          labelcolor=TEXT_COL, fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "dashboard.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"[Plotter] Dashboard → {path}")