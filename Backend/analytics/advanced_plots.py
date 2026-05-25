import os
import json
import math

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"
TEXT_COL = "#e0e0e0"
ALGO_COLORS = {
    "Q-Learning": "#2196F3",
    "SARSA":      "#FF9800",
    "FixedTime":  "#9E9E9E",
}


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)


def _ma(data, k=15):
    if not data: return []
    return [sum(data[max(0, i-k):i+1]) / (i - max(0, i-k) + 1) for i in range(len(data))]


def _ma50(data): return _ma(data, k=50)


def _stats(data):
    if not data: return {"mean": 0, "std": 0, "cv": 0}
    n    = len(data)
    mean = sum(data) / n
    var  = sum((v - mean)**2 for v in data) / n
    std  = math.sqrt(var)
    cv   = std / abs(mean) * 100 if mean != 0 else 0
    return {"mean": mean, "std": std, "cv": cv}


def _jain(wait_per_ep):
    """Jain Fairness Index per episode."""
    result = []
    for i in range(len(wait_per_ep)):
        w  = wait_per_ep[max(0, i-2):i+3]
        n  = len(w)
        if n == 0: result.append(1.0); continue
        s1 = sum(w); s2 = sum(v**2 for v in w)
        result.append(min((s1**2) / (n * s2 + 1e-9), 1.0))
    return result


def _load(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None


class AdvancedPlotter:
    def __init__(self, dpi=120):
        self.dpi = dpi

    def plot_all(self, metrics_path: str, save_dir: str, algo_name: str) -> bool:
        if not _MPL: return False
        data = _load(metrics_path)
        if not data: return False
        os.makedirs(save_dir, exist_ok=True)

        eps   = data.get("episodes", list(range(1, len(data["reward"]) + 1)))
        color = ALGO_COLORS.get(algo_name, "#9C27B0")

        # 1. Preference alignment
        if data.get("preference_rate"):
            self._preference_plot(eps, data["preference_rate"],
                                  algo_name, os.path.join(save_dir, "preference.png"))

        # 2. Action distribution bar
        if data.get("action_dist") and data["action_dist"]:
            last_n   = data["action_dist"][-30:]
            ncols    = min(4, len(last_n[0]) if last_n else 4)
            avg_dist = [round(sum(ep[i] for ep in last_n) / len(last_n), 1)
                        for i in range(ncols)]
            self._action_bar(avg_dist, algo_name,
                             os.path.join(save_dir, "action_dist.png"))

        if data.get("wait"):
            # 3. Jain Fairness Index
            jfi = _jain(data["wait"])
            self._line(eps, _ma(jfi), "#8BC34A",
                       f"{algo_name} — Jain Fairness Index",
                       "JFI (1 = perfectly fair)",
                       os.path.join(save_dir, "fairness.png"),
                       hline=(0.9, "Target ≥ 0.9"))

            # 4. 50-ep wait trend
            self._wait_trend(eps, data["wait"], algo_name,
                             os.path.join(save_dir, "wait_trend_50ep.png"))

        # 5. Advanced 6-panel dashboard
        self._dashboard_6(data, eps, algo_name, color, save_dir)
        return True

    # ── Helpers ───────────────────────────────────────────────────────

    def _line(self, x, y, color, title, ylabel, save, hline=None):
        if not y: return
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DARK_BG)
        ax.plot(x[:len(y)], y, color=color, linewidth=1.8)
        if hline:
            ax.axhline(y=hline[0], color="#ef4444", linewidth=1,
                       linestyle="--", label=hline[1])
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, fontsize=8)
        _style(ax, title, "Episode", ylabel)
        plt.tight_layout()
        fig.savefig(save, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)

    def _preference_plot(self, x, pref_data, algo_name, save):
        ma15    = _ma(pref_data, 15)
        n       = len(pref_data)
        p_first = sum(pref_data[:50]) / 50 if n >= 50 else (pref_data[0] if pref_data else 0)
        p_last  = sum(pref_data[-50:]) / 50 if n >= 50 else (pref_data[-1] if pref_data else 0)
        arrow   = ("↑ converging" if p_last > p_first + 1
                   else ("→ stable" if abs(p_last - p_first) <= 2 else "↓ regressing"))

        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DARK_BG)
        ax.plot(x[:n], pref_data, color="#FFD700", alpha=0.2, linewidth=0.8, label="Raw")
        ax.plot(x[:len(ma15)], ma15, color="#FFD700", linewidth=2.0, label="15-ep avg")
        ax.axhline(y=50, color="#ef4444", linewidth=1, linestyle="--", label="50% target")
        ax.axhline(y=33, color="#666666", linewidth=0.8, linestyle=":", label="Random (~33%)")

        mid_y = max(ma15) * 0.78 if ma15 else 45
        ax.annotate(
            f"First 50: {p_first:.1f}%\n→ Last 50: {p_last:.1f}%\n{arrow}",
            xy=(x[len(x)//2], mid_y), color="#cccccc", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                      edgecolor=GRID_COL, alpha=0.85)
        )
        _style(ax, f"{algo_name} — Preference Alignment",
               "Episode", "% Decisions = Best Valid Dir")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout()
        fig.savefig(save, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)

    def _action_bar(self, action_dist, algo_name, save):
        dirs   = ["N", "E", "S", "W"][:len(action_dist)]
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=DARK_BG)
        ax.set_facecolor(PANEL_BG)
        bars = ax.barh(dirs, action_dist, color=colors[:len(dirs)], edgecolor=GRID_COL)
        for bar, val in zip(bars, action_dist):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color=TEXT_COL, fontsize=9)
        ax.axvline(25, color="#ef4444", linewidth=1, linestyle="--", label="Uniform 25%")
        _style(ax, f"{algo_name} — Action Distribution (last 30 eps)",
               "% of Decisions", "Direction")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout()
        fig.savefig(save, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)

    def _wait_trend(self, x, wait, algo_name, save):
        ma15 = _ma(wait, 15)
        ma50 = _ma50(wait)
        n    = len(wait)

        fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
        ax.plot(x[:n], wait, color="#ef4444", alpha=0.12, linewidth=0.6, label="Raw")
        ax.plot(x[:len(ma15)], ma15, color="#ef4444", alpha=0.5,
                linewidth=1.2, label="15-ep avg")
        ax.plot(x[:len(ma50)], ma50, color="#ffffff",
                linewidth=2.2, label="50-ep avg (trend)")

        if len(ma50) >= 50:
            sv  = ma50[49]
            ev  = ma50[-1]
            pct = (sv - ev) / sv * 100 if sv > 0 else 0
            ax.annotate(f"ep 50: {sv:.1f}s", xy=(x[49], sv),
                        xytext=(x[49] + max(len(x)//20, 5), sv + 1.5),
                        color="#aaaaaa", fontsize=8)
            clr = "#22c55e" if pct > 0 else "#ef4444"
            ax.annotate(f"ep {n}: {ev:.1f}s\n({-pct:+.1f}%)",
                        xy=(x[-1], ev),
                        xytext=(x[-1] - max(len(x)//6, 30), ev + 1.5),
                        color=clr, fontsize=8)

        s = _stats(wait[-100:] if n >= 100 else wait)
        ax.text(0.02, 0.97,
                f"Last 100 eps\nMean: {s['mean']:.1f}s\n"
                f"Std:  {s['std']:.1f}s\nCV:   {s['cv']:.1f}%",
                transform=ax.transAxes, color="#bbbbbb", fontsize=7.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                          edgecolor=GRID_COL, alpha=0.85))

        _style(ax, f"{algo_name} — Wait Time Trend (50-ep moving average)",
               "Episode", "Avg Wait Time (s)")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout()
        fig.savefig(save, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)

    def _dashboard_6(self, data, eps, algo_name, color, save_dir):
        fig = plt.figure(figsize=(15, 8), facecolor=DARK_BG)
        fig.suptitle(f"{algo_name} — Advanced Training Summary",
                     color=TEXT_COL, fontsize=13)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

        def _stl(key, unit=""):
            d = data.get(key, [])
            s = _stats(d[-100:] if len(d) >= 100 else d)
            return f"Last 100: {s['mean']:.1f}{unit} ± {s['std']:.1f}{unit}"

        # Row 1: Reward, Wait, Throughput
        ax1 = fig.add_subplot(gs[0, 0])
        if data.get("reward"):
            r50 = _ma50(data["reward"])
            ax1.plot(eps[:len(r50)], r50, color=color, linewidth=1.8)
        _style(ax1, f"Reward (50-ep avg)\n{_stl('reward')}", "Episode", "Reward")

        ax2 = fig.add_subplot(gs[0, 1])
        if data.get("wait"):
            w50 = _ma50(data["wait"])
            ax2.plot(eps[:len(w50)], w50, color="#ef4444", linewidth=2.2)
            if len(w50) > 1:
                ax2.fill_between(eps[:len(w50)], w50, w50[0],
                                 where=[v < w50[0] for v in w50],
                                 alpha=0.2, color="#22c55e")
        _style(ax2, f"Wait Time (50-ep avg) ← Key\n{_stl('wait', 's')}",
               "Episode", "Wait (s)")

        ax3 = fig.add_subplot(gs[0, 2])
        if data.get("throughput"):
            t15 = _ma(data["throughput"])
            ax3.plot(eps[:len(t15)], t15, color="#00BCD4", linewidth=1.8)
        _style(ax3, f"Throughput (15-ep avg)\n{_stl('throughput')}",
               "Episode", "Vehicles Cleared")

        # Row 2: Queue, Preference, Epsilon
        ax4 = fig.add_subplot(gs[1, 0])
        if data.get("queue"):
            q15 = _ma(data["queue"])
            ax4.plot(eps[:len(q15)], q15, color="#FF9800", linewidth=1.8)
        _style(ax4, f"Queue Length (15-ep avg)\n{_stl('queue', ' veh')}",
               "Episode", "Vehicles")

        ax5 = fig.add_subplot(gs[1, 1])
        pf, pl, arr = 0, 0, "→"
        if data.get("preference_rate"):
            p15 = _ma(data["preference_rate"])
            ax5.plot(eps[:len(p15)], p15, color="#FFD700", linewidth=1.8)
            ax5.axhline(y=50, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.7)
            ax5.axhline(y=33, color="#666666", linewidth=0.6, linestyle=":", alpha=0.6)
            pr  = data["preference_rate"]
            pf  = round(sum(pr[:50]) / 50, 0) if len(pr) >= 50 else 0
            pl  = round(sum(pr[-50:]) / 50, 0) if len(pr) >= 50 else 0
            arr = "↑" if pl > pf else ("→" if abs(pl-pf) <= 2 else "↓")
        _style(ax5, f"Preference Alignment\n{pf:.0f}% → {pl:.0f}% ({arr})",
               "Episode", "% Best Valid Dir")

        ax6 = fig.add_subplot(gs[1, 2])
        if data.get("epsilon"):
            ax6.plot(eps[:len(data["epsilon"])], data["epsilon"],
                     color="#9C27B0", linewidth=1.8)
            ax6.axhline(y=0.08, color="#ef4444", linewidth=0.8,
                        linestyle="--", alpha=0.7, label="ε_min=0.08")
            ax6.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=8)
        _style(ax6, "Epsilon Decay", "Episode", "ε")

        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        path = os.path.join(save_dir, "advanced_dashboard.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)