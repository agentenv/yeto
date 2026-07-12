#!/usr/bin/env python
"""Render paper figures from recorded experiment tables.

Data sources (checked into the repo):
  fig1        docs/EXP2_25.md (seed-223 H x mu grid) + docs/EXP2_27.md (seed-251 grid)
  fig-wall    docs/EXP2_25.md E3 (anchor-panel sweep on the EXP2.23 sync capture)
  fig-mech    experiment-results/EXP2/rda-rho-law/summary.json (RDA rho lag kernel)

Styling follows the dataviz skill reference palette (light mode):
sequential blue ramp for magnitude, fixed categorical order (blue, aqua, yellow),
recessive grid/axes, direct labels, text in ink tokens rather than series color.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURF = "#ffffff"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"

# Sequential blue ramp, steps 100 -> 700 (light -> dark = low -> high loss).
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "savefig.dpi": 200,
    "axes.grid": False,
})


def heat_panel(ax, grid, row_labels, col_labels, title):
    """One annotated loss heatmap with its own scale (per-panel norm)."""
    arr = np.array(grid)
    norm = Normalize(vmin=arr.min(), vmax=arr.max())
    ax.imshow(arr, cmap=CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=10, pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(arr.shape[0]):
        best = arr[i].argmin()
        for j in range(arr.shape[1]):
            dark = norm(arr[i, j]) > 0.55
            txt = f"{arr[i, j]:.4f}"
            if j == best:
                txt += "\nbest"
            ax.text(j, i, txt, ha="center", va="center",
                    color="#ffffff" if dark else INK,
                    fontsize=8.5,
                    fontweight="bold" if j == best else "normal")


def fig1():
    seed223 = [
        [1.3519, 1.3632, 1.4383],   # H=16
        [1.3578, 1.3614, 1.4193],   # H=64
        [1.3805, 1.3674, 1.3977],   # H=256
    ]
    seed251 = [
        [1.627433, 1.641685, 1.636382],  # H=16
        [1.645687, 1.640791, 1.639274],  # H=256
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 3.1),
        gridspec_kw={"width_ratios": [3, 3], "wspace": 0.28})
    heat_panel(axes[0], seed223, ["H=16", "H=64", "H=256"],
               [r"$\mu$=0", r"$\mu$=0.5", r"$\mu$=0.9"],
               "Seed 223 (development), $\\eta$=0.175")
    heat_panel(axes[1], seed251, ["H=16", "H=256"],
               [r"$\mu$=0" + "\n" + r"$\eta$=0.175",
                r"$\mu$=0.5" + "\n" + r"$\eta$=0.175",
                r"$\mu$=0" + "\n" + r"$\eta$=0.28"],
               "Seed 251 (preregistered, fresh)")
    fig.suptitle("Held-out eval loss across sync horizon H and outer momentum",
                 fontsize=11, y=1.02)
    fig.text(0.5, -0.06,
             "Darker = higher loss (worse); each panel has its own scale. "
             "Best cell per row in bold.",
             ha="center", fontsize=8, color=INK2)
    fig.savefig(HERE / "fig1-h-mu-heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_wall():
    panels = [2, 8, 16]
    headroom = [44.2, 53.8, 54.0]
    sel_rate = [0.475, 0.1625, 0.1625]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(panels, headroom, color=BLUE, linewidth=1.6, marker="o",
            markersize=6.5, markerfacecolor=BLUE,
            markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
    for x, y, r in zip(panels, headroom, sel_rate):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK,
                    fontweight="bold")
        ax.annotate(f"sel. rate {r:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(panels, [str(p) for p in panels])
    ax.set_xlim(1.5, 22)
    ax.set_ylim(38, 60)
    ax.set_xlabel("Anchor probe size (256-row eval panels)")
    ax.set_ylabel("Oracle headroom captured (%)")
    ax.set_title("Measurement wall: probe size stops buying selection quality",
                 fontsize=10.5, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("2$\\to$8 panels: +9.6 pts", xy=(4, 49.5), fontsize=8,
                color=INK2, ha="center")
    ax.annotate("8$\\to$16: +0.2 pts", xy=(11.3, 56.0), fontsize=8,
                color=INK2, ha="center")
    fig.savefig(HERE / "fig-wall-headroom-vs-panels.png", bbox_inches="tight")
    plt.close(fig)


def fig_mechanism():
    with open(REPO / "experiment-results/EXP2/rda-rho-law/summary.json") as f:
        summary = json.load(f)
    lags = [1, 2, 3, 4]
    series = []  # (H, color, [rho per lag])
    for h, color in (("16", BLUE), ("64", AQUA), ("256", YELLOW)):
        rhos = [summary["horizons"][h]["lags"][str(k)]["rho_energy_weighted"]
                for k in lags]
        series.append((h, color, rhos))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhline(0, color=BASE, linewidth=1)
    for h, color, rhos in series:
        ax.plot(lags, rhos, color=color, linewidth=1.6, marker="o",
                markersize=6.5, markerfacecolor=color, markeredgecolor=SURF,
                markeredgewidth=1.2, zorder=3, label=f"H={h}")
        ax.annotate(f"H={h}", (lags[0], rhos[0]),
                    textcoords="offset points", xytext=(-8, 4), ha="right",
                    fontsize=9, color=INK)
    ax.set_xticks(lags)
    ax.set_xlim(0.4, 4.3)
    ax.set_xlabel("Lag k (outer rounds)")
    ax.set_ylabel(r"$\rho_k$ = energy-weighted autocorrelation")
    ax.set_title("Pseudo-gradient lag kernel of production RDA-merged deltas\n"
                 r"(seed 223, $\mu$=0 captures, replay-verified bit-exact)",
                 fontsize=10, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(HERE / "fig-mechanism-rho-lag-kernel.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig_wall()
    fig_mechanism()
    print("rendered:", *sorted(p.name for p in HERE.glob("*.png")))
