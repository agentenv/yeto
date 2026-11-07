"""Small line charts: PNG through matplotlib when it is installed (optional
dependency), otherwise a text sparkline the channels inline into the message."""

from __future__ import annotations

import io
import math
import statistics
from dataclasses import dataclass, field

SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class Chart:
    title: str
    series: list[tuple[str, list[float], list[float]]] = field(default_factory=list)  # (label, xs, ys)
    xlabel: str = ""
    time_axis: bool = False  # xs are unix seconds
    hlines: list[tuple[str, float]] = field(default_factory=list)  # reference lines

    def empty(self) -> bool:
        return not any(len(ys) >= 2 for _, _, ys in self.series)


def _clean(xs: list, ys: list) -> tuple[list[float], list[float]]:
    pts = [(x, y) for x, y in zip(xs, ys, strict=False) if isinstance(y, (int, float)) and not isinstance(y, bool)
           and math.isfinite(y) and isinstance(x, (int, float))]
    return [p[0] for p in pts], [float(p[1]) for p in pts]


def sparkline(ys: list[float], width: int = 24) -> str:
    vals = [y for y in ys if isinstance(y, (int, float)) and math.isfinite(y)]
    if not vals:
        return ""
    if len(vals) > width:  # bucket means
        n = len(vals)
        vals = [statistics.fmean(vals[i * n // width:(i + 1) * n // width]) for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[3] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def text_lines(chart: Chart) -> list[str]:
    """Sparkline fallback: one line per series with its range."""
    out = []
    for label, xs, ys in chart.series:
        _, ys = _clean(xs, ys)
        if ys:
            out.append(f"{chart.title} {label}: {sparkline(ys)} {min(ys):.4g}..{max(ys):.4g} (last {ys[-1]:.4g})")
    return out


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def render_png(chart: Chart) -> bytes | None:
    """PNG bytes, or None without matplotlib or without data."""
    if chart.empty() or not available():
        return None
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.4, 2.6), dpi=110)
    ax = fig.add_subplot(1, 1, 1)
    for label, xs, ys in chart.series:
        xs, ys = _clean(xs, ys)
        if not ys:
            continue
        if chart.time_axis:
            from datetime import datetime

            xs = [datetime.fromtimestamp(x) for x in xs]
        ax.plot(xs, ys, label=label, linewidth=1.4, marker="o" if len(ys) <= 12 else None, markersize=3)
    for label, y in chart.hlines:
        ax.axhline(y, linestyle="--", linewidth=1.0, color="grey")
        ax.annotate(label, xy=(0, y), xycoords=("axes fraction", "data"), fontsize=7, color="grey",
                    va="bottom")
    ax.set_title(chart.title, fontsize=10)
    if chart.xlabel:
        ax.set_xlabel(chart.xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    if len(chart.series) > 1:
        ax.legend(fontsize=7, loc="best")
    if chart.time_axis:
        fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()
