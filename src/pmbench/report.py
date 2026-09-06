"""Figures for the writeup.

Every chart here also has a text/CSV twin, because three of the five series
colors sit below 3:1 against the chart surface: the palette passes CVD
separation but carries a contrast WARN, which obliges visible labels and a table
view rather than colour alone. Hence direct labels on every line and a CSV
emitted beside each PNG.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import abstention_curve, by_category, by_horizon, by_config  # noqa: E402
from .baselines import Row  # noqa: E402
from .scoring import brier, reliability_bins, summarise  # noqa: E402

# Reference palette (light surface). Categorical hues are assigned in fixed
# order and never cycled; a 6th config folds into "Other" rather than inventing
# a hue.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKET_COLOR = INK_2  # the market baseline is chrome, not a competitor series


def _style(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def _label_ends(ax, entries: list[tuple[float, str, str]], *, min_gap: float = 0.05,
                x: float = 1.01) -> None:
    """Direct-label series at the right edge, nudged apart so they never stack.

    Series routinely finish in the same bin; fixed offsets collide as soon as
    two of them land close together, so positions are de-collided in axes
    coordinates before drawing.
    """
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    items = sorted(((y - lo) / span, text, color) for y, text, color in entries)
    placed: list[float] = []
    for frac, _, _ in items:
        # max(), not a while-loop: `placed[-1] + min_gap - placed[-1]` can round
        # to fractionally less than min_gap, and the loop condition then never
        # goes false.
        placed.append(frac if not placed else max(frac, placed[-1] + min_gap))
    # Re-centre if the stack ran off the top.
    overflow = max(0.0, (placed[-1] if placed else 0.0) - 1.0)
    for pos, (_, text, color) in zip(placed, items):
        ax.annotate(text, xy=(x, pos - overflow), xycoords="axes fraction",
                    color=color, fontsize=9, va="center", annotation_clip=False)


def _fig(w: float = 8.0, h: float = 5.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path, *, tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _csv(path: Path, header: Sequence[str], rows: Sequence[Sequence]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------


def reliability_diagram(rows: Sequence[Row], out: Path, *, n_bins: int = 10) -> list[Path]:
    """Predicted probability vs observed frequency. The diagonal is perfect
    calibration; above it is underconfident, below it is overconfident."""
    fig, ax = _fig()
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.5, linestyle="--",
            zorder=1, label="_perfect")
    # Sits mid-diagonal, not in the top-right corner where the series labels go.
    ax.text(0.46, 0.44, "perfectly calibrated", color=MUTED, fontsize=9,
            ha="left", va="top", rotation=45, rotation_mode="anchor")

    table: list[list] = []
    ends: list[tuple[float, str, str]] = []
    groups = by_config(rows)

    # The market price gets its own curve: it is the baseline that matters and
    # seeing where it sits on the diagonal is half the story.
    priced = [r for r in rows if r.market_price is not None]
    if priced:
        seen: dict[str, Row] = {r.market_id: r for r in priced}
        mb = reliability_bins([r.market_price for r in seen.values()],
                              [r.outcome for r in seen.values()], n_bins)
        if mb:
            ax.plot([b.mean_pred for b in mb], [b.freq_actual for b in mb],
                    color=MARKET_COLOR, linewidth=2.0, marker="s", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                    label="market price")
            ax.scatter([b.mean_pred for b in mb], [b.freq_actual for b in mb],
                       s=[max(40, min(400, 18 * b.n)) for b in mb],
                       color=MARKET_COLOR, alpha=0.20, edgecolors="none", zorder=2)
            ends.append((mb[-1].freq_actual, "market price", MARKET_COLOR))
            table += [["market price", b.lo, b.hi, b.n, b.mean_pred, b.freq_actual]
                      for b in mb]

    for i, (name, rs) in enumerate(sorted(groups.items())):
        color = SERIES[i % len(SERIES)]
        bins = reliability_bins([r.probability for r in rs], [r.outcome for r in rs], n_bins)
        if not bins:
            continue
        ax.plot([b.mean_pred for b in bins], [b.freq_actual for b in bins],
                color=color, linewidth=2.0, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label=name)
        # Bin counts drive marker area, so a bucket with 3 markets is visibly
        # not a bucket with 80.
        ax.scatter([b.mean_pred for b in bins], [b.freq_actual for b in bins],
                   s=[max(40, min(400, 18 * b.n)) for b in bins],
                   color=color, alpha=0.25, edgecolors="none", zorder=2)
        ends.append((bins[-1].freq_actual, name, color))
        table += [[name, b.lo, b.hi, b.n, b.mean_pred, b.freq_actual] for b in bins]

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.subplots_adjust(right=0.80)
    _style(ax, xlabel="mean forecast probability", ylabel="observed frequency of YES",
           title="Reliability — below the diagonal is overconfident")
    fig.text(0.01, 0.01, "marker area \u221d markets in the bin",
             color=MUTED, fontsize=9, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _label_ends(ax, ends)
    # tight_layout would undo the right margin the labels live in.
    p1 = _save(fig, out, tight=False)
    p2 = _csv(out.with_suffix(".csv"),
              ["series", "bin_lo", "bin_hi", "n", "mean_pred", "freq_actual"], table)
    return [p1, p2]


def baseline_chart(rows: Sequence[Row], out: Path) -> list[Path]:
    """Model Brier against the market price on the same markets, per config."""
    groups = by_config(rows)
    names, model_b, market_b = [], [], []
    for name, rs in sorted(groups.items()):
        priced = [r for r in rs if r.market_price is not None]
        if not priced:
            continue
        names.append(name)
        model_b.append(sum(brier(r.probability, r.outcome) for r in priced) / len(priced))
        market_b.append(sum(brier(r.market_price, r.outcome) for r in priced) / len(priced))

    fig, ax = _fig(8.0, max(3.2, 0.85 * len(names) + 2.2))
    y = range(len(names))
    for i, (yy, m, k) in enumerate(zip(y, model_b, market_b)):
        color = SERIES[i % len(SERIES)]
        # A dumbbell: the gap IS the result, so draw the gap.
        ax.plot([k, m], [yy, yy], color=GRID, linewidth=3, zorder=1,
                solid_capstyle="round")
        ax.scatter([k], [yy], s=90, color=MARKET_COLOR, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([m], [yy], s=110, color=color, zorder=4,
                   edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(f"{m:.3f}", (m, yy), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=INK_2, fontsize=9)
    ax.axvline(0.25, color=BASELINE, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate("always 0.5", xy=(0.25, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -10), textcoords="offset points",
                color=MUTED, fontsize=9, va="top")
    ax.set_yticks(list(y)); ax.set_yticklabels(names, color=INK_2, fontsize=10)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.invert_yaxis()
    _style(ax, xlabel="Brier score (lower is better)",
           title="Model vs market price, same markets")
    ax.scatter([], [], s=90, color=MARKET_COLOR, label="market price")
    ax.scatter([], [], s=110, color=SERIES[0], label="model config")
    # Below the plot: with only a handful of rows any in-axes corner collides
    # with a mark.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.13 - 0.4 / max(len(names), 1)))
    p1 = _save(fig, out)
    p2 = _csv(out.with_suffix(".csv"), ["config", "brier_model", "brier_market"],
              list(zip(names, model_b, market_b)))
    return [p1, p2]


def abstention_chart(rows: Sequence[Row], out: Path) -> list[Path]:
    """Brier as the most-disagreeing forecasts are dropped. A flat line means
    ensemble spread carries no signal."""
    fig, ax = _fig()
    table = []
    ends: list[tuple[float, str, str]] = []
    for i, (name, rs) in enumerate(sorted(by_config(rows).items())):
        curve = abstention_curve(rs)
        if not curve.keep_fracs:
            continue
        color = SERIES[i % len(SERIES)]
        ax.plot(curve.keep_fracs, curve.briers, color=color, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(f"{name}  ρ={curve.correlation:+.2f}",
                    (curve.keep_fracs[-1], curve.briers[-1]),
                    textcoords="offset points", xytext=(8, 0), color=color, fontsize=9)
        table += [[name, k, n, b, curve.correlation]
                  for k, n, b in zip(curve.keep_fracs, curve.ns, curve.briers)]
    ax.invert_xaxis()
    _style(ax, xlabel="fraction of forecasts kept (dropping highest ensemble spread first)",
           ylabel="Brier score", title="Does disagreement predict error?")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.subplots_adjust(right=0.74)
    _label_ends(ax, ends)
    p1 = _save(fig, out, tight=False)
    p2 = _csv(out.with_suffix(".csv"),
              ["config", "keep_frac", "n", "brier", "spearman_err_vs_sd"], table)
    return [p1, p2]


def horizon_chart(rows: Sequence[Row], out: Path) -> list[Path]:
    """Does forecast quality decay with horizon length?"""
    fig, ax = _fig()
    table = []
    ends: list[tuple[float, str, str]] = []
    buckets: list[str] = []
    for i, (name, rs) in enumerate(sorted(by_config(rows).items())):
        data = by_horizon(rs)
        if not data:
            continue
        buckets = list(data)
        color = SERIES[i % len(SERIES)]
        xs = range(len(data))
        ys = [s.brier for s, _ in data.values()]
        ax.plot(list(xs), ys, color=color, linewidth=2.0, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ends.append((ys[-1], name, color))
        table += [[name, b, s.n, s.brier, mkt] for b, (s, mkt) in data.items()]
        mkts = [mkt for _, mkt in data.values()]
        if all(m is not None for m in mkts):
            ax.plot(list(xs), mkts, color=MARKET_COLOR, linewidth=2.0, linestyle=":",
                    marker="s", markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label="market price" if i == 0 else "_x")
    if buckets:
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets, color=INK_2, fontsize=10)
    _style(ax, xlabel="days to resolution at forecast time", ylabel="Brier score",
           title="Forecast quality vs horizon")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.subplots_adjust(right=0.82)
    _label_ends(ax, ends)
    p1 = _save(fig, out, tight=False)
    p2 = _csv(out.with_suffix(".csv"),
              ["config", "bucket", "n", "brier_model", "brier_market"], table)
    return [p1, p2]


def category_chart(rows: Sequence[Row], out: Path, config_name: str | None = None) -> list[Path]:
    """Where the model is least bad, next to the market on the same questions."""
    rs = [r for r in rows if config_name is None or r.config_name == config_name]
    data = by_category(rs)
    if not data:
        return []
    cats = list(data)
    fig, ax = _fig(8.0, max(3.2, 0.6 * len(cats) + 2.2))
    y = range(len(cats))
    for i, (yy, cat) in enumerate(zip(y, cats)):
        s, mkt = data[cat]
        if mkt is not None:
            ax.plot([mkt, s.brier], [yy, yy], color=GRID, linewidth=3, zorder=1,
                    solid_capstyle="round")
            ax.scatter([mkt], [yy], s=80, color=MARKET_COLOR, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([s.brier], [yy], s=100, color=SERIES[0], zorder=4,
                   edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(f"n={s.n}", (max(s.brier, mkt or 0), yy),
                    textcoords="offset points", xytext=(10, -3),
                    color=MUTED, fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(cats, color=INK_2, fontsize=10)
    ax.set_ylim(-0.7, len(cats) - 0.3)
    ax.invert_yaxis()
    _style(ax, xlabel="Brier score (lower is better)",
           title=f"By category{f' — {config_name}' if config_name else ''}")
    ax.scatter([], [], s=80, color=MARKET_COLOR, label="market price")
    ax.scatter([], [], s=100, color=SERIES[0], label="model")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.13 - 0.4 / max(len(cats), 1)))
    p1 = _save(fig, out)
    p2 = _csv(out.with_suffix(".csv"), ["category", "n", "brier_model", "brier_market"],
              [[c, data[c][0].n, data[c][0].brier, data[c][1]] for c in cats])
    return [p1, p2]


def build_all(rows: Sequence[Row], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written += reliability_diagram(rows, out_dir / "reliability.png")
    written += baseline_chart(rows, out_dir / "baselines.png")
    written += abstention_chart(rows, out_dir / "abstention.png")
    written += horizon_chart(rows, out_dir / "horizon.png")
    written += category_chart(rows, out_dir / "category.png")
    return written
