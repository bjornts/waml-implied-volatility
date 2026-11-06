"""Draw the report figure: one collapse up close, then error and violations for each model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "runs"
TRACE_PATH = ROOT / "results" / "variants" / "stability_trace.json"
FIGURE_DIR = ROOT / "figures"

LABELS = {
    "data_only": "Data-only",
    "pinn": "PINN",
    "dcpinn": "DCPINN",
    "dcpinn_waml": "WamL (paper)",
    "dcpinn_waml_no_pde": "WamL, no PDE",
    "waml_floor": "WamL, floor",
}
SINGLE, DOUBLE = "#2a78d6", "#eb6834"  # Validated categorical slots 1 and 2.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRACE_START, TRACE_END = 800, 1200
# Violation rates reach exactly zero, so this panel is linear up to 0.1% and logarithmic above.
VIOLATION_LINEAR_LIMIT = 0.1


def load_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(RUN_DIR.glob("noisy__*.json")):
        run = json.loads(path.read_text())
        if run["model"] in LABELS:
            rows.append(
                {
                    "model": run["model"],
                    "precision": run["precision"],
                    "iv_rmse": run["metrics"]["iv_rmse"],
                    "viol_any": run["metrics"]["viol_any"],
                }
            )
    return pd.DataFrame(rows)


def _style(axis, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=8.5, color=INK)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, length=0, labelsize=7.5)
    axis.tick_params(which="minor", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def _trace_panel(axis) -> None:
    """Show the quote loss around the step where one balance weight jumps."""
    history = json.loads(TRACE_PATH.read_text())["history"]
    window = [h for h in history if TRACE_START <= h["step"] <= TRACE_END]
    jump = next(h["step"] for h in window if h["balance"]["dK"] > 1.0)
    axis.plot(
        [h["step"] for h in window],
        [h["losses"]["quote"] for h in window],
        color=SINGLE,
        linewidth=1.6,
    )
    axis.axvline(jump, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    axis.annotate(
        "$\\lambda_{\\partial K}$ jumps\nto $3.9\\times10^{15}$",
        xy=(jump, 2.5e-4),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    axis.set_yscale("log")
    axis.set_xlabel("Training step", fontsize=7.5, color=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    _style(axis, "One collapse: quote loss")


def _seed_panel(axis, data: pd.DataFrame, column: str, title: str) -> None:
    """Plot each seed as a dot and each median as a tick, so outliers stay visible."""
    order = list(LABELS)
    styles = {"single": (SINGLE, "o", -0.16), "double": (DOUBLE, "s", 0.16)}
    for precision, (color, marker, shift) in styles.items():
        subset = data[data["precision"] == precision]
        for row, model in enumerate(order):
            values = subset.loc[subset["model"] == model, column]
            if values.empty:
                continue
            y = row + shift
            axis.scatter(values, [y] * len(values), s=15, marker=marker, color=color, alpha=0.8,
                         linewidths=0, zorder=3)
            axis.plot([values.median()] * 2, [y - 0.15, y + 0.15], color=INK, linewidth=1.3, zorder=4)
    axis.set_yticks(range(len(order)), [LABELS[model] for model in order])
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    _style(axis, title)


def main() -> None:
    data = load_metrics()
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.65), constrained_layout=True, width_ratios=[1.0, 1.1, 1.1]
    )
    _trace_panel(axes[0])
    _seed_panel(axes[1], data, "iv_rmse", "Held-out IV RMSE")
    axes[1].set_xscale("log")
    _seed_panel(axes[2], data, "viol_any", "Violations (% of grid)")
    axes[2].set_xscale("symlog", linthresh=VIOLATION_LINEAR_LIMIT)
    axes[2].set_xticks([0, 0.1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
    axes[2].tick_params(labelleft=False)

    handles = [
        plt.Line2D([], [], marker="o", color=SINGLE, linestyle="", markersize=4, label="Single precision"),
        plt.Line2D([], [], marker="s", color=DOUBLE, linestyle="", markersize=4, label="Double precision"),
        plt.Line2D([], [], color=INK, linewidth=1.3, label="Median of 5 seeds"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=7.5)
    figure.savefig(FIGURE_DIR / "results.pdf")
    figure.savefig(FIGURE_DIR / "results.png", dpi=200)
    print(f"Wrote {FIGURE_DIR / 'results.pdf'}")


if __name__ == "__main__":
    main()
