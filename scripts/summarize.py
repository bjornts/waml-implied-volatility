"""Combine finished runs into summary files and the report's results table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "runs"
MODEL_ORDER = [
    "data_only",
    "pinn",
    "dcpinn",
    "dcpinn_waml",
    "dcpinn_waml_no_pde",
    "dcpinn_waml_step0",
    "waml_floor",
]
COLUMNS = [
    "price_rmse",
    "iv_rmse",
    "iv_rmse_all",
    "mesh_iv_rmse",
    "mesh_iv_rmse_excl_corner",
    "corner_iv_rmse",
    "viol_dK",
    "viol_dKK",
    "viol_dT",
    "viol_any",
    "viol_any_excl_corner",
    "pde_mse_double",
    "seconds",
    "nonfinite_points",
    "max_point_weight",
    "max_balance_pde",
    "stalled",
    "collapsed",
]
# A collapsed run ends more than COLLAPSE_RATIO times above its best logged quote loss, and
# also above COLLAPSE_LOSS, about what a flat surface achieves on these quotes.
COLLAPSE_RATIO = 10.0
COLLAPSE_LOSS = 1e-4
TABLE_LABELS = {
    "data_only": "Data-only",
    "pinn": "PINN",
    "dcpinn": "DCPINN",
    "dcpinn_waml": "WamL, paper's rule",
    "dcpinn_waml_no_pde": "WamL, no PDE loss",
    "waml_floor": "WamL, gradient floor",
}
TABLE_COLUMNS = [
    ("noisy", "single", "Noisy, single"),
    ("noisy", "double", "Noisy, double"),
    ("clean", "single", "Clean, single"),
    ("sparse", "single", "Sparse, single"),
]


def load_runs() -> pd.DataFrame:
    """Read one row per run, including how far the WamL weights moved."""
    rows = []
    for path in sorted(RUN_DIR.glob("*.json")):
        run = json.loads(path.read_text())
        history = run["history"]
        last = history[-1]
        rows.append(
            {
                "scenario": run["scenario"],
                "model": run["model"],
                "precision": run["precision"],
                "seed": run["seed"],
                "seconds": run["seconds"],
                "nonfinite_points": run["nonfinite_points"],
                "max_point_weight": max(last["point_weight_max"].values()),
                "max_balance_pde": max(h["balance"].get("pde", 1.0) for h in history),
                # A stalled run has the same quote loss over the second half of training.
                "stalled": history[len(history) // 2]["losses"]["quote"] == last["losses"]["quote"],
                "collapsed": last["losses"]["quote"] > COLLAPSE_LOSS
                and last["losses"]["quote"]
                > COLLAPSE_RATIO * min(h["losses"]["quote"] for h in history),
                **run["metrics"],
            }
        )
    return pd.DataFrame(rows)


def _cells(runs: pd.DataFrame, scenario: str, precision: str, model: str) -> str:
    """Median IV error and violations, then the number of runs that collapsed."""
    rows = runs[
        (runs["scenario"] == scenario) & (runs["precision"] == precision) & (runs["model"] == model)
    ]
    return (
        f"{rows['iv_rmse'].median():.3f} & {rows['viol_any'].median():.2f} & "
        f"{int(rows['collapsed'].sum())}/{len(rows)}"
    )


def write_latex_table(runs: pd.DataFrame) -> None:
    """Write the report's table, so that its numbers always come from the run files."""
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for *_, label in TABLE_COLUMNS)
    rules = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(TABLE_COLUMNS)))
    head = " & ".join(["IV", r"Viol.\,\%", "Coll."] * len(TABLE_COLUMNS))
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(TABLE_COLUMNS) + "}",
        r"\toprule",
        f" & {groups}" + r"\\",
        rules,
        f"Model & {head}" + r"\\",
        r"\midrule",
    ]
    for model, label in TABLE_LABELS.items():
        cells = " & ".join(_cells(runs, scenario, precision, model) for scenario, precision, _ in TABLE_COLUMNS)
        lines.append(f"{label} & {cells}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results" / "table_results.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = load_runs()
    write_latex_table(runs)
    grouped = runs.groupby(["scenario", "precision", "model"])[COLUMNS]
    summary = grouped.agg(["mean", "std", "count"]).reindex(MODEL_ORDER, level="model")
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(ROOT / "results" / "summary.csv")
    runs.to_csv(ROOT / "results" / "all_runs.csv", index=False)

    shown = ["price_rmse", "iv_rmse", "mesh_iv_rmse", "viol_any", "viol_any_excl_corner", "seconds", "collapsed"]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for key, table in runs.groupby(["scenario", "precision"]):
        print(f"\n=== {key[0]} / {key[1]} ({table['seed'].nunique()} seeds)")
        means = table.groupby("model")[shown + ["stalled"]].mean().reindex(MODEL_ORDER).dropna(how="all")
        stds = table.groupby("model")[shown].std().reindex(MODEL_ORDER).dropna(how="all")
        print(means.map(lambda value: f"{value:.4g}"))
        print("standard deviation:")
        print(stds.map(lambda value: f"{value:.3g}"))


if __name__ == "__main__":
    main()
