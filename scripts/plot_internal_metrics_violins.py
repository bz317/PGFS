#!/usr/bin/env python3
"""PGFS internal-test violins: Original + PGFS (qed) + PGFS (delta_qed).

Loads precomputed arrays from the sibling GenMolRL plot cache
(``run_detailed_results/experiments_vis/plot_cache/internal/``).

- **QED** — per-episode max QED for PGFS (paper-style); start-molecule QED for Original.
- **Diversity** — bootstrap structural diversity (Morgan FP, Tanimoto).
- **SA** — max-QED-molecule SA for PGFS; start-molecule SA for Original.

Outputs:
  - ``figures/pgfs_internal_qed_violin.png``
  - ``figures/pgfs_internal_diversity_violin.png``
  - ``figures/pgfs_internal_sa_violin.png``
  - ``figures/pgfs_internal_3x3_violin.png`` (metrics × methods grid)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    HAS_SEABORN = False

PGFS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = (
    PGFS_ROOT.parent / "GenMolRL/run_detailed_results/experiments_vis/plot_cache/internal"
)

METHODS: tuple[str, ...] = ("Original", "PGFS (qed)", "PGFS (delta_qed)")
CACHE_FILES: dict[str, str] = {
    "Original": "original_molecules.json",
    "PGFS (qed)": "pgfs_qed.json",
    "PGFS (delta_qed)": "pgfs_delta_qed.json",
}

METHOD_STYLE: dict[str, dict[str, str]] = {
    "Original": {"color": "#B0B0B0", "label": "Original"},
    "PGFS (qed)": {"color": "#F58518", "label": "PGFS (qed)"},
    "PGFS (delta_qed)": {"color": "#FDB462", "label": "PGFS (δ_qed)"},
}

METRICS: tuple[tuple[str, str, str, float, float, list[float]], ...] = (
    ("qed", "QED ↑", "qed", 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    (
        "diversity",
        "Diversity ↑",
        "diversity_bootstrap",
        0.0,
        1.0,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ),
    ("sa", "SA ↓", "sa", 1.0, 8.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
)


def default_cache_dir() -> Path:
    return DEFAULT_CACHE_DIR.resolve()


def load_caches(cache_dir: Path) -> dict[str, dict]:
    caches: dict[str, dict] = {}
    for method, filename in CACHE_FILES.items():
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing plot cache for {method}: {path}")
        caches[method] = json.loads(path.read_text(encoding="utf-8"))
    return caches


def metric_values(cache: dict, key: str) -> list[float]:
    values = cache.get(key, [])
    return [float(v) for v in values]


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _plot_cell_violin(
    ax: plt.Axes,
    values: list[float],
    *,
    color: str,
    y_min: float,
    y_max: float,
    annotate: bool = True,
) -> None:
    if not values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(y_min, y_max)
        return

    if HAS_SEABORN:
        sns.violinplot(
            y=values,
            color=color,
            inner="box",
            cut=0,
            linewidth=1.0,
            ax=ax,
        )
        for violin in ax.collections:
            violin.set_alpha(0.85)
            violin.set_edgecolor("#333333")
        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata) == 2 and abs(ydata[1] - ydata[0]) < 1e-9:
                line.set_color("white")
                line.set_linewidth(2.0)
                line.set_zorder(5)
    else:
        parts = ax.violinplot(values, positions=[0], showmeans=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.85)
            body.set_edgecolor("#333333")
        ax.boxplot(
            values,
            positions=[0],
            widths=0.08,
            patch_artist=False,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2.0},
        )

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.35, linestyle="-", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if annotate:
        med = median(values)
        y_annot = min(y_max - 0.02, max(y_min + 0.02, med))
        ax.annotate(
            f"{med:.3f}",
            xy=(0, y_annot),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.92,
            },
            zorder=7,
        )


def _plot_metric_row(
    ax: plt.Axes,
    caches: dict[str, dict],
    *,
    cache_key: str,
    y_min: float,
    y_max: float,
    title: str,
    y_ticks: list[float],
) -> None:
    labels = [METHOD_STYLE[m]["label"] for m in METHODS]
    colors = [METHOD_STYLE[m]["color"] for m in METHODS]
    plot_data: list[list[float]] = [
        metric_values(caches[method], cache_key) for method in METHODS
    ]

    if HAS_SEABORN:
        import pandas as pd

        rows: list[dict] = []
        for method, values in zip(METHODS, plot_data):
            label = METHOD_STYLE[method]["label"]
            for value in values:
                rows.append({"method": label, "value": value})
        plot_df = pd.DataFrame(rows)
        sns.violinplot(
            data=plot_df,
            x="method",
            y="value",
            hue="method",
            order=labels,
            hue_order=labels,
            palette=colors,
            inner="box",
            cut=0,
            linewidth=1.0,
            legend=False,
            ax=ax,
        )
        for violin in ax.collections[: len(METHODS)]:
            violin.set_alpha(0.85)
            violin.set_edgecolor("#333333")
        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata) == 2 and abs(ydata[1] - ydata[0]) < 1e-9:
                line.set_color("white")
                line.set_linewidth(2.0)
                line.set_zorder(5)
    else:
        positions = list(range(len(METHODS)))
        parts = ax.violinplot(
            plot_data, positions=positions, showmeans=False, showmedians=False
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.85)
            body.set_edgecolor("#333333")
        ax.boxplot(
            plot_data,
            positions=positions,
            widths=0.08,
            patch_artist=False,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2.0},
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    for i, method in enumerate(METHODS):
        med = median(plot_data[i])
        y_annot = min(y_max - 0.02, max(y_min + 0.02, med))
        ax.annotate(
            f"{med:.3f}",
            xy=(i, y_annot),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.92,
            },
            zorder=7,
        )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", alpha=0.35, linestyle="-", linewidth=0.6)
    ax.grid(axis="x", alpha=0.35, linestyle="-", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_single_metric(
    caches: dict[str, dict],
    metric_spec: tuple[str, str, str, float, float, list[float]],
    *,
    output_png: Path,
    output_pdf: Path,
    suptitle: str,
) -> None:
    _slug, title, cache_key, y_min, y_max, y_ticks = metric_spec
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    _plot_metric_row(
        ax,
        caches,
        cache_key=cache_key,
        y_min=y_min,
        y_max=y_max,
        title=title,
        y_ticks=y_ticks,
    )
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_3x3_grid(
    caches: dict[str, dict],
    *,
    output_png: Path,
    output_pdf: Path,
    suptitle: str,
) -> None:
    n_rows = len(METRICS)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.8 * n_rows),
        constrained_layout=True,
        sharex=False,
    )

    for row_i, (_slug, row_title, cache_key, y_min, y_max, y_ticks) in enumerate(METRICS):
        for col_i, method in enumerate(METHODS):
            ax = axes[row_i, col_i]
            color = METHOD_STYLE[method]["color"]
            label = METHOD_STYLE[method]["label"]
            values = metric_values(caches[method], cache_key)
            _plot_cell_violin(
                ax,
                values,
                color=color,
                y_min=y_min,
                y_max=y_max,
            )
            n = len(values)
            med = median(values)
            if row_i == 0:
                ax.set_title(f"{label}\n(n={n})", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{row_title}", fontsize=10)
            ax.set_yticks(y_ticks)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(caches: dict[str, dict], output_csv: Path) -> None:
    rows: list[dict] = []
    for method in METHODS:
        cache = caches[method]
        rows.append(
            {
                "method": method,
                "molecule_selection": cache.get("molecule_selection", ""),
                "n_episodes": len(cache.get("qed", [])),
                "qed_median": median(metric_values(cache, "qed")),
                "diversity_median": median(metric_values(cache, "diversity_bootstrap")),
                "sa_median": median(metric_values(cache, "sa")),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_csv, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="GenMolRL internal plot-cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PGFS_ROOT / "figures",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    out_dir = args.output_dir.resolve()
    caches = load_caches(cache_dir)

    suptitle = (
        "Internal test (Bi pool) — PGFS uses max-QED per episode; "
        "Original = start molecules"
    )

    for slug, *_rest in METRICS:
        plot_single_metric(
            caches,
            next(m for m in METRICS if m[0] == slug),
            output_png=out_dir / f"pgfs_internal_{slug}_violin.png",
            output_pdf=out_dir / f"pgfs_internal_{slug}_violin.pdf",
            suptitle=suptitle,
        )
        print(f"Wrote {out_dir / f'pgfs_internal_{slug}_violin.png'}")

    plot_3x3_grid(
        caches,
        output_png=out_dir / "pgfs_internal_3x3_violin.png",
        output_pdf=out_dir / "pgfs_internal_3x3_violin.pdf",
        suptitle=suptitle,
    )
    print(f"Wrote {out_dir / 'pgfs_internal_3x3_violin.png'}")

    summary_path = out_dir / "pgfs_internal_metrics.csv"
    write_summary_csv(caches, summary_path)
    print(f"Wrote {summary_path}")

    for method in METHODS:
        cache = caches[method]
        print(
            f"{method} [{cache.get('molecule_selection', '')}]: "
            f"qed={median(metric_values(cache, 'qed')):.4f} "
            f"div={median(metric_values(cache, 'diversity_bootstrap')):.4f} "
            f"sa={median(metric_values(cache, 'sa')):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
