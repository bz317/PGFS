#!/usr/bin/env python3
"""Download W&B metrics and plot PGFS training curves for the README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

RUNS = {
    "delta_qed": {
        "path": "/boqiaoz-cambridge/GenMolRL_Bi/runs/6ryqkars",
        "label": "ΔQED per step",
        "reward_desc": (
            "Per-step reward: QED(product_t) − QED(product_{t−1})  "
            "(config: reward: delta_qed)"
        ),
        "color": "#2563eb",
    },
    "qed": {
        "path": "/boqiaoz-cambridge/GenMolRL_Bi/runs/3d7j4vp2",
        "label": "QED per step",
        "reward_desc": (
            "Per-step reward: QED(product_t)  (config: reward: qed)"
        ),
        "color": "#dc2626",
    },
}

X_AXIS = "train/global_step"
TRAIN_METRIC = "train/mean_reward"
EVAL_METRIC = "eval/mean_reward"
EVAL_LABEL = "test/mean_reward"


def fetch_metric_history(run, metric: str, *, samples: int) -> pd.DataFrame:
    """Fetch one metric at a time so W&B does not merge sparse series."""
    history = run.history(
        keys=[metric, X_AXIS],
        pandas=True,
        samples=samples,
        x_axis=X_AXIS,
    )
    if history.empty:
        return history
    history = history.sort_values(X_AXIS).drop_duplicates(subset=X_AXIS, keep="last")
    return history[[X_AXIS, metric]].dropna(subset=[metric])


def fetch_run_histories(run, *, samples: int) -> dict[str, pd.DataFrame]:
    train = fetch_metric_history(run, TRAIN_METRIC, samples=samples)
    eval_df = fetch_metric_history(run, EVAL_METRIC, samples=samples)
    if not eval_df.empty:
        eval_df = eval_df.rename(columns={EVAL_METRIC: EVAL_LABEL})
    return {"train": train, "eval": eval_df}


def plot_combined(histories: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)

    panel_specs = [
        (TRAIN_METRIC, "Train mean reward"),
        (EVAL_LABEL, "Test mean reward"),
    ]

    max_step = max(
        max(
            (payload["train"][X_AXIS].max() if not payload["train"].empty else 0),
            (payload["eval"][X_AXIS].max() if not payload["eval"].empty else 0),
        )
        for payload in histories.values()
    )

    for ax, (metric_key, title) in zip(axes, panel_specs):
        for name, payload in histories.items():
            source = payload["train"] if metric_key == TRAIN_METRIC else payload["eval"]
            if source.empty or metric_key not in source.columns:
                continue
            series = source[[X_AXIS, metric_key]].dropna()
            ax.plot(
                series[X_AXIS],
                series[metric_key],
                label=payload["label"],
                color=RUNS[name]["color"],
                linewidth=1.2 if metric_key == TRAIN_METRIC else 1.8,
                alpha=0.85,
                marker="o" if metric_key == EVAL_LABEL else None,
                markersize=3 if metric_key == EVAL_LABEL else 0,
            )
        ax.set_title(title)
        ax.set_xlabel("Environment step (train/global_step)")
        ax.set_ylabel("Mean reward")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(
        f"PGFS paper-style training curves (bimolecular Bi setup, up to {max_step:,.0f} steps)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    combined_path = out_dir / "pgfs_training_curves.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {combined_path}")


def plot_individual(name: str, payload: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    meta = RUNS[name]

    panel_specs = [
        (payload["train"], TRAIN_METRIC, "Train mean reward", False),
        (payload["eval"], EVAL_LABEL, "Test mean reward", True),
    ]

    for ax, (df, metric_key, title, use_markers) in zip(axes, panel_specs):
        series = df[[X_AXIS, metric_key]].dropna()
        ax.plot(
            series[X_AXIS],
            series[metric_key],
            color=meta["color"],
            linewidth=1.2 if not use_markers else 1.8,
            marker="o" if use_markers else None,
            markersize=3 if use_markers else 0,
        )
        ax.set_title(title)
        ax.set_xlabel("Environment step (train/global_step)")
        ax.set_ylabel("Mean reward")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(f"PGFS — {meta['label']}", fontsize=12, y=1.02)
    fig.text(0.5, -0.02, meta["reward_desc"], ha="center", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / f"pgfs_{name}_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def save_csv(name: str, payload: dict, out_dir: Path) -> None:
    train = payload["train"].rename(columns={TRAIN_METRIC: "train/mean_reward"})
    eval_df = payload["eval"].rename(columns={EVAL_LABEL: "test/mean_reward"})
    merged = pd.merge(
        train,
        eval_df,
        on=X_AXIS,
        how="outer",
    ).sort_values(X_AXIS)
    csv_path = out_dir / f"pgfs_{name}_metrics.csv"
    merged.to_csv(csv_path, index=False)
    print(
        f"Wrote {csv_path} "
        f"(train={len(train)}, test={len(eval_df)}, merged={len(merged)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Max points per series from W&B history API (train is downsampled).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    api = wandb.Api()
    for name, meta in RUNS.items():
        run = api.run(meta["path"])
        payload = fetch_run_histories(run, samples=args.samples)
        histories[name] = {
            **payload,
            "run": run,
            "label": meta["label"],
        }
        max_step = max(
            payload["train"][X_AXIS].max() if not payload["train"].empty else 0,
            payload["eval"][X_AXIS].max() if not payload["eval"].empty else 0,
        )
        print(
            f"{name}: run={run.name}, id={run.id}, reward={run.config.get('reward', '?')}, "
            f"train_pts={len(payload['train'])}, test_pts={len(payload['eval'])}, "
            f"max_step={max_step:.0f}"
        )
        save_csv(name, payload, args.out_dir)
        plot_individual(name, payload, args.out_dir)

    plot_combined(histories, args.out_dir)


if __name__ == "__main__":
    main()
