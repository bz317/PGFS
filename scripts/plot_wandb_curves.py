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

METRICS = {
    "train/mean_reward": "train/mean_reward",
    "eval/mean_reward": "test/mean_reward",
}


def fetch_history(run_path: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(run_path)
    keys = list(METRICS.keys()) + ["_step"]
    history = run.history(keys=keys, pandas=True, samples=5000)
    if history.empty:
        raise RuntimeError(f"No history for {run_path}")
    history = history.sort_values("_step").drop_duplicates(subset="_step", keep="last")
    history = history.rename(columns={"eval/mean_reward": "test/mean_reward"})
    return history, run


def plot_combined(histories: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)

    panel_specs = [
        ("train/mean_reward", "Train mean reward"),
        ("test/mean_reward", "Test mean reward"),
    ]

    max_step = max(
        payload["history"]["_step"].max() for payload in histories.values()
    )

    for ax, (metric_key, title) in zip(axes, panel_specs):
        for name, payload in histories.items():
            df = payload["history"]
            if metric_key not in df.columns:
                continue
            series = df[["_step", metric_key]].dropna()
            ax.plot(
                series["_step"],
                series[metric_key],
                label=payload["label"],
                color=RUNS[name]["color"],
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xlabel("Environment step")
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
    df = payload["history"]

    panel_specs = [
        ("train/mean_reward", "Train mean reward"),
        ("test/mean_reward", "Test mean reward"),
    ]

    for ax, (metric_key, title) in zip(axes, panel_specs):
        series = df[["_step", metric_key]].dropna()
        ax.plot(
            series["_step"],
            series[metric_key],
            color=meta["color"],
            linewidth=1.8,
        )
        ax.set_title(title)
        ax.set_xlabel("Environment step")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for name, meta in RUNS.items():
        history, run = fetch_history(meta["path"])
        histories[name] = {
            "history": history,
            "run": run,
            "label": meta["label"],
        }
        print(
            f"{name}: run={run.name}, id={run.id}, config reward={run.config.get('reward', '?')}, "
            f"steps={history['_step'].max():.0f}"
        )
        csv_path = args.out_dir / f"pgfs_{name}_metrics.csv"
        history.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
        plot_individual(name, histories[name], args.out_dir)

    plot_combined(histories, args.out_dir)


if __name__ == "__main__":
    main()
