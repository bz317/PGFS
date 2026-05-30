#!/usr/bin/env python3
"""Train PGFS (TD3 + kNN) on bimolecular reaction templates."""

from __future__ import annotations

import argparse
import os

from pgfs.algorithms.td3.train import train
from pgfs.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PGFS (Policy Gradient on Feasible Syntheses)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--reward", choices=["delta_qed", "final_qed", "qed"])
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-episode-len", type=int)
    parser.add_argument("--resume-checkpoint", help="TD3 checkpoint_<N>.tar to resume from")
    parser.add_argument("--run-id", help="Existing runs/<id> directory (and W&B id when resuming)")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--wandb-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["algorithm"] = "TD3"
    if args.reward:
        config["reward"] = args.reward
    if args.max_episode_len is not None:
        config["max_episode_len"] = args.max_episode_len
    if args.resume_checkpoint:
        config.setdefault("training", {})["resume_checkpoint"] = args.resume_checkpoint
    if args.run_id:
        config.setdefault("training", {})["run_id"] = args.run_id
    if args.total_timesteps is not None:
        config.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    if args.wandb_resume:
        config["wandb_resume"] = True

    experiment_name = args.experiment_name or config.get(
        "experiment_name",
        f"PGFS_{config['reaction_mode']}_{config['reward']}",
    )
    suffix = os.environ.get("WANDB_RUN_POST_APPEND", "").strip()
    if suffix and not experiment_name.endswith(f"_{suffix}"):
        experiment_name = f"{experiment_name}_{suffix}"
    os.environ.setdefault("WANDB_PROJECT", config.get("project", "PGFS"))
    train(config, experiment_name)


if __name__ == "__main__":
    main()
