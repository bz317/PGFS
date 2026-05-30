"""Shared trainer helpers for PGFS."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from pgfs.config import project_root, resolve_path
from pgfs.logging.wandb_metrics import define_ppo_compatible_metrics
from pgfs.registry import register_envs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(run_id: str) -> Path:
    path = project_root() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_wandb_resume_env() -> None:
    value = os.environ.get("WANDB_RESUME")
    if value is not None and value not in {"allow", "must", "never", "auto"}:
        os.environ.pop("WANDB_RESUME", None)


def init_wandb(config: dict, algorithm: str, experiment_name: str):
    _sanitize_wandb_resume_env()
    project = os.getenv("WANDB_PROJECT", config.get("project", "PGFS"))
    init_kw = {
        "project": project,
        "name": experiment_name,
        "job_type": f"train-{algorithm.lower()}",
        "save_code": True,
        "resume": "allow" if config.get("wandb_resume") else "never",
        "config": config,
    }
    entity = os.getenv("WANDB_ENTITY") or config.get("entity")
    if entity:
        init_kw["entity"] = entity
    run_id = (config.get("training") or {}).get("run_id")
    if config.get("wandb_resume") and run_id:
        init_kw["id"] = str(run_id)
        init_kw["resume"] = "must"
    run = wandb.init(**init_kw)
    define_ppo_compatible_metrics()
    return run


def env_kwargs(config: dict, *, eval_env: bool = False) -> dict:
    dataset = config["dataset"]
    env_cfg = config["env"]
    max_episode_len = config.get(
        "max_episode_len", env_cfg.get("max_episode_len", env_cfg.get("max_steps", 5))
    )
    algorithm_family = env_cfg["algorithm_family"]
    eval_r2_pool = str(dataset.get("eval_r2_pool", "test")).lower()
    if eval_r2_pool not in {"test", "train"}:
        raise ValueError(
            f"dataset.eval_r2_pool must be 'test' or 'train', got {eval_r2_pool!r}"
        )
    if eval_env and eval_r2_pool == "train":
        reactant_file = dataset["training_file"]
        start_pool_file = dataset.get("test_file")
    else:
        reactant_file = dataset["training_file"] if not eval_env else dataset.get("test_file")
        start_pool_file = None
    if reactant_file is None:
        raise KeyError("dataset.test_file must be set for evaluation environments")
    kwargs = {
        "reactant_file": resolve_path(reactant_file),
        "template_file": resolve_path(dataset["templates_file"]),
        "reaction_mode": config["reaction_mode"],
        "algorithm_family": algorithm_family,
        "action_design": env_cfg.get("action_design", "discrete"),
        "masking": config["masking"],
        "reward": config["reward"],
        "max_steps": max_episode_len,
        "use_stop_action": env_cfg.get("use_stop_action", True),
        "stop_early_penalty": env_cfg.get("stop_early_penalty", 0.0),
        "stop_penalty_until_step": env_cfg.get("stop_penalty_until_step", -1),
        "invalid_reaction_penalty": env_cfg.get("invalid_reaction_penalty", -1.0),
        "reward_round_digits": env_cfg.get("reward_round_digits"),
        "info_qed_round_digits": env_cfg.get("info_qed_round_digits"),
        "render_mode": "human" if eval_env else None,
        "append_action_mask_to_obs": env_cfg.get("append_action_mask_to_obs"),
        "molecule_representation": env_cfg.get("molecule_representation", "morgan"),
        "state_representation": env_cfg.get("state_representation"),
        "r2_representation": env_cfg.get("r2_representation"),
        "rlv2_norm_training_file": resolve_path(dataset["training_file"]),
    }
    if start_pool_file is not None:
        kwargs["start_pool_file"] = resolve_path(start_pool_file)
    if eval_env:
        kwargs["start_strategy"] = "cycle_pool"
    elif dataset.get("fixed_start_smiles"):
        kwargs["start_strategy"] = "fixed"
        kwargs["fixed_start_smiles"] = dataset["fixed_start_smiles"]
    else:
        kwargs["start_strategy"] = dataset.get("start_strategy", "random_pool")
        if dataset.get("start_smiles_file"):
            kwargs["start_smiles_file"] = resolve_path(dataset["start_smiles_file"])
    return kwargs


__all__ = ["set_seed", "run_dir", "init_wandb", "env_kwargs", "register_envs"]
