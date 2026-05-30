"""Unified molecule-design environment for PPO, A2C, and TD3/PGFS."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pgfs.chem.datasets import load_pickle
from pgfs.chem.fingerprints import morgan_fp_array  # noqa: F401  (kept for backwards compat re-exports)
from pgfs.chem.reaction_manager import BI_TYPE, ReactionManager
from pgfs.chem.representations import make_representation
from pgfs.config import resolve_path
from pgfs.algorithms.td3.constants import TD3_UNI_DISCRETE_ACTION_DESIGN
from pgfs.envs.action_spaces import ActionSpaceSpec
from pgfs.envs.masking import MaskProvider
from pgfs.envs.rewards import RewardFunction, qed
from pgfs.envs.start_strategies import StartStrategy

logger = logging.getLogger(__name__)

_RLV2_ALIASES = frozenset({"rlv2", "moldset"})


def _needs_rlv2(name: str | None) -> bool:
    return bool(name and name.strip().lower() in _RLV2_ALIASES)


def _rlv2_fit_smiles(
    rep_names: list[str],
    rlv2_train_file: str | None,
    reactants: dict,
) -> list[str] | None:
    if not any(_needs_rlv2(name) for name in rep_names):
        return None
    if rlv2_train_file is not None and not Path(str(rlv2_train_file) + ".rlv2_norm.npz").exists():
        try:
            train_pool = load_pickle(Path(rlv2_train_file))
            return list(train_pool.keys())
        except FileNotFoundError:
            return list(reactants.keys())
    if rlv2_train_file is None:
        return list(reactants.keys())
    return None


class MoleculeDesignEnv(gym.Env):
    """Configurable reaction-template environment.

    `sb3_multidiscrete` intentionally preserves the current factorized
    `MultiDiscrete([T, R2])` design for Bi mode.
    """

    metadata = {"render_modes": ["human", "console", "rgb_array"]}

    def __init__(
        self,
        reactant_file: str,
        template_file: str,
        reaction_mode: str = "uni",
        algorithm_family: str = "sb3_discrete",
        action_design: str = "discrete",
        masking: str = "substructure",
        reward: str = "delta_qed",
        max_steps: int = 5,
        render_mode: str | None = None,
        start_strategy: str = "random_pool",
        start_smiles_file: str | None = None,
        fixed_start_smiles: str | None = None,
        use_stop_action: bool = True,
        stop_early_penalty: float = 0.0,
        stop_penalty_until_step: int = -1,
        invalid_reaction_penalty: float = -1.0,
        reward_round_digits: int | None = None,
        info_qed_round_digits: int | None = None,
        append_action_mask_to_obs: bool | None = None,
        start_pool_file: str | None = None,
        molecule_representation: str = "morgan",
        state_representation: str | None = None,
        r2_representation: str | None = None,
        rlv2_norm_training_file: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.current_state: str | None = None
        self.previous_state: str | None = None
        self.initial_qed = 0.0
        self.steps_log: dict[int, dict] = {}
        self.reaction_mode = reaction_mode
        self.algorithm_family = algorithm_family
        self.action_design = action_design
        self.use_stop_action = bool(use_stop_action)
        self.stop_early_penalty = float(stop_early_penalty)
        self.stop_penalty_until_step = int(stop_penalty_until_step)

        self.reactants = load_pickle(Path(resolve_path(reactant_file)))
        raw_templates = load_pickle(Path(resolve_path(template_file)))
        all_reactions = ReactionManager(raw_templates, self.reactants)
        self.templates = all_reactions.templates_for_mode(reaction_mode)
        self.reaction_manager = ReactionManager(self.templates, self.reactants)
        self.num_templates = len(self.templates)
        self.reactant_keys = list(self.reactants.keys())

        # Pluggable molecule representations (morgan / maccs / rlv2).
        # ``molecule_representation`` sets BOTH state (R(1) obs) and R(2) action
        # vectors — the default for PPO/A2C/GraphTrans and legacy Bi-TD3 runs.
        # PGFS-only split knobs ``state_representation`` / ``r2_representation``
        # decouple the axes (paper: ECFP state + RLV2 action).
        if state_representation is not None or r2_representation is not None:
            if algorithm_family != "td3_pgfs" or reaction_mode != "bi":
                raise ValueError(
                    "state_representation / r2_representation require "
                    "algorithm_family=td3_pgfs and reaction_mode=bi"
                )
            state_name = state_representation or molecule_representation
            r2_name = r2_representation or molecule_representation
        else:
            state_name = molecule_representation
            r2_name = molecule_representation

        rlv2_train_file = (
            resolve_path(rlv2_norm_training_file)
            if rlv2_norm_training_file is not None
            else None
        )
        rlv2_fit_smiles = _rlv2_fit_smiles([state_name, r2_name], rlv2_train_file, self.reactants)
        rep_kwargs = {
            "training_file": rlv2_train_file,
            "training_smiles": rlv2_fit_smiles,
        }
        self.state_representation = make_representation(state_name, **rep_kwargs)
        self.r2_representation = make_representation(r2_name, **rep_kwargs)
        # Back-compat aliases: ``representation`` drives state obs; when the
        # split knobs are unset both axes share the same bundle.
        self.representation = self.state_representation
        self.molecule_representation_name = self.state_representation.name
        self.molecule_representation_is_binary = bool(self.state_representation.is_binary)
        self.r2_representation_name = self.r2_representation.name
        self.r2_representation_dim = int(self.r2_representation.dim)
        self.r2_representation_is_binary = bool(self.r2_representation.is_binary)

        # Rebuild reactant vectors for the R(2) FAISS index / warm-up replay
        # entries. Pickled values are Morgan FPs; replace when R(2) ≠ morgan.
        if self.r2_representation.name != "morgan":
            for smiles in self.reactant_keys:
                self.reactants[smiles] = self.r2_representation.fn(smiles)

        self.mask_provider = MaskProvider(masking, use_stop_action=self.use_stop_action)
        self.reward_fn = RewardFunction(
            reward,
            invalid_penalty=invalid_reaction_penalty,
            round_digits=reward_round_digits,
            qed_round_digits=info_qed_round_digits,
        )
        self.info_qed_round_digits = info_qed_round_digits
        # `start_pool_file` lets the eval env source start molecules from a
        # different pickle than the R2 bank in `self.reactants`. This is
        # required for `sb3_multidiscrete` (bi-reaction) so the model's R2
        # action head — sized to the training reactant pool — keeps the same
        # dimension when evaluated on a held-out start pool. When None the
        # legacy behaviour is preserved: start molecules are sampled from the
        # same dict that backs R2 selection.
        start_pool = (
            load_pickle(Path(resolve_path(start_pool_file)))
            if start_pool_file is not None
            else self.reactants
        )
        self.start_strategy = StartStrategy(start_strategy, fixed_start_smiles, start_smiles_file)
        self.start_strategy.initialize(start_pool)

        spec = ActionSpaceSpec(
            family=algorithm_family,
            reaction_mode=reaction_mode,
            action_design=action_design,
            use_stop_action=self.use_stop_action,
        )
        self.action_space = spec.build(self.num_templates, len(self.reactants))
        if append_action_mask_to_obs is None:
            append_action_mask_to_obs = algorithm_family in {"sb3_discrete", "sb3_multidiscrete"}
        self.append_action_mask_to_obs = bool(append_action_mask_to_obs)
        self.base_obs_dim = int(self.state_representation.dim)
        mask_dim = len(self.action_masks()) if self.append_action_mask_to_obs else 0
        # Binary state reps (morgan / maccs) live in {0, 1}^d; continuous RLV2
        # state lives in ~[-1, +1]^d after the normaliser.
        obs_low = 0.0 if self.state_representation.is_binary else -1.0
        self.observation_space = spaces.Box(
            low=obs_low,
            high=1.0,
            shape=(self.base_obs_dim + mask_dim,),
            dtype=np.float32,
        )

        if self.algorithm_family == "td3_pgfs":
            _allowed_td3_designs = {"pgfs_continuous_r2", TD3_UNI_DISCRETE_ACTION_DESIGN}
            if self.action_design not in _allowed_td3_designs:
                raise ValueError(
                    f"td3_pgfs requires env.action_design in {_allowed_td3_designs}, got {self.action_design!r}"
                )
            if self.action_design == TD3_UNI_DISCRETE_ACTION_DESIGN and self.reaction_mode != "uni":
                raise ValueError(
                    f"env.action_design {TD3_UNI_DISCRETE_ACTION_DESIGN!r} requires reaction_mode: uni"
                )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_log = {}
        self.current_state = self.start_strategy.sample(self)
        if not self.start_strategy.validate(self.current_state):
            raise ValueError(f"Invalid start molecule: {self.current_state}")
        self.previous_state = self.current_state
        self.initial_qed = qed(self.current_state)
        return self._get_obs(), self._get_info()

    def action_masks(self) -> np.ndarray:
        if self.algorithm_family == "sb3_multidiscrete":
            return self.mask_provider.multidiscrete_mask(self.reaction_manager, self.current_state)
        return self.mask_provider.template_mask_with_stop(self.reaction_manager, self.current_state)

    def _get_obs(self) -> np.ndarray:
        obs = self.state_representation.fn(self.current_state).astype(np.float32, copy=False)
        if not self.append_action_mask_to_obs:
            return obs
        mask = self.action_masks().astype(np.float32)
        return np.concatenate([obs, mask]).astype(np.float32, copy=False)

    def _get_info(self) -> dict:
        q = qed(self.current_state)
        if self.info_qed_round_digits is not None:
            q = round(q, int(self.info_qed_round_digits))
        return {
            "SMILES": self.current_state,
            "QED": q,
            "initial_QED": self.initial_qed,
            "step": self.current_step,
        }

    def _parse_action(self, action) -> tuple[int, str | None]:
        if self.algorithm_family == "td3_pgfs":
            if isinstance(action, tuple):
                template = action[0]
                if hasattr(template, "detach"):
                    template = int(template.detach().reshape(-1).argmax().item())
                r2 = action[1]
                if hasattr(r2, "detach"):
                    return int(template), None
                return int(template), r2
            return int(action), None
        if self.algorithm_family == "sb3_multidiscrete":
            template_index = int(action[0])
            reactant_index = int(action[1])
            if template_index >= self.num_templates:
                return template_index, None
            template = self.templates[template_index]
            if template.get("type") == BI_TYPE:
                return template_index, self.reactant_keys[reactant_index]
            return template_index, None
        return int(action), None

    def step(self, action):
        self.current_step += 1
        template_index, r2 = self._parse_action(action)

        if self.use_stop_action and template_index == self.num_templates:
            feasible_count = int(
                self.reaction_manager.get_mask(
                    self.current_state, kind=self.mask_provider.mode
                ).sum().item()
            )
            reward = self.reward_fn.stop_reward(
                current_step=self.current_step,
                stop_early_penalty=self.stop_early_penalty,
                stop_penalty_until_step=self.stop_penalty_until_step,
                feasible_template_count=feasible_count,
            )
            info = self._get_info()
            info.update(
                {
                    "stop": True,
                    "stop_reward": reward,
                    "stop_feasible_count": feasible_count,
                }
            )
            return self._get_obs(), reward, False, True, info

        template = self.templates.get(template_index)
        if template is None:
            info = self._get_info()
            info["bad_template_index"] = template_index
            return self._get_obs(), self.reward_fn.invalid_penalty, False, True, info

        previous_state = self.current_state
        new_state = self.reaction_manager.apply_reaction(previous_state, template, r2)
        if new_state is None:
            self.current_state = None
            info = self._get_info()
            info["reaction_failed"] = True
            return np.zeros(self.observation_space.shape[0], dtype=np.float32), self.reward_fn.invalid_penalty, False, True, info

        self.steps_log[self.current_step] = {
            "r1": previous_state,
            "template": template.get("name", str(template_index)),
            "r2": r2,
            "product": new_state,
        }
        self.previous_state = previous_state
        self.current_state = new_state
        reward = self.reward_fn.step_reward(previous_state, new_state)
        terminated = self.current_step >= self.max_steps
        has_next_template = bool(
            self.reaction_manager.feasible_first_reactant_templates(
                new_state,
                kind=self.mask_provider.mode,
            )
        )
        truncated = (not has_next_template) and not self.use_stop_action
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "console":
            for step, item in self.steps_log.items():
                print(f"Step {step}: {item}")
        return None

    def close(self):
        return None
