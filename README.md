# PGFS —  Policy Gradient for Forward Synthesis

## News

We have released **PGFS++** at [https://github.com/bz317/PGFS_plus_plus](https://github.com/bz317/PGFS_plus_plus).

Successful? reproduction of **Policy Gradient for Forward Synthesis (PGFS)** from the paper:

> **Learning to Navigate the Synthetically Accessible Chemical Space Using Reinforcement Learning**  
> Gottipati et al., 2020 — [arXiv:2004.12485](https://arxiv.org/pdf/2004.12485)

This repo implements the **bimolecular** setup from §4.3, Algorithm 1, and Figure 2 of that paper.

Both shipped configs use the **same paper-style setup** (ECFP state, RLV2 action, kNN k=1, `r2_available` masking for **T_mask**, no Stop, horizon 5) — see [Hyperparameters (§4.3)](#hyperparameters-paper-43). They differ only in the **per-step reward signal** — see [Reward modes](#reward-modes) below.

## Contents

- [Performance](#performance)
- [Workflow](#workflow)
  - [1. Install](#1-install)
  - [2. Run](#2-run)
  - [3. Results](#3-results)
- [Reward modes](#reward-modes)
- [Internal test evaluation](#internal-test-evaluation)
- [What this implements](#what-this-implements)
- [Hyperparameters (paper §4.3)](#hyperparameters-paper-43)
  - [Difference from the original paper](#difference-from-the-original-paper)
- [Repository layout](#repository-layout)
  - [Bundled data (`data/Bi/`)](#bundled-data-databi)
- [Installation details](#installation-details)
  - [Option A — conda (recommended)](#option-a--conda-recommended)
  - [Option B — pip only](#option-b--pip-only)
- [Reference](#reference)
- [License](#license)

---

## Performance

<p align="center">
  <img src="figures/qed_panel.png" width="90%" alt="Performance of PGFS and its variants">
</p>

<p align="center">
  <em>Performance of PGFS and its variants.</em>
</p>

---

## Workflow

### 1. Install

```bash
git clone https://github.com/bz317/PGFS.git PGFS
cd PGFS

# One-shot conda env (PyTorch + CUDA 12.1, RDKit, FAISS-GPU, …)
conda env create -f env.yml
conda activate pgfs

# Install this package in editable mode
pip install -e .
```

**Smoke test** (no W&B account needed):

```bash
WANDB_MODE=offline python scripts/train.py \
  --config configs/paper_style_delta_qed.yaml \
  --total-timesteps 5000
```

**CPU-only:** edit `env.yml` — drop `pytorch-cuda=12.1`, replace `faiss-gpu` with `faiss-cpu`, then recreate the env.

**Optional:** log in to Weights & Biases for online metrics (`wandb login`).

### 2. Run

**Local training** (default: **ΔQED per step**, paper §4.3 hyperparameters, 1M env steps):

```bash
conda activate pgfs
cd PGFS
python scripts/train.py --config configs/paper_style_delta_qed.yaml
```

**Reward variants** (identical architecture / hyperparameters; only `reward` in the YAML changes):

```bash
# (1) ΔQED per step — default, recommended for molecule improvement
python scripts/train.py --config configs/paper_style_delta_qed.yaml

# (2) QED per step — absolute QED of each product (dense, paper-style horizon)
python scripts/train.py --config configs/paper_style_qed.yaml
```

Other launch options:

```bash
# Wrapper script (activates conda, sets PYTHONPATH)
bash run_launcher/run_train.sh

# HPC (SLURM, 1× GPU, 36 h) — default config is paper_style_delta_qed.yaml
sbatch run_launcher/HPC/slurm_gpu_paper_style

# Absolute-QED reward on SLURM
CONFIG=configs/paper_style_qed.yaml sbatch run_launcher/HPC/slurm_gpu_paper_style
```

**Resume** from a checkpoint:

```bash
python scripts/train.py --config configs/paper_style_delta_qed.yaml \
  --resume-checkpoint runs/<run_id>/td3_checkpoints/checkpoint_100000.tar \
  --run-id <run_id>
```

### 3. Results

Training creates three output locations under the repo root:

```text
PGFS/
├── runs/
│   └── <wandb_run_id>/              # one directory per training run
│       └── td3_checkpoints/
│           ├── checkpoint_100000.tar   # periodic saves (model_save_freq)
│           ├── checkpoint_200000.tar
│           └── final_model.pth         # written at end of training
├── wandb/                           # local W&B sync (if WANDB_MODE=online)
│   └── run-<timestamp>-<id>/
│       ├── files/config.yaml
│       └── logs/
└── logs/                            # SLURM / launcher tee logs
    └── pgfs_paper_style_<jobid>.log
```

**Checkpoints** (`*.tar` / `final_model.pth`) contain actor + critic weights, optimizer state, training step count, and (optionally) replay buffer.

**W&B metrics** (when online) include:

| Namespace | Examples |
|-----------|----------|
| `train/` | `mean_reward`, `global_step`, `critic_loss`, `actor_loss`, `temperature` |
| `eval/` | `mean_reward` (test pool return), `mean_final_delta_qed`, `mean_final_qed`, `mean_ep_length`, `max_qed`, `n_molecules` |

Primary molecule-quality metric: **`eval/mean_final_delta_qed`** (mean QED improvement over the held-out test reactant pool).

---

## Reward modes

| Mode | Config key | Per-step reward | Config file |
|------|------------|-----------------|-------------|
| **(1) ΔQED per step** | `reward: delta_qed` | `QED(product_t) − QED(product_t-1)` — rewards *improvement* at each reaction | `configs/paper_style_delta_qed.yaml` |
| **(2) QED per step** | `reward: qed` | `QED(product_t)` — rewards *absolute* drug-likeness of each intermediate/final product | `configs/paper_style_qed.yaml` |

Override from the CLI: `--reward delta_qed` or `--reward qed`.

---

## Internal test evaluation

Held-out **Bi test pool** (~12.7k start molecules). Violins compare **Original** (unmodified starts) with the two PGFS reward modes trained to 1M steps:

| Method | W&B run | Reward (per step) |
|--------|---------|-------------------|
| **PGFS (qed)** | [`3d7j4vp2`](https://wandb.ai/boqiaoz-cambridge/GenMolRL_Bi/runs/3d7j4vp2) | `QED(product_t)` |
| **PGFS (δ_qed)** | [`6ryqkars`](https://wandb.ai/boqiaoz-cambridge/GenMolRL_Bi/runs/6ryqkars) | `QED(product_t) − QED(product_{t−1})` |

**Metrics** (paper-aligned molecule selection):

| Metric | Original | PGFS (qed) | PGFS (δ_qed) |
|--------|----------|------------|--------------|
| **QED ↑** (median) | 0.703 | **0.812** | 0.810 |
| **Diversity ↑** (median) | 0.896 | 0.867 | **0.876** |
| **SA ↓** (median) | **2.435** | 3.288 | 3.345 |

- **QED** — PGFS reports the **max QED per episode** (best molecule along the 5-step trajectory), matching the paper; Original uses start-molecule QED.
- **Diversity** — bootstrap structural diversity (Morgan FP, Tanimoto; 400 bootstrap samples of 1000 molecules).
- **SA** — synthetic accessibility of the max-QED molecule (PGFS) or start molecule (Original); lower = easier to synthesize.

**3×3 grid** (rows: QED / Diversity / SA; columns: Original / PGFS (qed) / PGFS (δ_qed)):

![PGFS internal test: QED, diversity, and SA](figures/pgfs_internal_3x3_violin.png)

**Per-metric violins** (three methods side-by-side):

| QED ↑ | Diversity ↑ | SA ↓ |
|-------|-------------|------|
| ![QED](figures/pgfs_internal_qed_violin.png) | ![Diversity](figures/pgfs_internal_diversity_violin.png) | ![SA](figures/pgfs_internal_sa_violin.png) |

Regenerate from the GenMolRL plot cache (sibling repo; no re-parsing of detailed logs):

```bash
# From PGFS/ — expects ../GenMolRL/run_detailed_results/experiments_vis/plot_cache/internal/
python scripts/plot_internal_metrics_violins.py
```

Override cache location if needed: `--cache-dir /path/to/plot_cache/internal`.

---

## What this implements

PGFS learns a policy over **reaction templates** T and **second reactants** R(2) (building blocks) under synthesis constraints. At each step the agent observes the current molecule R(1), picks a template, outputs a continuous RLV2 vector for R(2), and kNN maps that vector to a discrete building block.

See [Hyperparameters (§4.3)](#hyperparameters-paper-43) for the full paper-matched settings and how **T_mask** is built.

---

## Hyperparameters (paper §4.3)

Both configs (`paper_style_delta_qed.yaml`, `paper_style_qed.yaml`) share the settings below. Only `reward` differs — see [Reward modes](#reward-modes).

**Environment and synthesis graph**

- `reaction_mode: bi` — bimolecular template pool (unimolecular + bimolecular templates together).
- `max_episode_len: 5` — fixed synthesis depth; no learned early stopping.
- `use_stop_action: false` — no Stop action in the action space (paper PGFS).
- `invalid_reaction_penalty: -1.0` — reward when forward synthesis fails.
- `action_design: pgfs_continuous_r2` — discrete template via `f` + Gumbel-Softmax; continuous RLV2 vector via `π`; kNN maps the vector to a discrete building block.

**State and action representations** (YAML keys → paper names)

- `state_representation: morgan` → **Morgan ECFP fingerprint (ECFP4)** for R(1): radius 2, 1024 bits, binary 1024-d vector (paper §4.3 “ECFP state”).
- `r2_representation: rlv2` → **RLV2 / MolDSet** for R(2): 35 RDKit descriptors (PGFS Appendix A), z-scored using statistics fit on the training building-block pool (`reactants_train.pkl.rlv2_norm.npz`), output in ~[-1, +1] per dimension.
- `append_action_mask_to_obs: false` — **T_mask** is applied inside the actor (logit masking), not concatenated onto the state vector.

**Template masking — T_mask** (`masking: r2_available`, Figure 2)

- We build **T_mask** with the **`r2_available`** check: from current R(1), mark each template feasible if R(1) matches its first-reactant pattern and (for bimolecular templates) at least one R(2) building block matches the second-reactant pattern. Masked logits use **T ← T ⊙ T_mask** (element-wise product) before Gumbel-Softmax.

**kNN second-reactant retrieval** (Algorithm 1)

- `knn_top_k: 1` — retrieve the single nearest building block in RLV2 space (paper: k = 1).
- `knn_score_mode: product` — forward-react R(1) + T + R(2)_i for each candidate; score the **product** with the active reward function; **hard argmax** over products.
- `knn_random_epsilon: 0.0` — no ε-greedy tie-breaking on kNN (PGFS hard argmax).

**Neural networks** — four FC layers each (3 hidden ReLU + final activation; §4.3)

- **Template head `f`** (`f_hidden_dims: [256, 128, 128]`, `f_final_activation: tanh`)
  - Input: ECFP state (1024-d).
  - Hidden: 256 → 128 → 128, ReLU.
  - Output: template logits, **tanh** → masked Gumbel-Softmax sample over templates.
- **R(2) head `π`** (`pi_hidden_dims: [256, 256, 167]`)
  - Input: ECFP state + one-hot selected template.
  - Hidden: 256 → 256 → 167, ReLU.
  - Output: **Linear(167 → 35) + tanh** → continuous RLV2 action vector.
- **Critic `Q`** — **twin** networks (`critic_hidden_dims: [256, 64, 16]`, TD3)
  - Input: ECFP state + template one-hot + RLV2 vector (35-d).
  - Hidden: 256 → 64 → 16, ReLU.
  - Output: scalar Q, **linear** (no tanh on Q).
- **Auxiliary template loss:** `f_ce_loss_coef: 1.0` — cross-entropy on stored template indices (Algorithm 1, line 21).

**TD3 / policy optimization** (§4.3 + Algorithm 1)

- Optimizer: **Adam** — `actor_lr: 1e-4` on `f` + `π`, `critic_lr: 3e-4` on twin `Q`.
- `gamma: 0.99`, `tau: 0.005` — discount and target soft-update.
- `batch_size: 32`, `buffer_size: 1000000` — replay batch and buffer capacity.
- `policy_freq: 2` — delayed policy update (TD3).
- `policy_noise: 0.2`, `noise_clip: 0.2` — target policy smoothing (clip ±0.2).
- `noise_std: 0.1` — Gaussian exploration noise on the **π** output, N(0, 0.1).
- `initial_temperature: 1.0`, `min_temperature: 0.1` — Gumbel-Softmax temperature annealed exponentially from 1.0 → 0.1 over training.
- `symmetric_target_actor: true` — target actor uses the same Gumbel-Softmax procedure as the online actor (Algorithm 1, line 17), not deterministic argmax.
- `start_timesteps: 3000` — random-action warm-up before gradient updates (paper: 3k steps).
- `warmup_stop_probability: 0.0` — no Stop sampling during warm-up (consistent with `use_stop_action: false`).
- `training_random_action_prob: 0.0` — no ε-greedy template exploration after warm-up.
- `entropy_regularization: false`, `auto_tune_alpha: false` — standard TD3 (not SAC-style entropy tuning).

**Training molecule protocol**

- `start_strategy: random_pool` — training episodes start from a random building block in `reactants_train.pkl`.
- `eval_r2_pool: train` — at evaluation, R(2) candidates come from the training building-block pool (paper-compatible).

### Difference from the original paper

1. **Eval protocol**: full test reactant pool (~12k molecules) one-by-one eval (paper uses a random subset of 2k molecules).

---

## Repository layout

```text
PGFS/
├── env.yml                # conda environment (recommended install path)
├── pyproject.toml         # pip install -e .
├── requirements.txt       # pip-only fallback
├── pgfs/                  # Python package (env, TD3 agent, kNN, chem)
├── configs/               # paper_style_delta_qed.yaml, paper_style_qed.yaml
├── data/Bi/               # bundled reactants + templates (~11 MB)
├── scripts/train.py       # CLI entry point
├── scripts/plot_wandb_curves.py  # optional W&B training-curve plots
├── scripts/plot_internal_metrics_violins.py  # QED / diversity / SA violins (README)
├── figures/               # performance panel + internal-test violins (for README)
├── run_launcher/          # run_train.sh + HPC/slurm_gpu_paper_style
├── runs/                  # checkpoints (created at train time)
├── wandb/                 # W&B local files (created at train time)
└── logs/                  # SLURM logs (created at submit time)
```

### Bundled data (`data/Bi/`)

| File | Description |
|------|-------------|
| `reactants_train.pkl` | Training building blocks (R2 pool + training starts) |
| `reactants_test.pkl` | Held-out test reactants (eval R1 cycle) |
| `templates.pkl` | Reaction templates (uni + bi) |
| `reactants_train.pkl.rlv2_norm.npz` | Pre-fit RLV2 normalisation stats |

Pickle schema: `{smiles: descriptor_array}` for reactants; `{id: template_dict}` for templates.

---

## Installation details

### Option A — conda (recommended)

```bash
conda env create -f env.yml
conda activate pgfs
pip install -e .
```

Update an existing environment:

```bash
conda env update -n pgfs -f env.yml --prune
```

### Option B — pip only

If you already have PyTorch, RDKit, and FAISS:

```bash
pip install -r requirements.txt
pip install -e .
```

| Package | Role |
|---------|------|
| PyTorch | TD3 actor / critic |
| RDKit | Reactions, QED, Morgan FP, RLV2 descriptors |
| FAISS | kNN over R(2) building-block keys |
| Gymnasium | Environment API |
| PyYAML | Config loading |
| Weights & Biases | Metrics + run tracking |


---

## Reference

**Learning to Navigate the Synthetically Accessible Chemical Space Using Reinforcement Learning**  
Gottipati et al., 2020 — [arXiv:2004.12485](https://arxiv.org/pdf/2004.12485) (PGFS)

---

## License

See LICENSE (if present) or the parent GenMolRL repository license.
