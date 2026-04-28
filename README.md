# TMNF — Trackmania Nations Forever RL

A reinforcement-learning agent that drives Trackmania Nations Forever autonomously. It offers a menu of training algorithms — hill-climbing, neural-net mutate-and-keep, tabular Q-learning, MCTS, a genetic algorithm, and CMA-ES — all trained live against TMInterface against the reference track A03. See [`CLAUDE.md`](CLAUDE.md) for the full architecture documentation.

--- [TMNF — Trackmania Nations Forever RL](#tmnf--trackmania-nations-forever-rl)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Automated (recommended)](#automated-recommended)
  - [Manual](#manual)
- [Quick start](#quick-start)
- [Configuring a run](#configuring-a-run)
- [Policy types](#policy-types)
  - [Policy-specific params](#policy-specific-params)
  - [Training phases (hill\_climbing only)](#training-phases-hill_climbing-only)
  - [Episode warmup](#episode-warmup)
- [LIDAR](#lidar)
- [Grid search](#grid-search)
- [Distributed training](#distributed-training)
  - [Coordinator](#coordinator)
  - [Worker](#worker)
  - [Auth model](#auth-model)
  - [Heartbeats and re-queue](#heartbeats-and-re-queue)
  - [Status endpoint](#status-endpoint)
  - [Same-machine smoke test](#same-machine-smoke-test)
- [Azure worker VMs](#azure-worker-vms)
  - [Topology](#topology)
  - [One-time Azure prerequisites](#one-time-azure-prerequisites)
  - [Terraform deployment order](#terraform-deployment-order)
  - [Per-VM setup](#per-vm-setup)
  - [Running a distributed sweep on Azure](#running-a-distributed-sweep-on-azure)
  - [Cost controls](#cost-controls)
  - [Teardown](#teardown)
  - [Follow-up](#follow-up)
- [Reward \& parameter tuning reference](#reward--parameter-tuning-reference)
  - [When a reward param needs changing](#when-a-reward-param-needs-changing)
  - [When a training param needs changing](#when-a-training-param-needs-changing)
  - [Episode length vs finish signals](#episode-length-vs-finish-signals)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

The training loop drives the live TMNF game via TMInterface, so this project only runs on Windows.

- **Windows 10/11.** `pywin32`, the `mss` window-capture backend, and the `tminterface` Python bindings all attach to the running game process.
- **Trackmania Nations Forever** installed (free from Ubisoft / Nadeo).
- **TMInterface 1.4.x.** Later versions changed the Python API and are **not** compatible — see the note in `setup_and_run.ps1`. Official installer: https://donadigo.com/tminterface
- **Python 3.11+** and **Poetry**.

---

## Installation

### Automated (recommended)

The repo ships with a PowerShell bootstrap script that installs Python, Git, Poetry, and TMInterface 1.4, runs `poetry install --with tmnf`, launches TMInterface, and invokes the command you pass it:

```powershell
.\setup_and_run.ps1 "python main.py my_experiment"
```

The script is idempotent — re-running it on an already-configured machine simply skips the install steps and goes straight to launching TMInterface + your command. Pass `-DryRun` to preview without executing.

### Manual

If you'd rather install the prerequisites yourself:

1. Install Windows, TMNF, TMInterface 1.4.x, Python 3.11+, and Poetry (see Prerequisites above).
2. From the repo root, install the Python dependencies:
   ```bash
   poetry install --with tmnf
   ```
3. Verify the install by running the unit tests:
   ```bash
   poetry run python -m pytest tests/
   ```
4. Launch TMInterface manually, then run the training commands below.

---

## Quick start

```bash
# Single experiment
python main.py <experiment_name> [--no-interrupt] [--re-initialize]

# Grid search over multiple param combinations
python grid_search.py config/my_grid.yaml [--no-interrupt]
```

Results land in `experiments/<track>/<name>/results/`.

`--re-initialize` ignores any existing weights file and reruns probe + cold-start from scratch.

---

## Configuring a run

On first run, `config/training_params.yaml` is copied into `experiments/<track>/<name>/training_params.yaml`. Edit the experiment copy to tune without affecting other experiments.

```yaml
track: a03_centerline
speed: 10.0
in_game_episode_s: 30.0
n_sims: 100
mutation_scale: 0.05
mutation_share: 1.0      # fraction of weights perturbed per mutation
probe_s: 15.0
cold_restarts: 20
cold_sims: 5
n_lidar_rays: 8          # 0 = disabled

policy_type: genetic     # see Policy types below

policy_params:
  # type-specific hyperparams (see below)
```

---

## Policy types

Set `policy_type` in `training_params.yaml`. Each type uses the same `n_sims` budget but runs a different training loop.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `hill_climbing` | Mutate-and-keep `WeightedLinearPolicy` | Default. Includes probe + cold-start phases. |
| `neural_net` | Mutate-and-keep `NeuralNetPolicy` (MLP) | Pure numpy, no framework needed. Configure `hidden_sizes`. |
| `epsilon_greedy` | Tabular Q-learning, ε-greedy exploration | ε decays per episode. Q-table is in-memory only. |
| `mcts` | UCT-style online Q-learner (UCB1) | Approximation — no env cloning, builds value table over real episodes. |
| `genetic` | Population of `WeightedLinearPolicy`, evolutionary | `n_sims` = number of generations; total episodes = `n_sims × population_size`. |
| `cmaes` | CMA-ES over flat `WeightedLinearPolicy` weights (Hansen 2016) | Adapts full covariance matrix; automatic step-size control via CSA. |

### Policy-specific params

```yaml
# neural_net
policy_params:
  hidden_sizes: [16, 16]

# epsilon_greedy
policy_params:
  n_bins: 3
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.05
  alpha: 0.1
  gamma: 0.99

# mcts
policy_params:
  n_bins: 3
  c: 1.41        # UCB1 exploration constant
  alpha: 0.1
  gamma: 0.99

# genetic
policy_params:
  population_size: 20
  elite_k: 3
  # mutation_scale inherited from top-level

# cmaes
policy_params:
  population_size: 20   # λ — offspring per generation
  initial_sigma: 0.3    # starting step size (adapts automatically via CSA)
```

#### CMA-ES details

`CMAESPolicy` implements `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over the concatenated `[steer | accel | brake]` weight vector of a `WeightedLinearPolicy` (~63 dimensions for the base observation space).

Each generation:
1. **Sample** — draw λ offspring from `N(mean, σ²·C)` using the cached eigen-factorization `C = B D² Bᵀ`
2. **Evaluate** — run one episode per offspring
3. **Update** — weighted mean recombination, cumulative step-size adaptation (CSA) for σ, and rank-1 + rank-μ covariance update

The champion (best individual seen across all generations) is saved in standard `WeightedLinearPolicy` YAML format so analytics, heatmaps, and inference work without modification. Step-size σ is reported per generation in logs and adapts automatically — no `mutation_scale` tuning needed.

`n_sims` controls the number of generations; total episodes = `n_sims × population_size`.

### Training phases (hill_climbing only)

1. **Probe** — runs 6 fixed-action episodes (brake/accel × left/straight/right; coast skipped) for `probe_s` seconds each to establish a reward floor.
2. **Cold-start search** — up to `cold_restarts` rounds of random-init hill-climbing, `cold_sims` sims each. Stops early if the floor is beaten.
3. **Greedy** — `n_sims` iterations of mutate-and-keep.

All other policy types skip probe and cold-start and go straight to greedy.

### Episode warmup

The first 100 steps of every episode force full-throttle straight (`accel + straight, no brake`) regardless of the policy. This covers the braking-start phase so the policy's Q-table / weights are not poisoned by forced behaviour.

---

## LIDAR

Set `n_lidar_rays > 0` to append wall-distance observations from a screenshot-based LIDAR sensor. The `LidarSensor` class:

- Captures the game window via MSS
- Converts to a 128×32 binary edge image (grayscale → threshold → Canny → dilate → blur)
- Raycasts `n_lidar_rays` evenly spaced angles from 0 to π, returning normalised distances in ~[0, 1]

LIDAR rays are appended to the observation vector. All policies handle variable-length observations automatically; `WeightedLinearPolicy` auto-migrates existing weights files to add new LIDAR keys (initialised to 0.0).

Requires: `mss`, `opencv-python`, `pywin32`.

---

## Grid search

Copy `config/grid_search_template.yaml`, set any param to a list to sweep it:

```yaml
base_name: "gs_v1"
training_params:
  mutation_scale: [0.05, 0.1, 0.2]   # 3-way sweep
  n_sims: 50
  ...
reward_params:
  centerline_weight: [-0.1, -0.5]    # 2-way sweep
  ...
```

This creates one experiment per unique combination (3 × 2 = 6 here). Experiment names encode only the varied params: `gs_v1__ms0.05__cw_n0.1`, etc.

---

## Distributed training

Larger grid searches can saturate a single Windows box long before the sweep completes. When that happens, `grid_search.py --distribute` turns one machine into an HTTP coordinator that hands work items to any number of worker machines. Distribution is at the **experiment level** — each worker runs one full `train_rl()` at a time — so every worker still needs its own live TMInterface game session.

### Coordinator

Start the coordinator on the machine that owns the grid config:

```bash
python grid_search.py config/my_grid.yaml --distribute
```

This serves work items over HTTP on port `5555` and blocks until every combination has a result. Override the port with `--port 6000` on the CLI, or set `distribute.port` in the grid config YAML. The coordinator writes each returned run into the local `experiments/<track>/<name>/` tree and runs analytics exactly as a local grid search would.

### Worker

On each worker machine (one per TMInterface session), point at the coordinator:

```bash
python -m distributed.worker --coordinator http://<coordinator-host>:5555 --token <secret>
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--coordinator URL` | *required* | Coordinator base URL |
| `--token SECRET` | env `TMNF_GRID_TOKEN` | Shared secret; see auth below |
| `--worker-id ID` | hostname | Identifier shown in coordinator logs |
| `--heartbeat-interval S` | `15` | Seconds between heartbeat POSTs |
| `--no-interrupt` | off | Skip all "Press Enter" prompts |
| `--re-initialize` | off | Ignore existing weights files |
| `--log-level LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

The worker polls `GET /work`, runs the returned combo locally, then posts the resulting `ExperimentData` to `POST /result`. It exits once the coordinator reports every item complete.

### Auth model

Every request carries `Authorization: Bearer <token>`; a mismatch returns `401 Unauthorized`. Token precedence:

1. `--token` on the CLI
2. `TMNF_GRID_TOKEN` environment variable
3. Auto-generated UUID (coordinator only — logged on startup so you can copy it to workers)

Workers have no auto-generated fallback: they must receive the token via `--token` or `TMNF_GRID_TOKEN`.

### Heartbeats and re-queue

While a worker is running an experiment, it POSTs `/heartbeat` every `--heartbeat-interval` seconds. The coordinator re-queues any item whose worker has been silent for `--heartbeat-timeout` seconds (default `60`, overridable via `distribute.heartbeat_timeout` in the config). A crashed or disconnected worker therefore loses at most one partial run before the combo is retried on another machine.

### Status endpoint

`GET /status` returns a JSON summary with queue depth, completed count, and active workers — handy for monitoring from `curl` or a dashboard without tailing logs.

### Same-machine smoke test

To verify the pipeline end-to-end on a single box, open two shells:

```bash
# shell 1 — coordinator
python grid_search.py config/my_grid.yaml --distribute --token hunter2

# shell 2 — worker (needs TMInterface running)
python -m distributed.worker --coordinator http://localhost:5555 --token hunter2
```

When the worker finishes every combo, both processes exit on their own.

---

## Azure worker VMs

The `infrastructure/` directory ships a three-stage Terraform stack (`auth/`, `remote_state/`, `environment/`) that provisions a full distributed-training fleet on Azure: one coordinator VM plus N worker VMs, a Key Vault for admin passwords, and an NSG locked down to your public IP. Trackmania Nations Forever itself still has to be installed by hand on each VM — everything else is automated.

### Topology

| Component | Default SKU | Notes |
|---|---|---|
| Coordinator VM | `Standard_B1ms` | One per fleet, runs `grid_search.py --distribute` |
| Worker VMs | `Standard_D2as_v5` | `worker_vm_count` copies, run `distributed.worker` |
| OS | Windows 11 Pro 24H2 | Matches the TMInterface requirement |
| Resource group | `rg-<project_name>` | Default `rg-tmnf-ai` (from `project_name = "tmnf-ai"` in `terraform.tfvars.example`) |
| Key Vault | `kv-<project_name>-<random>` | Stores generated admin passwords for every VM |
| NSG | `nsg-<project_name>` | Inbound RDP (3389) from `my_ip_address` only |

### One-time Azure prerequisites

1. An Azure subscription with sufficient vCPU quota in the target region (`centralindia` by default — the coordinator + 2 default workers need ~5 vCPUs).
2. The Azure CLI installed and logged in: `az login`.
3. Your Azure AD user object ID, needed so Terraform can grant you access to the Key Vault:
   ```bash
   az ad signed-in-user show --query id -o tsv
   ```
4. Your current public IP for the RDP NSG rule:
   ```bash
   curl ifconfig.me
   ```
5. Terraform ≥ 1.5 on the machine running the deploy.

### Terraform deployment order

The three stages must be applied in order — `auth/` creates the service principal with federated OIDC, `remote_state/` creates the storage backend, `environment/` creates the actual VMs and reads its state from that backend.

```bash
# 1. Service principal + federated credentials for GitHub Actions
cd infrastructure/auth
terraform init
terraform apply

# 2. Remote-state storage account + container
cd ../remote_state
terraform init
terraform apply

# 3. Wire the environment stack to the remote state
cd ../environment
cp backend.conf.example backend.conf        # fill in outputs from step 2
cp terraform.tfvars.example terraform.tfvars # set my_object_id, my_ip_address, worker_vm_count

# 4. Deploy the VMs
terraform init -backend-config=backend.conf
terraform apply
```

The public IPs and admin-user for every VM are printed in the `vm_details` output:

```bash
terraform output vm_details
```

Required variables in `terraform.tfvars` (see `infrastructure/environment/variables.tf` for all of them):

| Variable | Purpose |
|---|---|
| `my_object_id` | Your AAD user object ID — gets Key Vault read access |
| `my_ip_address` | Your public IP — allowed through the NSG on 3389 |
| `worker_vm_count` | Number of worker VMs (main cost lever) |
| `worker_vm_size` | SKU for worker VMs (default `Standard_D2as_v5`) |
| `coordinator_vm_size` | SKU for the coordinator (default `Standard_B1ms`) |
| `admin_username` | Local admin login name (default `adminuser`) |

### Per-VM setup

Fetch the generated admin password from Key Vault — the secret name is `<project_name>-worker-<index>-password` (or `<project_name>-coordinator-password`):

```bash
az keyvault secret show \
  --vault-name <kv-name> \
  --name tmnf-ai-worker-0-password \
  --query value -o tsv
```

RDP into the VM with `adminuser` and that password, then clone the repo and run the bootstrap:

```powershell
git clone https://github.com/espenhk/tmnf-ai.git
cd tmnf-ai
.\setup_and_run.ps1 ""
```

`setup_and_run.ps1` installs Python 3.11, Git, Poetry, and TMInterface 1.4, and runs `poetry install --with tmnf`. Trackmania Nations Forever itself still needs to be installed manually on each worker (Ubisoft / Nadeo download); the coordinator VM doesn't need TMNF since it only schedules work.

### Running a distributed sweep on Azure

```bash
# 1. Start every VM in the resource group
az vm start --ids $(az vm list -g rg-tmnf-ai --query "[].id" -o tsv)
```

Then on the **coordinator** VM (RDP in first):

```powershell
python grid_search.py config/my_grid.yaml --distribute
```

Copy the coordinator URL and token that it logs. On each **worker** VM:

```powershell
python -m distributed.worker --coordinator http://<coordinator-ip>:5555 --token <secret>
```

When the sweep finishes, deallocate every VM to stop billing (stopped-but-allocated VMs still cost money — `deallocate` is the right verb):

```bash
az vm deallocate --ids $(az vm list -g rg-tmnf-ai --query "[].id" -o tsv)
```

> **Operational caveat** — the Terraform NSG only opens inbound **3389** (RDP). Port **5555** (coordinator HTTP) is not exposed to the public internet. Workers reach the coordinator over the shared VNet via its private IP, or you can add an NSG rule / use RDP port-forwarding while testing. Never expose the coordinator port without adding authentication-aware firewall rules.

### Cost controls

- `worker_vm_count` is the main cost lever — scale it down before committing the grid-search run and back up only for the sweep itself.
- Always `az vm deallocate` when the fleet is idle. Stopped-but-allocated VMs keep accruing compute charges.
- `worker_vm_size` and `coordinator_vm_size` are configurable — drop to a smaller SKU for smoke tests.

### Teardown

Tear down the compute when you're done for the day:

```bash
cd infrastructure/environment
terraform destroy
```

This removes the resource group, VMs, NSG, Key Vault, and NICs but leaves the remote state intact so the next `apply` is incremental. Only destroy `infrastructure/remote_state/` and `infrastructure/auth/` when you're permanently abandoning the project — those stages hold the Terraform state backend and GitHub Actions OIDC credentials.

### Follow-up

A cloud-init / custom-script extension that fully automates per-VM setup (including the TMNF install itself) is a natural next step but is not part of the current Terraform. Contributions welcome — open an issue before picking it up.

---

## Reward & parameter tuning reference

### When a reward param needs changing

| Param | Symptom |
|---|---|
| `progress_weight` | All runs score similarly (low reward variance) → scale too high. Car ignores all other signals entirely → reduce. |
| `centerline_weight` | Car wanders 10+ m but still improves → too weak. Car hugs centerline but drives slowly / avoids steering → too strong. |
| `centerline_exp` | Car oscillates wide/narrow alternately → increase to 3–4 to punish large drifts more sharply without tightening near-center tolerance. |
| `accel_bonus` | >90% accel steps but car still crashes or never steers → too high; reduce. Coasting dominates after cold-start → too low; increase. |
| `step_penalty` | Car times out without finishing when it clearly could → too small; increase to −0.1 or more. |
| `finish_bonus` | Car never reaches the finish despite good driving → too small; increase to 5000–10000. Also check `in_game_episode_s` is long enough to allow finishing. |
| `finish_time_weight` | Car finishes but brakes before the line → weight too small; increase magnitude. |
| `airborne_penalty` | Car bounces/falls frequently but penalty has no effect → fires rarely or too small vs other signals; increase. |

### When a training param needs changing

| Param | Symptom |
|---|---|
| `mutation_scale` | Hill-climbing never improves (all candidates regress) → too large, mutations destroy good weights. Improvement rate high early then stalls → too small to escape local minima; increase. |
| `n_sims` | Best reward plateaus in first 30% of sims → wasted budget; reduce and run more experiments. Large variance in final reward across independent runs → increase. |
| `in_game_episode_s` | Most episodes truncate (not crash/finish) → too short; car never reaches the finish. Most episodes crash very early → may be too long; reduce or tighten `crash_threshold_m`. |
| `crash_threshold_m` | Too many mid-episode crashes introducing noise → lower threshold to terminate faster. Too many truncated episodes when car is actually driving okay → threshold may be fine; check `in_game_episode_s`. |
| `cold_restarts` / `cold_sims` | Cold-start never beats probe floor → increase restarts or sims per restart. Cold-start always wins on restart 1 → too many restarts; reduce. |
| `speed` | Car shows oscillation or instability at 10× but not at 5× → reduce game speed. |
| `probe_s` | Probe rewards are all near-zero (car barely moves) → too short; increase so each action produces a meaningful episode. |

### Episode length vs finish signals

`in_game_episode_s = 13.0` with `par_time_s = 60.0` means the finish line is unreachable in a normal episode. `finish_bonus` and `finish_time_weight` have no effect until `in_game_episode_s` is increased to cover the full track. This is intentional during early training (focus on the track start), but should be revisited once the policy is competent for the first section.

---

## Repository layout

```
tmnf-ai/
├── main.py                 # Entry point — python main.py <experiment_name>
├── grid_search.py          # Local + distributed grid search driver
├── setup_and_run.ps1       # Windows bootstrap (Python/Poetry/TMInterface + run)
├── pyproject.toml          # Poetry dependencies (tmnf / tmnf-test groups)
├── framework/              # Game-agnostic RL primitives (base env, policies, training loop, analytics)
├── games/
│   └── tmnf/               # Trackmania-specific code — the real source of truth
│       ├── env.py          # TMNFEnv Gymnasium environment
│       ├── clients/        # TMInterface bridge (base, RL, instruction)
│       ├── lidar.py        # Screenshot-based wall-distance sensor
│       ├── policies.py     # TMNF policy implementations
│       ├── reward.py       # RewardCalculator + RewardConfig
│       ├── track.py        # Centerline loader / projection
│       └── tools/          # CLI helpers (build_centerline, debug_straight, …)
├── distributed/            # HTTP coordinator + worker for multi-box sweeps
├── infrastructure/         # Terraform stack for Azure worker VMs — see infrastructure/README.md
├── clients/                # Backward-compat shim → re-exports games.tmnf.clients.*
├── rl/                     # Backward-compat shim → re-exports games.tmnf.env
├── config/                 # Master training / reward / grid-search config templates
├── tracks/                 # Centerline `.npy` files
├── replays/                # `.Replay.Gbx` inputs for centerline builds
├── tests/                  # Pytest suite (policies, env, reward, distributed, …)
├── experiments/            # Per-experiment results (git-ignored)
├── runs/                   # Saved run artefacts
├── plans/                  # Design notes
└── .github/                # CI workflows + agent configs
```

> **Heads-up on shims** — the top-level `clients/` and `rl/` directories are thin backward-compat re-exports for code that moved under `games/tmnf/`. Treat `games/tmnf/` as the source of truth when reading, debugging, or editing — the shims only exist so older import paths keep working.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| **"Game window not found" / LIDAR returns all zeros** | Trackmania window isn't focused or its title doesn't start with `TmForever` (see `games/tmnf/lidar.py`). Bring the game window to the foreground and retry. |
| **`401 Unauthorized` in worker logs** | Token mismatch between coordinator and worker. Pass the same value via `--token` on both ends, or set `TMNF_GRID_TOKEN` in the environment. |
| **Worker never picks up work** | Coordinator is unreachable. Double-check the `--coordinator` URL, any host firewall, and — on Azure — that the coordinator port (5555 by default) is reachable. The current Terraform NSG does **not** open 5555; use the VNet private IP, add an NSG rule, or RDP port-forward while testing. |
| **`poetry install` fails on Linux** | Expected. The `tmnf` group depends on `pywin32`, `mss`, and `tminterface`, none of which work off Windows. On non-Windows CI, install `--with tmnf-test` instead — it pulls only the cross-platform deps the test suite needs. |
| **No progress after cold-start** | The reward config is likely off. Start with the [Reward & parameter tuning reference](#reward--parameter-tuning-reference) tables — `progress_weight`, `centerline_weight`, and `accel_bonus` are the usual culprits. |
| **Weights file schema mismatch on load** | `WeightedLinearPolicy` auto-migrates new observation keys, but if load still fails, rerun with `--re-initialize` to discard the stale weights and re-run probe + cold-start. |
