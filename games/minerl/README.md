# MineRL (Minecraft)

Minecraft RL integration via [MineRL](https://minerl.readthedocs.io/) — a Gymnasium-compatible wrapper around Minecraft's Java server. Phase 1 uses a 3-dim vector observation (compass angle + inventory counts); pixel observations are deferred to Phase 2.

- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Rewards](#rewards)
- [Supported environments](#supported-environments)
- [Supported policies](#supported-policies)
- [Known limitations](#known-limitations)

---

## Installation

### Prerequisites

1. **Java 8** (OpenJDK 8 recommended — MineRL's server does not work with Java 9+):
   ```bash
   # Ubuntu / Debian
   sudo apt-get install openjdk-8-jdk
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   ```

2. **MineRL Python package:**
   ```bash
   pip install minerl
   ```

No Minecraft account is required. MineRL ships a bundled Minecraft client under Microsoft's research license.

### Headless Linux (CI / server)

Set the `MINERL_HEADLESS` environment variable so MineRL skips the display requirement:

```bash
export MINERL_HEADLESS=1
```

MineRL also requires a virtual display or the `Xvfb` server when `MINERL_HEADLESS` is not set:
```bash
sudo apt-get install xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

---

## Running

```bash
# Default map (MineRLNavigateDense-v0)
python main.py myrun --game minerl --no-interrupt

# Choose a specific MineRL environment
python main.py treechop --game minerl --track MineRLTreechop-v0 --no-interrupt
```

The first run creates `experiments/minerl/<policy>/<map>/<name>/` and copies both master configs in. Edit the experiment-local copies to tune without affecting other runs.

---

## Configuration

| File | Purpose |
|---|---|
| `games/minerl/config/training_params.yaml` | Episode length, policy type, mutation parameters |
| `games/minerl/config/reward_config.yaml` | Native reward scaling, step penalty, finish bonus |

Master configs are copied into `experiments/minerl/<policy>/<map>/<name>/` on first run.

---

## Observation space

Phase 1: compact 3-dim `float32` vector derived from the MineRL obs dict.

| Index | Feature | Scale | Description |
|---|---|---|---|
| 0 | `compass_angle` | 180.0 | Compass bearing to goal in degrees [-180, 180] |
| 1 | `inventory_dirt` | 64.0 | Dirt blocks in inventory [0, 64] |
| 2 | `inventory_log` | 64.0 | Log blocks in inventory [0, 64] |

Each value is divided by its scale before being passed to a policy, so normalised values are roughly in `[-1, 1]`.

The pixel observation (`pov`, 64×64×3) is available from the underlying MineRL env but is not included in Phase 1. A CNN-based Phase 2 is tracked in issue #215.

---

## Action space

`Discrete(9)` — integer index decoded to a MineRL action dict. Indices map to:

| Index | Label | MineRL keys set |
|---|---|---|
| 0 | `noop` | *(none — all zeros)* |
| 1 | `forward` | `forward=1` |
| 2 | `forward+jump` | `forward=1, jump=1` |
| 3 | `forward+attack` | `forward=1, attack=1` |
| 4 | `attack` | `attack=1` |
| 5 | `left+forward` | `left=1, forward=1` |
| 6 | `right+forward` | `right=1, forward=1` |
| 7 | `back` | `back=1` |
| 8 | `forward+sprint` | `forward=1, sprint=1` |

All keys not listed default to `0` (or `"none"` for craft/equip/place slots).

---

## Rewards

Reward configuration for MineRL. See `games/minerl/config/reward_config.yaml` for defaults.

| Parameter | Default | Description |
|---|---|---|
| `native_reward_scale` | `1.0` | Multiplier applied to the native MineRL reward signal each step |
| `step_penalty` | `-0.001` | Per-step time cost (encourages faster task completion) |
| `finish_bonus` | `100.0` | One-time bonus when the episode terminates with `terminated=True` (goal reached) |

**Total reward per step:**

```
reward = native_reward * native_reward_scale
       + step_penalty
       + (finish_bonus if terminated else 0)
```

**Tuning notes:**
- `native_reward_scale=0.0` with a large `finish_bonus` trains purely on sparse goal completion.
- Increase `native_reward_scale` (e.g. `2.0–5.0`) to amplify the dense reward signal from `MineRLNavigateDense-v0`.
- Keep `step_penalty` small (`-0.001` to `-0.01`); too large dominates the native signal.

---

## Supported environments

Phase 1 supports MineRL environments that expose the obs keys used by the vector spec. Others may work if they include `compassAngle` and/or `inventory`.

| Environment ID | Goal | Key obs features | Episode length |
|---|---|---|---|
| `MineRLNavigateDense-v0` | Navigate to a target diamond block | `compassAngle`, `inventory.dirt` | 2400 steps (120 s) |
| `MineRLTreechop-v0` | Chop trees and collect logs | `inventory.log` | 8000 steps (400 s) |

Set the environment with `map_name` in `training_params.yaml` or via `--track`:

```bash
python main.py treechop --game minerl --track MineRLTreechop-v0
```

---

## Supported policies

All framework policies that work with a `Discrete(9)` action space and a 3-dim continuous observation vector are compatible.

| Policy | Recommended? | Notes |
|---|---|---|
| `genetic` | ✓ default | Population evolutionary search; good for sparse rewards |
| `cmaes` | ✓ | CMA-ES over linear policy weights; auto-adapts step size |
| `hill_climbing` | ✓ | Fastest sanity check; no `policy_params` required |
| `neural_net` | ✓ | MLP with mutate-and-keep; good for non-linear features |
| `epsilon_greedy` | ✓ | Tabular Q-learning; practical for coarse binnings |
| `ucb_q` | ✓ | UCB1 online Q-learning; no ε schedule to tune |
| `neural_dqn` | ✓ | Deep Q-network with replay buffer; most sample-efficient |
| `reinforce` | ✓ | Monte Carlo policy gradient |
| `ppo` / `a2c` | ✓ | SB3 on-policy gradient (requires `poetry install --with deep_rl`) |
| `lstm` | ○ | Useful if partial observability matters (no inventory memory) |
| `alphazero_mcts` | ✗ | Gated off — MineRL binds to a JVM subprocess and is not cloneable |

---

## Known limitations

- **Phase 1 is vector obs only.** The `pov` pixel observation is not included; a CNN policy is needed for pixel-based training and is tracked in issue #215 (Phase 2).
- **Java 8 required.** MineRL's bundled Minecraft server does not run under Java 9+.
- **Slow startup.** The first `env.reset()` in an episode takes several seconds as the Java server launches; subsequent steps are much faster.
- **No Gymnasium render support.** `render_mode="human"` / `render_mode="rgb_array"` are not forwarded to the inner MineRL env in Phase 1.
