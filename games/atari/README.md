# Atari

Atari 2600 integration via [ale-py](https://github.com/Farama-Foundation/Arcade-Learning-Environment) + Gymnasium. No separate game binary is needed; ale-py ships the (MIT-licensed) ROMs as Python wheels.

- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Rewards](#rewards)
- [Example commands](#example-commands)
- [Supported policies](#supported-policies)
- [Licensing note](#licensing-note)

---

## Installation

### Python dependencies

Install ale-py (also pulls in the Atari ROM data):

```bash
pip install ale-py
# or, via the optional Poetry group:
poetry install --with atari
```

`ale-py` registers the `ALE/<Game>-v5` envs with Gymnasium on import; no extra `register_envs` call is needed at the call site.

---

## Running

No external process required. Gymnasium / ALE manages the emulator lifecycle in-process.

```bash
python main.py smoke --game atari --no-interrupt
```

By default the integration runs `Pong-v5`. Override with the `map_name` config key or the CLI `--track` flag:

```bash
python main.py breakout_run --game atari --track Breakout-v5 --no-interrupt
```

The first run creates `experiments/atari/<policy>/<map>/<name>/` and copies both master configs in. Edit the experiment-local copies to tune without affecting other runs.

---

## Configuration

| File | Purpose |
|---|---|
| `games/atari/config/training_params.yaml` | Episode settings, policy type, hyperparams |
| `games/atari/config/reward_config.yaml` | Reward shaping |

Key training params:

| Key | Default | Notes |
|---|---|---|
| `map_name` | `Pong-v5` | Any ALE-registered game id; accepts either `"Pong-v5"` or `"ALE/Pong-v5"`. |
| `in_game_episode_s` | `60.0` | Wall-clock seconds (~`* 60` env steps at 60 fps) before truncation. |
| `n_sims` | `100` | Greedy iterations / generations. |
| `policy_type` | `genetic` | Any framework policy (see table below). |

---

## Observation space

128-byte console RAM, exposed by ALE in `obs_type="ram"` mode. Each byte is presented as a separate float32 feature (`ram_000`–`ram_127`), scaled by 255.0 so normalised values land in `[0, 1]`.

| Feature | Scale | Description |
|---|---|---|
| `ram_000`–`ram_127` | 255.0 | Atari 2600 console RAM byte (0–255) |

Using RAM rather than raw pixels keeps the observation a fixed-size flat vector, which means every framework flat-observation policy works out of the box.

Pixel-based observations and CNN policies are deferred to a follow-up (see issue #217's "factor `SC2CNNPolicy` into a game-agnostic `framework/cnn_policy.py`" pre-req).

---

## Action space

The integration accepts a 1-D action vector and maps it to the underlying ALE `Discrete(N)` space (N varies per game; 18 is the maximum):

* Continuous policies (`hill_climbing`, `neural_net`, `genetic`, `cmaes`, `lstm`, …) emit `action[0] ∈ [-1, 1]`, which is linearly mapped to `[0, N − 1]`.
* Tabular policies (`epsilon_greedy`, `mcts`) use `DISCRETE_ACTIONS = [[0], [1], …, [17]]`. Out-of-range indices for games with smaller legal sets (e.g. `Pong-v5` has 6 actions) clamp to `NOOP` (0).

---

## Rewards

Configured in `games/atari/config/reward_config.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `native_reward_scale` | 1.0 | Multiplier applied to the raw per-step ALE score delta. The native signal is game-specific — in Pong it is +1 for a point scored and −1 for a point conceded; in Breakout it is the brick-break score. |
| `clip_sign` | `false` | When `true`, clips the per-step reward to {−1, 0, +1} before applying `native_reward_scale`. This is the DQN-paper convention for stabilising training across games with wildly different score magnitudes. |
| `step_penalty` | 0.0 | Flat per-step cost added to the (optionally clipped, scaled) native reward. Disabled by default — the native signal already provides temporal pressure in most Atari games. |

---

## Example commands

### Single experiment

```bash
python main.py pong_genetic --game atari --track Pong-v5
```

Results saved to `experiments/atari/genetic/Pong-v5/pong_genetic/results/`.

### Grid search

A ready-made DQN template is included:

```bash
python grid_search.py games/atari/config/gs_neural_dqn_template.yaml --game atari
```

The template sweeps `learning_rate × epsilon_decay_steps` (6 combos) on
`Pong-v5` using the DQN-paper conventions (reward clipping, double Q-targets,
Huber loss).  See the file header for tuning notes and second-pass ablation
suggestions.

---

## Supported policies

All flat-observation framework policies work on Atari. Set `policy_type` in `games/atari/config/training_params.yaml`.

| `policy_type` | Algorithm |
|---|---|
| `hill_climbing` | Mutate-and-keep linear policy |
| `neural_net` | MLP mutate-and-keep |
| `genetic` | Population of WeightedLinearPolicy, evolutionary |
| `cmaes` | (μ/μ_w, λ)-CMA-ES over flat weight vector |
| `epsilon_greedy` | Tabular Q-learning |
| `mcts` | UCT-style Q-learning |
| `neural_dqn` | Deep Q-network |
| `reinforce` | Monte Carlo policy gradient |
| `lstm` | LSTM + isotropic ES |
| `ppo` | On-policy actor-critic |

---

## Licensing note

`ale-py` itself is Apache 2.0. The Atari 2600 ROMs it bundles were [released under the MIT license in 2022](https://github.com/Farama-Foundation/Arcade-Learning-Environment#license), so no separate ROM download is required.

stable-retro (NES / SNES / Genesis / N64 ROMs) is **not** wired up by this integration — those ROMs are still copyright-encumbered and would require the user to legally own each title.
