# <GAME_TITLE>

<!-- One-sentence description of what this integration does. -->

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Python dependencies](#python-dependencies)
- [Running](#running)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Reward](#reward)
- [Example commands](#example-commands)
  - [Single experiment](#single-experiment)
  - [Grid search](#grid-search)
- [Supported policies](#supported-policies)

---

## Installation

### Prerequisites

- Python 3.11+, Poetry
<!-- Add game-specific prerequisites (game binary, SDK, etc.) -->

### Python dependencies

```bash
# Add any game-specific pip/poetry install commands here.
poetry install
```

---

## Running

<!-- Describe how to launch the game and connect the agent. -->

---

## Configuration

| File | Purpose |
|---|---|
| `games/<name>/config/training_params.yaml` | Episode settings, policy type, hyperparams |
| `games/<name>/config/reward_config.yaml` | Reward weights |

---

## Observation space

<!-- Describe the observation features extracted by obs_spec.py. -->

| Feature | Scale | Description |
|---|---|---|
| `example_feature` | 1.0 | Replace with real features |

---

## Action space

<!-- Describe the action space (continuous and/or discrete). -->

---

## Reward

Configured in `games/<name>/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `step_penalty` | −0.1 | Per-step time cost |

---

## Example commands

### Single experiment

```bash
python main.py my_run --game <name>
```

### Grid search

```bash
python grid_search.py my_grid.yaml --game <name>
```

---

## Supported policies

All policies in the framework work with this game. Set `policy_type` in
`games/<name>/config/training_params.yaml`.
