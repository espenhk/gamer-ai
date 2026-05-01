# Assetto Corsa Integration

This guide covers training agents against [Assetto Corsa](https://store.steampowered.com/app/244210/Assetto_Corsa/) via the `games/assetto_corsa/` package.

## Install

The Assetto Corsa integration depends on a Gymnasium-compatible AC wrapper that registers the `AssettoCorsa-v0` environment. Install the optional Poetry group:

```bash
poetry install --with assetto_corsa
```

Two viable upstream wrappers are known at the time of writing:

- `assetto-corsa-rl` — referenced by issue #79; install with `pip install assetto-corsa-rl` if/when published to PyPI.
- [`dasGringuen/assetto_corsa_gym`](https://github.com/dasGringuen/assetto_corsa_gym) — install from source per the repo's own instructions.

The framework imports the wrapper lazily from inside `games.assetto_corsa.clients.ac_client`, so neither package needs to be installed for unit tests to run on Linux CI.

## Pointing at a local AC installation

The wrapper reads the AC install path from environment variables (the exact name depends on which wrapper you use; check its README). Typical values:

```bash
export ASSETTO_CORSA_INSTALL_DIR="C:/Program Files (x86)/Steam/steamapps/common/assettocorsa"
```

On Windows you can use `scripts/launch_assetto.ps1` to ensure the game is running before training starts. On Linux/macOS `scripts/launch_assetto.sh` is a no-op (the binary is Windows-only) but the gym wrapper itself still works.

## Quick start

```bash
# 1. Make sure AC is running (Windows only).
pwsh scripts/launch_assetto.ps1

# 2. Train with the default genetic policy.
python main.py --game assetto first_run

# Equivalent to:
python -m rl.main --game assetto first_run
```

The first run creates `experiments/assetto_corsa/<track>/first_run/` and copies in:

- `training_params.yaml` — hyperparams (policy type, mutation, episode length).
- `reward_config.yaml`   — reward weights.
- `policy_weights.yaml`  — saved at the end of each improving sim.
- `results/`             — analytics plots and `results.md`.

Edit the experiment-local copies to tune without affecting other experiments.

## Observation space

See `games/assetto_corsa/obs_spec.py`. The base observation is 16 features (speed, lateral offset, yaw error, pitch, roll, track progress, steering, RPM, gear, four wheel slips, three angular-velocity components). Vision features can be appended via `n_vision > 0` in `training_params.yaml`.

## Action space

```
Box([-1, 0, 0], [1, 1, 1], shape=(3,), dtype=float32)
    [0] steer  ∈ [-1, 1]
    [1] accel  ∈ [0, 1]   (thresholded at 0.5 → bool)
    [2] brake  ∈ [0, 1]   (thresholded at 0.5 → bool)
```

This is identical to TMNF's action space, so framework policies (genetic, CMA-ES, DQN, REINFORCE, …) work unchanged.
