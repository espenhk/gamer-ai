# iRacing

[iRacing](https://www.iracing.com/) telemetry integration for the gamer-ai
reinforcement learning framework.  Uses
[pyirsdk](https://github.com/kutu/pyirsdk) to read live telemetry from the
iRacing simulator.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Python dependencies](#python-dependencies)
- [Running](#running)
  - [Telemetry-only mode (Phase 1)](#telemetry-only-mode-phase-1)
  - [Live action injection (Phase 2)](#live-action-injection-phase-2)
- [SimHub vs pyirsdk](#simhub-vs-pyirsdk)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Rewards](#rewards)
- [Example commands](#example-commands)
- [Supported policies](#supported-policies)

---

## Installation

### Prerequisites

- **Windows** — iRacing only runs on Windows.
- **iRacing subscription** — a valid iRacing account with the simulator
  installed.
- Python 3.11+, Poetry.

### Python dependencies

Install `pyirsdk`:

```bash
pip install pyirsdk
```

For **live action injection** (Phase 2), also install the vJoy driver
and the `pyvjoy` Python binding:

1. Download and install [vJoy](https://github.com/jshafer817/vJoy).
2. Install the Python binding:

```bash
pip install pyvjoy
```

Then install the rest of the project dependencies:

```bash
poetry install
```

---

## Running

### Telemetry-only mode (Phase 1)

In Phase 1, the agent **reads** telemetry (speed, position, lap
progress, tire temps, etc.) but does **not** inject actions into the
game.  You drive manually or replay a saved session while the agent
observes.

1. Launch iRacing and enter a practice session (or replay a recorded
   session).
2. Start the training loop:

```bash
python main.py smoke --game iracing --no-interrupt
```

The environment connects to the running iRacing instance via the
shared-memory telemetry API.

### Live action injection (Phase 2)

In Phase 2, the agent **sends steer/throttle/brake commands** to iRacing
via a [vJoy](https://github.com/jshafer817/vJoy) virtual joystick.

1. Install the vJoy driver and `pyvjoy` (see [Prerequisites](#python-dependencies)).
2. Configure iRacing to use the vJoy device as a controller
   (Options → Controls → assign axes for steer, throttle, brake).
3. Set `action_mode: live` in `training_params.yaml`:

```yaml
action_mode: live
```

4. Launch iRacing and enter a practice session.
5. Start the training loop:

```bash
python main.py my_run --game iracing --no-interrupt
```

The agent reads telemetry **and** injects actions each step.

> **Safety note:** The agent will actively control the car.  Start in a
> practice session on a safe track.  You can take back control at any
> time by pressing Escape in iRacing.

---

## SimHub vs pyirsdk

| | `pyirsdk` | SimHub |
|---|---|---|
| **What** | Pure Python SDK — reads iRacing shared memory directly | Universal telemetry hub (GUI app) with plugin ecosystem |
| **Latency** | Minimal (direct memory-mapped file read) | Slight overhead (SimHub processes data before forwarding) |
| **Programmability** | Full Python API, MIT-licensed | Plugin/script system; heavier setup |
| **Our choice** | ✅ Used in Phase 1 | Possible future integration for dashboard / overlay |

We use `pyirsdk` because it gives low-latency, programmatic access to
every telemetry variable without requiring a GUI intermediary.

---

## Configuration

| File | Purpose |
|---|---|
| `games/iracing/config/training_params.yaml` | Episode settings, policy type, hyperparams |
| `games/iracing/config/reward_config.yaml` | Reward weights |

### Key training params

| Parameter | Default | Description |
|---|---|---|
| `action_mode` | `telemetry_only` | `"telemetry_only"` (Phase 1 — read-only) or `"live"` (Phase 2 — vJoy injection) |
| `track` | `laguna_seca` | Reference track name |
| `in_game_episode_s` | `120.0` | Wall-clock seconds per episode |
| `policy_type` | `genetic` | Algorithm (see Supported policies below) |

---

## Observation space

21-dimensional float32 vector extracted from iRacing telemetry:

| Feature | Scale | Description |
|---|---|---|
| `speed_ms` | 100.0 | Vehicle speed (m/s) |
| `lateral_offset_m` | 5.0 | Metres from track centre |
| `track_progress` | 1.0 | Fraction of track completed [0, 1] |
| `yaw_error_rad` | π | Track heading minus car heading |
| `rpm` | 8000.0 | Engine RPM |
| `gear` | 6.0 | Current gear |
| `fuel_pct` | 1.0 | Fuel level [0, 1] |
| `throttle` | 1.0 | Throttle input [0, 1] |
| `brake` | 1.0 | Brake input [0, 1] |
| `steering` | 1.0 | Steering input [-1, 1] |
| `tire_load_fl`–`tire_load_rr` | 5000.0 | Per-wheel tyre load (N) |
| `tire_temp_fl`–`tire_temp_rr` | 150.0 | Per-wheel tyre surface temp (°C) |
| `brake_bias` | 1.0 | Brake bias front/rear |
| `lap_time_s` | 120.0 | Current lap elapsed time |
| `best_lap_time_s` | 120.0 | Session best lap time |

---

## Action space

Continuous: `Box([-1, 0, 0], [1, 1, 1], shape=(3,))`

| Output | Range | Effect |
|---|---|---|
| `steer` | [−1, 1] | Full left to full right |
| `accel` | [0, 1] | Throttle |
| `brake` | [0, 1] | Braking force |

Discrete policies use a 9-cell grid: {brake, coast, accel} × {left,
straight, right}.

In **telemetry-only** mode (`action_mode: telemetry_only`), actions are
computed by the policy but discarded.  In **live** mode
(`action_mode: live`), actions are injected into iRacing via vJoy.

---

## Rewards

Configured in `games/iracing/config/reward_config.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `progress_weight` | 10000.0 | Multiplied by the track-progress delta each step. Dominates the reward. |
| `centerline_weight` | −0.1 | Coefficient of the centerline penalty: `centerline_weight × |lateral_offset_m|^centerline_exp`. Negative — larger offsets cost more reward. |
| `centerline_exp` | 2.0 | Exponent for the centerline penalty. 2.0 = quadratic — small offsets are tolerated, large ones punished heavily. |
| `speed_weight` | 0.05 | Per-step bonus proportional to vehicle speed. Small tie-breaker. |
| `step_penalty` | −0.05 | Flat per-step time cost. |
| `finish_bonus` | 5000.0 | One-time reward at lap completion. |
| `off_track_penalty` | −10.0 | Applied each step when iRacing telemetry reports the car as off-track (`OnPitRoad` or `OffTrack` flag). Complements the centerline penalty for sharp track limits. |
| `lap_time_improvement_bonus` | 100.0 | One-time bonus awarded when the current lap time beats the session best (`best_lap_time_s` from telemetry). Encourages consistent lap-by-lap improvement. |
| `crash_threshold_m` | 25.0 | Episode terminates when `|lateral_offset_m|` exceeds this value. |

---

## Example commands

```bash
# Single experiment — telemetry-only (default)
python main.py my_iracing_run --game iracing --no-interrupt

# Live action injection (requires vJoy + pyvjoy)
# Set action_mode: live in training_params.yaml first, then:
python main.py my_iracing_run --game iracing --no-interrupt

# With a specific track override
python main.py my_run --game iracing --track laguna_seca
```

Results are saved to `experiments/iracing/<policy>/<track>/<name>/results/`.

---

## Supported policies

All framework policies work with iRacing.  Set `policy_type` in
`games/iracing/config/training_params.yaml`.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `hill_climbing` | Mutate-and-keep linear policy | Includes probe + cold-start phases |
| `neural_net` | MLP mutate-and-keep | Configure `hidden_sizes` |
| `epsilon_greedy` | Tabular Q-learning | Classical RL baseline |
| `mcts` | UCT-style Q-learning | Systematic exploration |
| `genetic` | Population evolutionary | Good for escaping local optima |
| `cmaes` | CMA-ES over flat weight vector | Best general-purpose for linear |
| `neural_dqn` | Deep Q-network | Gradient-based neural training |
| `reinforce` | Monte Carlo policy gradient | Stochastic policy |
| `lstm` | LSTM + isotropic ES | Temporal memory |
