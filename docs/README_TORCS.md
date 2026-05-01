# TORCS Game Integration

[TORCS (The Open Racing Car Simulator)](http://torcs.sourceforge.net/) is an open-source, cross-platform racing simulator. This module integrates TORCS into the *tmnf-ai* training framework as a drop-in alternative to the TMNF game module.

## Why TORCS?

- **Cross-platform** — runs on Linux, macOS, and Windows (TMNF is Windows-only)
- **Open-source** — GPL-licensed simulator with MIT-licensed Python wrapper
- **Gym-compatible** — `gym_torcs` provides a Gymnasium-style `step/reset` API
- **Rich sensors** — speed, position, lap progress, track-edge distances, wheel spin
- **Proven RL benchmark** — used in numerous academic papers

## Installation

### 1. Install TORCS

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install torcs
```

**macOS (Homebrew):**
```bash
brew install torcs
```

**Windows:**
Download the installer from [torcs.sourceforge.net](http://torcs.sourceforge.net/index.php?artid=3&catid=download).

### 2. Install the Python wrapper

```bash
pip install git+https://github.com/ugo-nama-kun/gym_torcs.git
```

### 3. Install the tmnf-ai TORCS dependencies

```bash
poetry install --with torcs
```

## Usage

### Single experiment

```bash
python main.py <experiment_name> --game torcs
```

### Configuration

Training parameters are in `games/torcs/config/training_params.yaml`.
Reward weights are in `games/torcs/config/reward_config.yaml`.

On first run, both are copied into your experiment directory for per-experiment tuning.

### Observation Space (19 features)

| Index | Name              | Scale    | Description                              |
|-------|-------------------|----------|------------------------------------------|
| 0     | speed_ms          | 50.0     | Vehicle speed in m/s                     |
| 1     | lateral_offset_m  | 5.0      | Metres from track centre                 |
| 2     | yaw_error_rad     | π        | Heading error vs track direction         |
| 3     | track_progress    | 1.0      | Fraction of lap completed [0, 1]         |
| 4     | rpm               | 10000.0  | Engine RPM                               |
| 5–8   | wheel_N_spin      | 200.0    | Wheel spin velocities (rad/s)            |
| 9–17  | track_edge_N      | 200.0    | Track-edge rangefinder distances          |
| 18    | track_position    | 1.0      | Normalised track position [-1, 1]        |

### Action Space

Same as TMNF: `Box([-1, 0, 0], [1, 1, 1], shape=(3,))` → `[steer, accel, brake]`.

## Policies

All existing policies (hill climbing, genetic, CMA-ES, neural net, etc.) work with TORCS out of the box — they operate on the flat observation vector and produce the same 3-dim action.

## Known Limitations

- **Memory leak on reset**: TORCS leaks memory when resetting without relaunching. The client uses `reset(relaunch=True)` by default to work around this.
- **Speed multiplier**: TORCS does not support the same speed multiplier as TMNF. Training runs at real-time speed (configurable via the TORCS server settings).
- **Vision mode**: Optional 64×64 pixel observations are supported but not appended to the observation vector by default. Set `vision=True` in the env constructor to enable.
