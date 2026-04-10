# CLAUDE.md

This repo contains two independent hobby projects sharing a single Python virtual environment managed by Poetry.

---

## Repository Structure

```
espenhk-hobby-projects/
├── .venv/                  # Shared virtual environment (created by Poetry)
├── skate/                  # Ice skating race predictor
├── tmnf/                   # Trackmania Nations Forever AI
├── pyproject.toml          # Shared dependencies for both projects
├── poetry.lock             # Locked dependency versions
├── README.md
└── CLAUDE.md
```

---

## Project: `skate`

### Purpose
Terminal app for live ice skating race tracking. Operator inputs lap splits as they happen; the app predicts finish times and tracks inter-skater gaps.

### Structure
```
skate/
├── race_predictor.py       # Entry point — run this
├── start.py                # Shortcut launcher
├── demo.py                 # Demo with fake data
├── models/
│   ├── skater.py           # Skater state + speed-based prediction
│   ├── race.py             # Race state management
│   ├── competition.py      # Competition + leaderboard
│   ├── person.py           # Skater profile entity
│   └── race_preset.py      # Distance configs (1500m, 3000m, 5000m, 10000m)
├── engine/
│   └── predictor.py        # Algorithms: simple / weighted / fatigue-adjusted
├── ui/
│   ├── cli.py              # Interactive CLI
│   └── base_ui.py          # Base display components
├── presets/                # JSON race distance configs
├── data/
│   ├── competitions/       # Competition fixture JSON files
│   └── people/             # Individual skater profiles (JSON)
├── scripts/
│   ├── parse_pdf.py        # Extract skater lists from PDF start lists
│   ├── manage_persons.py   # CRUD for skater database
│   └── populate_people_from_competition.py
└── tests/
    └── test_race_predictor.py  # Unit tests
```

### State
Complete and functional. All core features work. Skater profiles include historical PB/SB data. PDF parsing script exists for loading real competition start lists.

### Key design decisions
- Predictions use average speed (m/s) rather than raw lap times, which handles variable-distance first laps correctly.
- Time input is flexible: `MM:SS.mmm`, `SS.mmm`, or bare `SS`.
- Data is JSON-based, no database needed.

### Running tests
```bash
python -m pytest skate/tests/
```

---

## Project: `tmnf`

### Purpose
Drive in Trackmania Nations Forever autonomously — first with a hand-coded PD controller, then with a trained linear policy using hill-climbing.

### Structure
```
tmnf/
├── main.py                 # Entry point — run with: python main.py <experiment_name>
├── policies.py             # SimplePolicy (PD baseline) + WeightedLinearPolicy (trainable)
├── utils.py                # StateData, Vec3, Quat, WheelState data classes
├── track.py                # Centerline class: load .npy, project position
├── build_centerline.py     # Script to build centerline from a replay
├── instructions.py         # Predefined input instruction sequences
├── clients/
│   ├── phase.py            # Phase enum (BRAKING_START, RUNNING)
│   ├── instruction_client.py  # Replays a fixed instruction sequence
│   ├── adaptive_client.py  # PD controller following centerline (works on A03)
│   └── rl_client.py        # Thread-safe bridge for RL training
├── rl/
│   ├── env.py              # TMNFEnv — Gymnasium Env wrapping the game
│   ├── reward.py           # RewardCalculator + RewardConfig
│   ├── reward_config.yaml  # Master reward weights — copied into each new experiment
│   ├── train.py            # PPO training script (not primary path)
│   └── __init__.py
├── experiments/            # Per-experiment weights and reward configs (git-ignored)
│   └── <name>/
│       ├── policy_weights.yaml   # Saved best weights for this experiment
│       └── reward_config.yaml    # Reward config copy for this experiment
├── replays/
│   └── a03_centerline.Replay.Gbx
└── tracks/
    └── a03_centerline.npy
```

### Running
```bash
# From tmnf/
python main.py <experiment_name>
```

On first run with a new name, `experiments/<name>/` is created and the master `rl/reward_config.yaml` is copied in. Edit the experiment's copy to tune rewards without affecting other experiments. If no `policy_weights.yaml` exists, the cold-start phase runs automatically.

### Training phases

**1. Probe phase** (cold-start only — no weights file present)
Runs each of the 9 discrete actions as a constant policy for a short episode. Establishes a reward floor as a baseline for the search phase.

**2. Cold-start search**
Up to `COLD_RESTARTS` rounds of randomly-initialised hill-climbing, each running `COLD_SIMS` simulations. Stops early if any restart beats the probe floor. The best policy found across all restarts is saved and used as the starting point for the greedy phase.

**3. Greedy optimisation phase**
`N_SIMS` iterations of hill-climbing: mutate the current best weights, simulate, keep if improved. Best weights are saved to `experiments/<name>/policy_weights.yaml` after each improvement.

Subsequent runs (weights file already present) skip phases 1 and 2 and go straight to greedy.

### Configuring a run
All tunable parameters are at the top of `main()`:

| Variable | Default | Description |
|---|---|---|
| `SPEED` | 10.0 | Game speed multiplier (TMInterface max is 10×) |
| `IN_GAME_EPISODE_S` | 13.0 | In-game seconds per episode (braking + driving) |
| `N_SIMS` | 100 | Greedy phase simulations |
| `MUTATION_SCALE` | 0.1 | Std-dev of Gaussian noise per mutation (applied to normalised weights) |
| `PROBE_S` | 8.0 | In-game seconds for each probe run |
| `COLD_RESTARTS` | 5 | Max random restarts in cold-start search |
| `COLD_SIMS` | 10 | Hill-climb sims per cold-start restart |

### RL environment details

**Observation (15 floats):**
| Index | Name | Description |
|-------|------|-------------|
| 0 | speed_ms | Speed in m/s |
| 1 | lateral_offset_m | Metres from centreline (positive = right) |
| 2 | vertical_offset_m | Height above/below centreline |
| 3 | yaw_error_rad | track_yaw − car_yaw, wrapped to [−π, π] |
| 4–5 | pitch_rad, roll_rad | Car body angles |
| 6 | track_progress | [0, 1] along track |
| 7 | turning_rate | |
| 8–11 | wheel contacts | 4 wheels (bool as float) |
| 12–14 | angular_velocity | x, y, z |

**Action space:** Discrete(9) — {brake, coast, accel} × {left, straight, right}

**Termination:**
- Finished: `track_progress >= 1.0`
- Crashed: `|lateral_offset| > crash_threshold_m` (default 25 m)
- Truncated: episode time exceeded or hard crash `> 50 m`

**Reward components** (weights in `experiments/<name>/reward_config.yaml`):
- Progress reward (primary signal — proportional to track progress delta)
- Centerline penalty (quadratic in lateral offset)
- Speed bonus (small, breaks ties)
- Acceleration bonus (flat per step when throttle is pressed — discourages coasting)
- Per-step time penalty (encourages finishing fast)
- Finish bonus + finish time bonus/penalty relative to par time
- Airborne penalty (only when below/beside centreline)

### WeightedLinearPolicy
The trainable policy computes a score for each of the 9 actions using a learned weight vector dotted against a normalised observation vector. The action with the highest score is taken. Weights are stored in YAML and mutated by adding Gaussian noise scaled per-feature (normalised so all features contribute equally to each mutation step).

### Threading model
TMInterface is callback-driven (`on_run_step`); the RL loop is step-driven (`env.step()`). `RLClient` bridges these with a thread-safe action queue + state queue. The interface runs on a dedicated keepalive thread; `on_registered` sets an event that `TMNFEnv.__init__` waits on before returning.

### Key design decisions
- Reward weights live in per-experiment YAML so they can be tuned without touching code, and different experiments can have different reward shaping.
- Game speed can be set up to 10× during training for faster data collection.
- Adaptive client uses three steering terms (lateral P, lateral-velocity D, heading feedforward) — a strong hand-tuned baseline.
- `WeightedLinearPolicy.mutated()` normalises weights before adding noise so a mutation of `scale=0.1` means the same thing regardless of which feature's weight is being perturbed.

---

## Dependencies

Managed by Poetry. Run `poetry install` to create `.venv/` and install all dependencies.

`tminterface` and `pygbx` are not on PyPI — install from source before running `poetry install`.
