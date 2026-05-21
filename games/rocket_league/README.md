# Rocket League — gamer-ai integration

Single-agent reinforcement learning for Rocket League via
[RLGym](https://rlgym.org/).

---

## Requirements

| Component | Notes |
|---|---|
| **Rocket League** | Paid commercial game — must own on Steam or Epic Games. Windows only. |
| **Bakkesmod** | Free mod for Rocket League — required for RLGym. Download from [bakkesmod.com](https://www.bakkesmod.com/). |
| **RLGym plugin** | `rlgym_plugin` — install in Bakkesmod's plugin manager. |
| **Python package** | `pip install rlgym` |

> **Platform:** Windows only (Rocket League binary). RLGym can be coordinated
> from any OS via the `distributed/` infra, but the game process must run on
> Windows.

---

## Quick install

1. Own and install Rocket League on Steam or Epic Games.
2. Install Bakkesmod ([bakkesmod.com](https://www.bakkesmod.com/)).
3. In Bakkesmod, open *Plugin manager* and install the **RLGym** plugin.
4. Install the Python package:
   ```bash
   pip install rlgym
   ```
5. Run a smoke test:
   ```bash
   python main.py smoke --game rocket_league --no-interrupt
   ```

---

## Running

```bash
# Single experiment (default config)
python main.py myrun --game rocket_league --no-interrupt

# Grid search (genetic template)
python grid_search.py games/rocket_league/config/gs_genetic_template.yaml --game rocket_league
```

Available Rocket League grid-search templates:

- `games/rocket_league/config/gs_hill_climbing_template.yaml`
- `games/rocket_league/config/gs_neural_net_template.yaml`
- `games/rocket_league/config/gs_epsilon_greedy_template.yaml`
- `games/rocket_league/config/gs_mcts_template.yaml`
- `games/rocket_league/config/gs_genetic_template.yaml`

The first run creates `experiments/rocket_league/<policy>/<track>/<name>/` and
copies both master configs in.  Edit the experiment-local copies to tune
without affecting other runs.

---

## Observation space (70 floats, float32)

| Index range | Group | Description |
|---|---|---|
| 0–17  | **Self car** | pos x/y/z, vel x/y/z, ang\_vel x/y/z, forward x/y/z, up x/y/z, on\_ground, has\_flip, boost\_amount |
| 18–26 | **Ball** | pos x/y/z, vel x/y/z, ang\_vel x/y/z |
| 27–44 | **Opponent car** | same layout as self car |
| 45–47 | **Relative ball position** | ball\_pos − car\_pos |
| 48–50 | **Relative ball velocity** | ball\_vel − car\_vel |
| 51    | `dist_to_ball` | Euclidean distance from car to ball |
| 52    | `vel_towards_ball` | Signed velocity component towards ball |
| 53–56 | **Goal distances** | Ball/car to opponent/own goal |
| 57–59 | **Relative opponent pos** | opp\_pos − car\_pos |
| 60–69 | **Boost pads** | Binary availability of 10 nearest pads |

See `games/rocket_league/obs_spec.py` for the full spec with normalisation
scales.

---

## Action space

`Box([-1,-1,-1,-1,-1, 0, 0, 0], [1,1,1,1,1,1,1,1], shape=(8,), dtype=float32)`

| Index | Name | Range | Notes |
|---|---|---|---|
| 0 | `throttle`  | [−1, 1] | Forward (+) / reverse (−) |
| 1 | `steer`     | [−1, 1] | Left (−) / right (+) |
| 2 | `pitch`     | [−1, 1] | Nose down (−) / up (+); aerial control |
| 3 | `yaw`       | [−1, 1] | Air-only rotation |
| 4 | `roll`      | [−1, 1] | Barrel-roll; aerial only |
| 5 | `jump`      | [0, 1]  | Jump button; thresholded at 0.5 → bool |
| 6 | `boost`     | [0, 1]  | Boost button; thresholded at 0.5 → bool |
| 7 | `handbrake` | [0, 1]  | Powerslide; thresholded at 0.5 → bool |

---

## Reward signal

### `games/rocket_league/config/reward_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `vel_to_ball_weight` | `0.01` | Dense: reward ∝ velocity component towards ball each step. |
| `boost_weight` | `0.0` | Dense: per-step bonus while boost is active. Set negative to penalise waste. |
| `touch_bonus` | `1.0` | Sparse: one-time bonus when the car first touches the ball in an episode. |
| `goal_weight` | `10.0` | Sparse: reward when agent scores. |
| `concede_penalty` | `5.0` | Sparse: penalty when opponent scores (subtracted). |
| `step_penalty` | `-0.001` | Per-step time cost to encourage efficient play. |

**Tuning tips:**
- Start with `vel_to_ball_weight` and `touch_bonus` active to establish basic
  ball-chasing behaviour before adding `goal_weight`.
- Use a negative `boost_weight` (e.g. `-0.002`) to discourage boost spam while
  learning; relax it once the agent understands boost use.

---

## Training config

### `games/rocket_league/config/training_params.yaml`

| Parameter | Default | Description |
|---|---|---|
| `in_game_episode_s` | `300.0` | 5 minutes per episode (RLGym default match length). |
| `tick_skip` | `8` | Physics frames per `step()` call; 8 ≈ 15 Hz. |
| `n_sims` | `100` | Greedy generations for genetic/CMA-ES. |
| `policy_type` | `genetic` | Supported: `hill_climbing`, `neural_net`, `epsilon_greedy`, `mcts`, `genetic`. |

---

## Multi-agent / self-play

This integration is scoped to **single-agent vs built-in bot**.  Team play
and self-play are deferred to future issues that will extend the
`distributed/` coordinator with a "pair workers" mode.

---

## License

RLGym is open-source (Apache 2.0).  Rocket League itself is a commercial
product — you must own a valid copy.  `gamer-ai` does not bundle or
distribute any Rocket League assets.
