# Experiment: gs_cmaes_v1__pop40__sigma0.3

**Track:** a03_centerline

## Timings

- **Start:** 2026-04-29 03:36:50
- **End:** 2026-04-29 04:42:23
- **Total runtime:** 1h 05m 33.0s

| Phase | Duration |
|-------|----------|
| Greedy | 1h 05m 32.0s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | a03_centerline |
| track | a03_centerline |
| speed | 10.0 |
| n_sims | 20 |
| in_game_episode_s | 90.0 |
| probe_s | 8.0 |
| n_lidar_rays | 8 |
| policy_type | cmaes |
| population_size | 40 |
| initial_sigma | 0.3 |

### Reward Config

| Parameter | Value |
|-----------|-------|
| progress_weight | 10000.0 |
| centerline_weight | 0.0 |
| centerline_exp | 0.0 |
| speed_weight | 0.042 |
| step_penalty | -0.05 |
| finish_bonus | 5000.0 |
| finish_time_weight | -5.0 |
| par_time_s | 60.0 |
| accel_bonus | 0.5 |
| airborne_penalty | -0.83 |
| lidar_wall_weight | -5.0 |
| crash_threshold_m | 25.0 |
| track_name | a03_centerline |
| centerline_path | games/tmnf/tracks/a03_centerline.npy |

## Greedy Phase

Best reward: **+6619.4**

| Sim  | Reward   | Reason       | Result       |
|------|----------|--------------|-------------|
|    1 |  +1559.4 | timeout      | **NEW BEST** |
|    2 |  +1458.5 | timeout      |  |
|    3 |  +3132.2 | timeout      | **NEW BEST** |
|    4 |  +5824.1 | timeout      | **NEW BEST** |
|    5 |  +1675.3 | timeout      |  |
|    6 |  +2547.8 | timeout      |  |
|    7 |  +3276.1 | timeout      |  |
|    8 |  +3138.9 | timeout      |  |
|    9 |  +3124.4 | timeout      |  |
|   10 |  +3343.5 | timeout      |  |
|   11 |  +4827.6 | timeout      |  |
|   12 |  +5094.5 | timeout      |  |
|   13 |  +4921.2 | timeout      |  |
|   14 |  +4740.1 | timeout      |  |
|   15 |  +4717.1 | timeout      |  |
|   16 |  +6588.0 | timeout      | **NEW BEST** |
|   17 |  +6458.1 | timeout      |  |
|   18 |  +6493.8 | timeout      |  |
|   19 |  +6619.4 | timeout      | **NEW BEST** |
|   20 |  +6447.6 | timeout      |  |

![Greedy rewards](greedy_rewards.png)

![Greedy progress](greedy_progress.png)

![Greedy best run](greedy_best_run.png)

![Weight evolution](greedy_weight_evolution.png)

![Termination reasons](termination_reasons.png)

## Additional Plots

![Greedy action distribution](greedy_action_dist.png)

![Reward trajectory](reward_trajectory.png)

![Policy weight heatmap](policy_weights_heatmap.png)

