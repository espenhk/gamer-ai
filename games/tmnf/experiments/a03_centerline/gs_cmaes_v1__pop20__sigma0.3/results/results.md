# Experiment: gs_cmaes_v1__pop20__sigma0.3

**Track:** a03_centerline

## Timings

- **Start:** 2026-04-29 01:59:34
- **End:** 2026-04-29 02:32:04
- **Total runtime:** 32m 30.1s

| Phase | Duration |
|-------|----------|
| Greedy | 32m 29.1s |

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
| population_size | 20 |
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

Best reward: **+6465.6**

| Sim  | Reward   | Reason       | Result       |
|------|----------|--------------|-------------|
|    1 |  +1402.6 | timeout      | **NEW BEST** |
|    2 |  +1315.9 | timeout      |  |
|    3 |  +1489.5 | timeout      | **NEW BEST** |
|    4 |  +1686.4 | timeout      | **NEW BEST** |
|    5 |  +2171.0 | timeout      | **NEW BEST** |
|    6 |  +2787.1 | timeout      | **NEW BEST** |
|    7 |  +2993.8 | crash        | **NEW BEST** |
|    8 |  +3269.4 | timeout      | **NEW BEST** |
|    9 |  +3014.2 | timeout      |  |
|   10 |  +3066.6 | timeout      |  |
|   11 |  +4818.4 | timeout      | **NEW BEST** |
|   12 |  +4874.9 | timeout      | **NEW BEST** |
|   13 |  +4811.3 | timeout      |  |
|   14 |  +4817.5 | timeout      |  |
|   15 |  +4744.6 | timeout      |  |
|   16 |  +6349.7 | timeout      | **NEW BEST** |
|   17 |  +6403.7 | timeout      | **NEW BEST** |
|   18 |  +6437.5 | timeout      | **NEW BEST** |
|   19 |  +6465.6 | crash        | **NEW BEST** |
|   20 |  +6458.7 | timeout      |  |

![Greedy rewards](greedy_rewards.png)

![Greedy progress](greedy_progress.png)

![Greedy best run](greedy_best_run.png)

![Weight evolution](greedy_weight_evolution.png)

![Termination reasons](termination_reasons.png)

## Additional Plots

![Greedy action distribution](greedy_action_dist.png)

![Reward trajectory](reward_trajectory.png)

![Policy weight heatmap](policy_weights_heatmap.png)

