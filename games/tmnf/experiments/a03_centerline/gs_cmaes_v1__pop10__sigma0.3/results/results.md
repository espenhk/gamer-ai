# Experiment: gs_cmaes_v1__pop10__sigma0.3

**Track:** a03_centerline

## Timings

- **Start:** 2026-04-29 01:05:48
- **End:** 2026-04-29 01:22:14
- **Total runtime:** 16m 26.1s

| Phase | Duration |
|-------|----------|
| Greedy | 16m 25.0s |

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
| population_size | 10 |
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

Best reward: **+6508.2**

| Sim  | Reward   | Reason       | Result       |
|------|----------|--------------|-------------|
|    1 |  +1265.4 | timeout      | **NEW BEST** |
|    2 |  +1565.4 | timeout      | **NEW BEST** |
|    3 |  +1208.4 | timeout      |  |
|    4 |  +1547.4 | timeout      |  |
|    5 |  +1599.8 | timeout      | **NEW BEST** |
|    6 |  +2714.2 | crash        | **NEW BEST** |
|    7 |  +3042.6 | timeout      | **NEW BEST** |
|    8 |  +3090.6 | timeout      | **NEW BEST** |
|    9 |  +3019.1 | crash        |  |
|   10 |  +3013.3 | timeout      |  |
|   11 |  +4465.5 | timeout      | **NEW BEST** |
|   12 |  +4613.8 | crash        | **NEW BEST** |
|   13 |  +5470.6 | timeout      | **NEW BEST** |
|   14 |  +5695.6 | timeout      | **NEW BEST** |
|   15 |  +2834.7 | timeout      |  |
|   16 |  +4096.0 | crash        |  |
|   17 |  +6508.2 | timeout      | **NEW BEST** |
|   18 |  +4273.2 | timeout      |  |
|   19 |  +6341.0 | timeout      |  |
|   20 |  +5733.6 | crash        |  |

![Greedy rewards](greedy_rewards.png)

![Greedy progress](greedy_progress.png)

![Greedy best run](greedy_best_run.png)

![Weight evolution](greedy_weight_evolution.png)

![Termination reasons](termination_reasons.png)

## Additional Plots

![Greedy action distribution](greedy_action_dist.png)

![Reward trajectory](reward_trajectory.png)

![Policy weight heatmap](policy_weights_heatmap.png)

