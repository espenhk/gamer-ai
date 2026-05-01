# Experiment: gs_cmaes_v1__pop10__sigma0.1

**Track:** a03_centerline

## Timings

- **Start:** 2026-04-29 00:48:31
- **End:** 2026-04-29 01:05:37
- **Total runtime:** 17m 06.1s

| Phase | Duration |
|-------|----------|
| Greedy | 17m 05.1s |

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
| initial_sigma | 0.1 |

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

Best reward: **+6612.7**

| Sim  | Reward   | Reason       | Result       |
|------|----------|--------------|-------------|
|    1 |  +2234.3 | timeout      | **NEW BEST** |
|    2 |  +1558.4 | timeout      |  |
|    3 |  +1122.3 | timeout      |  |
|    4 |  +1282.2 | timeout      |  |
|    5 |  +1646.3 | timeout      |  |
|    6 |  +2827.7 | crash        | **NEW BEST** |
|    7 |  +3111.5 | timeout      | **NEW BEST** |
|    8 |  +3285.5 | crash        | **NEW BEST** |
|    9 |  +2616.8 | timeout      |  |
|   10 |  +3055.2 | timeout      |  |
|   11 |  +4326.8 | timeout      | **NEW BEST** |
|   12 |  +4366.0 | timeout      | **NEW BEST** |
|   13 |  +4682.4 | timeout      | **NEW BEST** |
|   14 |  +4661.4 | timeout      |  |
|   15 |  +4850.1 | timeout      | **NEW BEST** |
|   16 |  +3554.5 | timeout      |  |
|   17 |  +5993.2 | timeout      | **NEW BEST** |
|   18 |  +6517.1 | timeout      | **NEW BEST** |
|   19 |  +4763.6 | crash        |  |
|   20 |  +6612.7 | timeout      | **NEW BEST** |

![Greedy rewards](greedy_rewards.png)

![Greedy progress](greedy_progress.png)

![Greedy best run](greedy_best_run.png)

![Weight evolution](greedy_weight_evolution.png)

![Termination reasons](termination_reasons.png)

## Additional Plots

![Greedy action distribution](greedy_action_dist.png)

![Reward trajectory](reward_trajectory.png)

![Policy weight heatmap](policy_weights_heatmap.png)

