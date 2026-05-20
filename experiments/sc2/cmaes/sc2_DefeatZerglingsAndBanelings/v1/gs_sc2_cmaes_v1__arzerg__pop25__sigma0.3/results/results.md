# Experiment: gs_sc2_cmaes_v1__arzerg__pop25__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-12 06:57:11
- **End:** 2026-05-12 13:17:07
- **Total runtime:** 6h 19m 55.6s

| Phase | Duration |
|-------|----------|
| Greedy | 6h 19m 54.6s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatZerglingsAndBanelings |
| map_name | DefeatZerglingsAndBanelings |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | zerg |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 25 |
| initial_sigma | 0.3 |
| policy_params | {'eval_episodes': 5, 'population_size': 25, 'initial_sigma': 0.3} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 10.0 |
| win_bonus | 1000.0 |
| loss_penalty | -100.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 0.5 |
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+639.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +354.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +450.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +408.1 | 0.000    | —           | —       | finish       |  |
|    4 |   +379.3 | 0.000    | —           | —       | finish       |  |
|    5 |   +198.5 | 0.000    | —           | —       | finish       |  |
|    6 |   +388.4 | 0.000    | —           | —       | finish       |  |
|    7 |   +443.7 | 0.000    | —           | —       | finish       |  |
|    8 |   +283.0 | 0.000    | —           | —       | finish       |  |
|    9 |   +445.6 | 0.000    | —           | —       | finish       |  |
|   10 |   +421.5 | 0.000    | —           | —       | finish       |  |
|   11 |   +408.3 | 0.000    | —           | —       | finish       |  |
|   12 |   +522.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   13 |   +442.9 | 0.000    | —           | —       | finish       |  |
|   14 |   +450.5 | 0.000    | —           | —       | finish       |  |
|   15 |   +499.4 | 0.000    | —           | —       | finish       |  |
|   16 |   +373.0 | 0.000    | —           | —       | finish       |  |
|   17 |   +400.5 | 0.000    | —           | —       | finish       |  |
|   18 |   +456.2 | 0.000    | —           | —       | finish       |  |
|   19 |   +625.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   20 |   +552.6 | 0.000    | —           | —       | finish       |  |
|   21 |   +510.3 | 0.000    | —           | —       | finish       |  |
|   22 |   +473.4 | 0.000    | —           | —       | finish       |  |
|   23 |   +482.4 | 0.000    | —           | —       | finish       |  |
|   24 |   +492.2 | 0.000    | —           | —       | finish       |  |
|   25 |   +453.2 | 0.000    | —           | —       | finish       |  |
|   26 |   +402.2 | 0.000    | —           | —       | finish       |  |
|   27 |   +395.7 | 0.000    | —           | —       | finish       |  |
|   28 |   +548.8 | 0.000    | —           | —       | finish       |  |
|   29 |   +419.0 | 0.000    | —           | —       | finish       |  |
|   30 |   +393.2 | 0.000    | —           | —       | finish       |  |
|   31 |   +559.8 | 0.000    | —           | —       | finish       |  |
|   32 |   +443.0 | 0.000    | —           | —       | finish       |  |
|   33 |   +481.6 | 0.000    | —           | —       | finish       |  |
|   34 |   +509.4 | 0.000    | —           | —       | finish       |  |
|   35 |   +464.8 | 0.000    | —           | —       | finish       |  |
|   36 |   +510.9 | 0.000    | —           | —       | finish       |  |
|   37 |   +464.5 | 0.000    | —           | —       | finish       |  |
|   38 |   +639.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   39 |   +627.4 | 0.000    | —           | —       | finish       |  |
|   40 |   +414.6 | 0.000    | —           | —       | finish       |  |
|   41 |   +570.0 | 0.000    | —           | —       | finish       |  |
|   42 |   +527.1 | 0.000    | —           | —       | finish       |  |
|   43 |   +356.8 | 0.000    | —           | —       | finish       |  |
|   44 |   +480.9 | 0.000    | —           | —       | finish       |  |
|   45 |   +519.2 | 0.000    | —           | —       | finish       |  |
|   46 |   +476.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +550.3 | 0.000    | —           | —       | finish       |  |
|   48 |   +532.8 | 0.000    | —           | —       | finish       |  |
|   49 |   +392.7 | 0.000    | —           | —       | finish       |  |
|   50 |   +479.0 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Skipped frames](skipped_frames.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Build order](build_order.png)


![Reward trajectory](reward_trajectory.png)

