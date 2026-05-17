# Experiment: gs_sc2_cmaes_v1__arzerg__pop15__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-11 21:39:07
- **End:** 2026-05-12 01:16:58
- **Total runtime:** 3h 37m 51.1s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 37m 50.1s |

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
| population_size | 15 |
| initial_sigma | 0.5 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.5} |

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

Best reward: **+623.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +233.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +319.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +251.9 | 0.000    | —           | —       | finish       |  |
|    4 |   +131.2 | 0.000    | —           | —       | finish       |  |
|    5 |   +318.1 | 0.000    | —           | —       | finish       |  |
|    6 |   +300.7 | 0.000    | —           | —       | finish       |  |
|    7 |   +371.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |   +234.3 | 0.000    | —           | —       | finish       |  |
|    9 |   +237.9 | 0.000    | —           | —       | finish       |  |
|   10 |    +69.4 | 0.000    | —           | —       | finish       |  |
|   11 |   +257.6 | 0.000    | —           | —       | finish       |  |
|   12 |   +275.6 | 0.000    | —           | —       | finish       |  |
|   13 |   +230.8 | 0.000    | —           | —       | finish       |  |
|   14 |   +227.0 | 0.000    | —           | —       | finish       |  |
|   15 |   +195.1 | 0.000    | —           | —       | finish       |  |
|   16 |   +330.3 | 0.000    | —           | —       | finish       |  |
|   17 |   +321.8 | 0.000    | —           | —       | finish       |  |
|   18 |   +278.6 | 0.000    | —           | —       | finish       |  |
|   19 |   +192.7 | 0.000    | —           | —       | finish       |  |
|   20 |   +341.1 | 0.000    | —           | —       | finish       |  |
|   21 |   +303.9 | 0.000    | —           | —       | finish       |  |
|   22 |   +158.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +275.1 | 0.000    | —           | —       | finish       |  |
|   24 |   +377.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |   +339.8 | 0.000    | —           | —       | finish       |  |
|   26 |   +273.6 | 0.000    | —           | —       | finish       |  |
|   27 |   +335.6 | 0.000    | —           | —       | finish       |  |
|   28 |   +623.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   29 |   +313.7 | 0.000    | —           | —       | finish       |  |
|   30 |   +363.6 | 0.000    | —           | —       | finish       |  |
|   31 |   +395.6 | 0.000    | —           | —       | finish       |  |
|   32 |   +384.0 | 0.000    | —           | —       | finish       |  |
|   33 |   +380.1 | 0.000    | —           | —       | finish       |  |
|   34 |   +339.4 | 0.000    | —           | —       | finish       |  |
|   35 |   +238.6 | 0.000    | —           | —       | finish       |  |
|   36 |   +445.4 | 0.000    | —           | —       | finish       |  |
|   37 |   +320.8 | 0.000    | —           | —       | finish       |  |
|   38 |   +324.0 | 0.000    | —           | —       | finish       |  |
|   39 |   +333.7 | 0.000    | —           | —       | finish       |  |
|   40 |   +358.1 | 0.000    | —           | —       | finish       |  |
|   41 |   +198.1 | 0.000    | —           | —       | finish       |  |
|   42 |   +270.3 | 0.000    | —           | —       | finish       |  |
|   43 |   +285.6 | 0.000    | —           | —       | finish       |  |
|   44 |   +314.6 | 0.000    | —           | —       | finish       |  |
|   45 |   +368.0 | 0.000    | —           | —       | finish       |  |
|   46 |   +235.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +403.0 | 0.000    | —           | —       | finish       |  |
|   48 |   +349.0 | 0.000    | —           | —       | finish       |  |
|   49 |   +305.9 | 0.000    | —           | —       | finish       |  |
|   50 |   +367.1 | 0.000    | —           | —       | finish       |  |

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


![Reward trajectory](reward_trajectory.png)

