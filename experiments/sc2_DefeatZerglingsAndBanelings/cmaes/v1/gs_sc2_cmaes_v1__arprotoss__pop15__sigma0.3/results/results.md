# Experiment: gs_sc2_cmaes_v1__arprotoss__pop15__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-14 11:12:21
- **End:** 2026-05-14 14:49:29
- **Total runtime:** 3h 37m 07.9s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 37m 06.9s |

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
| agent_race | protoss |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 15 |
| initial_sigma | 0.3 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.3} |

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

Best reward: **+674.0**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +106.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +348.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +337.2 | 0.000    | —           | —       | finish       |  |
|    4 |   +338.7 | 0.000    | —           | —       | finish       |  |
|    5 |   +378.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |   +201.0 | 0.000    | —           | —       | finish       |  |
|    7 |   +333.5 | 0.000    | —           | —       | finish       |  |
|    8 |   +390.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +367.1 | 0.000    | —           | —       | finish       |  |
|   10 |   +566.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   11 |   +373.9 | 0.000    | —           | —       | finish       |  |
|   12 |   +448.9 | 0.000    | —           | —       | finish       |  |
|   13 |   +296.3 | 0.000    | —           | —       | finish       |  |
|   14 |   +393.8 | 0.000    | —           | —       | finish       |  |
|   15 |   +483.7 | 0.000    | —           | —       | finish       |  |
|   16 |   +545.0 | 0.000    | —           | —       | finish       |  |
|   17 |   +412.4 | 0.000    | —           | —       | finish       |  |
|   18 |   +295.4 | 0.000    | —           | —       | finish       |  |
|   19 |   +351.4 | 0.000    | —           | —       | finish       |  |
|   20 |   +469.2 | 0.000    | —           | —       | finish       |  |
|   21 |   +438.3 | 0.000    | —           | —       | finish       |  |
|   22 |   +194.8 | 0.000    | —           | —       | finish       |  |
|   23 |   +373.0 | 0.000    | —           | —       | finish       |  |
|   24 |   +302.7 | 0.000    | —           | —       | finish       |  |
|   25 |   +428.4 | 0.000    | —           | —       | finish       |  |
|   26 |   +494.9 | 0.000    | —           | —       | finish       |  |
|   27 |   +362.2 | 0.000    | —           | —       | finish       |  |
|   28 |   +329.6 | 0.000    | —           | —       | finish       |  |
|   29 |   +415.9 | 0.000    | —           | —       | finish       |  |
|   30 |   +421.4 | 0.000    | —           | —       | finish       |  |
|   31 |   +329.4 | 0.000    | —           | —       | finish       |  |
|   32 |   +274.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +507.8 | 0.000    | —           | —       | finish       |  |
|   34 |   +375.2 | 0.000    | —           | —       | finish       |  |
|   35 |   +325.3 | 0.000    | —           | —       | finish       |  |
|   36 |   +367.4 | 0.000    | —           | —       | finish       |  |
|   37 |   +339.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +330.8 | 0.000    | —           | —       | finish       |  |
|   39 |   +479.2 | 0.000    | —           | —       | finish       |  |
|   40 |   +482.9 | 0.000    | —           | —       | finish       |  |
|   41 |   +373.3 | 0.000    | —           | —       | finish       |  |
|   42 |   +485.1 | 0.000    | —           | —       | finish       |  |
|   43 |   +634.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   44 |   +492.1 | 0.000    | —           | —       | finish       |  |
|   45 |   +383.4 | 0.000    | —           | —       | finish       |  |
|   46 |   +524.2 | 0.000    | —           | —       | finish       |  |
|   47 |   +442.4 | 0.000    | —           | —       | finish       |  |
|   48 |   +674.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   49 |   +421.8 | 0.000    | —           | —       | finish       |  |
|   50 |   +635.2 | 0.000    | —           | —       | finish       |  |

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

