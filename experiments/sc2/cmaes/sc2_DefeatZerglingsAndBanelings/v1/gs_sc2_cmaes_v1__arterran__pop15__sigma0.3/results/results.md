# Experiment: gs_sc2_cmaes_v1__arterran__pop15__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-09 00:20:44
- **End:** 2026-05-09 04:02:46
- **Total runtime:** 3h 42m 02.1s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 42m 01.1s |

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
| agent_race | terran |
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

Best reward: **+501.3**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +141.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +315.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |    +87.4 | 0.000    | —           | —       | finish       |  |
|    4 |   +229.8 | 0.000    | —           | —       | finish       |  |
|    5 |   +296.0 | 0.000    | —           | —       | finish       |  |
|    6 |   +162.7 | 0.000    | —           | —       | finish       |  |
|    7 |   +247.8 | 0.000    | —           | —       | finish       |  |
|    8 |   +217.1 | 0.000    | —           | —       | finish       |  |
|    9 |    +82.0 | 0.000    | —           | —       | finish       |  |
|   10 |   +213.4 | 0.000    | —           | —       | finish       |  |
|   11 |   +113.6 | 0.000    | —           | —       | finish       |  |
|   12 |   +297.1 | 0.000    | —           | —       | finish       |  |
|   13 |   +297.8 | 0.000    | —           | —       | finish       |  |
|   14 |   +337.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |   +215.6 | 0.000    | —           | —       | finish       |  |
|   16 |   +256.4 | 0.000    | —           | —       | finish       |  |
|   17 |   +321.8 | 0.000    | —           | —       | finish       |  |
|   18 |   +238.9 | 0.000    | —           | —       | finish       |  |
|   19 |   +212.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +331.0 | 0.000    | —           | —       | finish       |  |
|   21 |   +325.1 | 0.000    | —           | —       | finish       |  |
|   22 |   +294.0 | 0.000    | —           | —       | finish       |  |
|   23 |   +275.1 | 0.000    | —           | —       | finish       |  |
|   24 |   +348.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |   +360.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   26 |   +292.7 | 0.000    | —           | —       | finish       |  |
|   27 |   +347.8 | 0.000    | —           | —       | finish       |  |
|   28 |   +458.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   29 |   +320.4 | 0.000    | —           | —       | finish       |  |
|   30 |   +274.9 | 0.000    | —           | —       | finish       |  |
|   31 |   +342.1 | 0.000    | —           | —       | finish       |  |
|   32 |   +364.2 | 0.000    | —           | —       | finish       |  |
|   33 |   +380.7 | 0.000    | —           | —       | finish       |  |
|   34 |   +179.0 | 0.000    | —           | —       | finish       |  |
|   35 |   +354.1 | 0.000    | —           | —       | finish       |  |
|   36 |   +397.1 | 0.000    | —           | —       | finish       |  |
|   37 |   +272.3 | 0.000    | —           | —       | finish       |  |
|   38 |   +329.7 | 0.000    | —           | —       | finish       |  |
|   39 |   +369.2 | 0.000    | —           | —       | finish       |  |
|   40 |   +499.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   41 |   +306.2 | 0.000    | —           | —       | finish       |  |
|   42 |   +454.5 | 0.000    | —           | —       | finish       |  |
|   43 |   +357.7 | 0.000    | —           | —       | finish       |  |
|   44 |   +501.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   45 |   +425.1 | 0.000    | —           | —       | finish       |  |
|   46 |   +470.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +330.9 | 0.000    | —           | —       | finish       |  |
|   48 |   +318.4 | 0.000    | —           | —       | finish       |  |
|   49 |   +367.7 | 0.000    | —           | —       | finish       |  |
|   50 |   +410.9 | 0.000    | —           | —       | finish       |  |

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

