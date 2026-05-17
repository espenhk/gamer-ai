# Experiment: gs_sc2_cmaes_v1__arterran__pop50__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-11 01:59:18
- **End:** 2026-05-11 14:29:52
- **Total runtime:** 12h 30m 34.0s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 30m 33.0s |

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
| population_size | 50 |
| initial_sigma | 0.5 |
| policy_params | {'eval_episodes': 5, 'population_size': 50, 'initial_sigma': 0.5} |

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

Best reward: **+796.3**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +354.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +653.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +712.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +363.9 | 0.000    | —           | —       | finish       |  |
|    5 |   +268.9 | 0.000    | —           | —       | finish       |  |
|    6 |   +361.3 | 0.000    | —           | —       | finish       |  |
|    7 |   +404.6 | 0.000    | —           | —       | finish       |  |
|    8 |   +453.8 | 0.000    | —           | —       | finish       |  |
|    9 |   +495.0 | 0.000    | —           | —       | finish       |  |
|   10 |   +531.4 | 0.000    | —           | —       | finish       |  |
|   11 |   +663.0 | 0.000    | —           | —       | finish       |  |
|   12 |   +631.8 | 0.000    | —           | —       | finish       |  |
|   13 |   +410.1 | 0.000    | —           | —       | finish       |  |
|   14 |   +538.6 | 0.000    | —           | —       | finish       |  |
|   15 |   +430.0 | 0.000    | —           | —       | finish       |  |
|   16 |   +578.3 | 0.000    | —           | —       | finish       |  |
|   17 |   +515.8 | 0.000    | —           | —       | finish       |  |
|   18 |   +469.0 | 0.000    | —           | —       | finish       |  |
|   19 |   +552.8 | 0.000    | —           | —       | finish       |  |
|   20 |   +420.3 | 0.000    | —           | —       | finish       |  |
|   21 |   +767.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   22 |   +428.6 | 0.000    | —           | —       | finish       |  |
|   23 |   +540.2 | 0.000    | —           | —       | finish       |  |
|   24 |   +457.1 | 0.000    | —           | —       | finish       |  |
|   25 |   +528.1 | 0.000    | —           | —       | finish       |  |
|   26 |   +588.0 | 0.000    | —           | —       | finish       |  |
|   27 |   +584.3 | 0.000    | —           | —       | finish       |  |
|   28 |   +568.7 | 0.000    | —           | —       | finish       |  |
|   29 |   +582.0 | 0.000    | —           | —       | finish       |  |
|   30 |   +562.6 | 0.000    | —           | —       | finish       |  |
|   31 |   +546.7 | 0.000    | —           | —       | finish       |  |
|   32 |   +587.8 | 0.000    | —           | —       | finish       |  |
|   33 |   +616.9 | 0.000    | —           | —       | finish       |  |
|   34 |   +594.1 | 0.000    | —           | —       | finish       |  |
|   35 |   +590.2 | 0.000    | —           | —       | finish       |  |
|   36 |   +534.9 | 0.000    | —           | —       | finish       |  |
|   37 |   +656.8 | 0.000    | —           | —       | finish       |  |
|   38 |   +572.5 | 0.000    | —           | —       | finish       |  |
|   39 |   +585.4 | 0.000    | —           | —       | finish       |  |
|   40 |   +796.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   41 |   +557.5 | 0.000    | —           | —       | finish       |  |
|   42 |   +479.5 | 0.000    | —           | —       | finish       |  |
|   43 |   +521.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +534.4 | 0.000    | —           | —       | finish       |  |
|   45 |   +482.1 | 0.000    | —           | —       | finish       |  |
|   46 |   +553.3 | 0.000    | —           | —       | finish       |  |
|   47 |   +575.9 | 0.000    | —           | —       | finish       |  |
|   48 |   +659.5 | 0.000    | —           | —       | finish       |  |
|   49 |   +534.2 | 0.000    | —           | —       | finish       |  |
|   50 |   +588.7 | 0.000    | —           | —       | finish       |  |

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

