# Experiment: gs_sc2_cmaes_v1__arterran__pop15__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-09 04:02:59
- **End:** 2026-05-09 07:46:50
- **Total runtime:** 3h 43m 50.9s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 43m 49.8s |

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

Best reward: **+602.4**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +251.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +237.0 | 0.000    | —           | —       | finish       |  |
|    3 |   +450.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +316.2 | 0.000    | —           | —       | finish       |  |
|    5 |   +395.1 | 0.000    | —           | —       | finish       |  |
|    6 |   +394.9 | 0.000    | —           | —       | finish       |  |
|    7 |   +388.3 | 0.000    | —           | —       | finish       |  |
|    8 |   +291.5 | 0.000    | —           | —       | finish       |  |
|    9 |   +349.2 | 0.000    | —           | —       | finish       |  |
|   10 |   +376.7 | 0.000    | —           | —       | finish       |  |
|   11 |   +342.8 | 0.000    | —           | —       | finish       |  |
|   12 |   +333.2 | 0.000    | —           | —       | finish       |  |
|   13 |   +393.4 | 0.000    | —           | —       | finish       |  |
|   14 |   +372.1 | 0.000    | —           | —       | finish       |  |
|   15 |   +298.3 | 0.000    | —           | —       | finish       |  |
|   16 |   +378.7 | 0.000    | —           | —       | finish       |  |
|   17 |   +374.8 | 0.000    | —           | —       | finish       |  |
|   18 |   +378.7 | 0.000    | —           | —       | finish       |  |
|   19 |   +386.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +323.6 | 0.000    | —           | —       | finish       |  |
|   21 |   +310.4 | 0.000    | —           | —       | finish       |  |
|   22 |   +299.7 | 0.000    | —           | —       | finish       |  |
|   23 |   +360.7 | 0.000    | —           | —       | finish       |  |
|   24 |   +270.7 | 0.000    | —           | —       | finish       |  |
|   25 |   +347.5 | 0.000    | —           | —       | finish       |  |
|   26 |   +403.2 | 0.000    | —           | —       | finish       |  |
|   27 |   +329.0 | 0.000    | —           | —       | finish       |  |
|   28 |   +349.4 | 0.000    | —           | —       | finish       |  |
|   29 |   +344.8 | 0.000    | —           | —       | finish       |  |
|   30 |   +431.0 | 0.000    | —           | —       | finish       |  |
|   31 |   +515.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   32 |   +483.5 | 0.000    | —           | —       | finish       |  |
|   33 |   +489.7 | 0.000    | —           | —       | finish       |  |
|   34 |   +486.6 | 0.000    | —           | —       | finish       |  |
|   35 |   +491.8 | 0.000    | —           | —       | finish       |  |
|   36 |   +298.1 | 0.000    | —           | —       | finish       |  |
|   37 |   +362.5 | 0.000    | —           | —       | finish       |  |
|   38 |   +440.7 | 0.000    | —           | —       | finish       |  |
|   39 |   +489.8 | 0.000    | —           | —       | finish       |  |
|   40 |   +459.9 | 0.000    | —           | —       | finish       |  |
|   41 |   +502.0 | 0.000    | —           | —       | finish       |  |
|   42 |   +436.6 | 0.000    | —           | —       | finish       |  |
|   43 |   +385.0 | 0.000    | —           | —       | finish       |  |
|   44 |   +602.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   45 |   +510.1 | 0.000    | —           | —       | finish       |  |
|   46 |   +446.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +344.7 | 0.000    | —           | —       | finish       |  |
|   48 |   +593.3 | 0.000    | —           | —       | finish       |  |
|   49 |   +598.6 | 0.000    | —           | —       | finish       |  |
|   50 |   +335.9 | 0.000    | —           | —       | finish       |  |

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

