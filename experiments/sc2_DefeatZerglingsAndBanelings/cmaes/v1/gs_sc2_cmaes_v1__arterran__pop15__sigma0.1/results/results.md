# Experiment: gs_sc2_cmaes_v1__arterran__pop15__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 20:58:09
- **End:** 2026-05-09 00:20:31
- **Total runtime:** 3h 22m 22.1s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 22m 21.0s |

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
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.1} |

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

Best reward: **+848.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +254.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +257.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +201.3 | 0.000    | —           | —       | finish       |  |
|    4 |   +332.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +476.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |   +317.5 | 0.000    | —           | —       | finish       |  |
|    7 |   +299.3 | 0.000    | —           | —       | finish       |  |
|    8 |   +534.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +515.6 | 0.000    | —           | —       | finish       |  |
|   10 |   +620.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   11 |   +501.1 | 0.000    | —           | —       | finish       |  |
|   12 |   +549.4 | 0.000    | —           | —       | finish       |  |
|   13 |   +527.0 | 0.000    | —           | —       | finish       |  |
|   14 |   +515.4 | 0.000    | —           | —       | finish       |  |
|   15 |   +431.2 | 0.000    | —           | —       | finish       |  |
|   16 |   +449.9 | 0.000    | —           | —       | finish       |  |
|   17 |   +629.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   18 |   +625.9 | 0.000    | —           | —       | finish       |  |
|   19 |   +565.3 | 0.000    | —           | —       | finish       |  |
|   20 |   +457.7 | 0.000    | —           | —       | finish       |  |
|   21 |   +572.2 | 0.000    | —           | —       | finish       |  |
|   22 |   +590.3 | 0.000    | —           | —       | finish       |  |
|   23 |   +620.9 | 0.000    | —           | —       | finish       |  |
|   24 |   +644.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |   +535.9 | 0.000    | —           | —       | finish       |  |
|   26 |   +557.3 | 0.000    | —           | —       | finish       |  |
|   27 |   +667.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   28 |   +587.1 | 0.000    | —           | —       | finish       |  |
|   29 |   +436.2 | 0.000    | —           | —       | finish       |  |
|   30 |   +412.9 | 0.000    | —           | —       | finish       |  |
|   31 |   +539.6 | 0.000    | —           | —       | finish       |  |
|   32 |   +457.2 | 0.000    | —           | —       | finish       |  |
|   33 |   +565.5 | 0.000    | —           | —       | finish       |  |
|   34 |   +611.5 | 0.000    | —           | —       | finish       |  |
|   35 |   +688.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   36 |   +795.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   37 |   +613.5 | 0.000    | —           | —       | finish       |  |
|   38 |   +542.0 | 0.000    | —           | —       | finish       |  |
|   39 |   +583.3 | 0.000    | —           | —       | finish       |  |
|   40 |   +321.4 | 0.000    | —           | —       | finish       |  |
|   41 |   +432.5 | 0.000    | —           | —       | finish       |  |
|   42 |   +616.6 | 0.000    | —           | —       | finish       |  |
|   43 |   +480.3 | 0.000    | —           | —       | finish       |  |
|   44 |   +552.9 | 0.000    | —           | —       | finish       |  |
|   45 |   +637.4 | 0.000    | —           | —       | finish       |  |
|   46 |   +848.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   47 |   +570.2 | 0.000    | —           | —       | finish       |  |
|   48 |   +487.3 | 0.000    | —           | —       | finish       |  |
|   49 |   +765.5 | 0.000    | —           | —       | finish       |  |
|   50 |   +505.9 | 0.000    | —           | —       | finish       |  |

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

