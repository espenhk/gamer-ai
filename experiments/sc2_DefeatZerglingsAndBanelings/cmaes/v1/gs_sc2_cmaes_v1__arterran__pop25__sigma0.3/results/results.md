# Experiment: gs_sc2_cmaes_v1__arterran__pop25__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-09 13:08:53
- **End:** 2026-05-09 19:35:15
- **Total runtime:** 6h 26m 21.7s

| Phase | Duration |
|-------|----------|
| Greedy | 6h 26m 20.7s |

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

Best reward: **+663.8**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +279.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +341.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +342.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +630.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +425.3 | 0.000    | —           | —       | finish       |  |
|    6 |   +658.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +470.2 | 0.000    | —           | —       | finish       |  |
|    8 |   +367.8 | 0.000    | —           | —       | finish       |  |
|    9 |   +471.1 | 0.000    | —           | —       | finish       |  |
|   10 |   +494.1 | 0.000    | —           | —       | finish       |  |
|   11 |   +545.2 | 0.000    | —           | —       | finish       |  |
|   12 |   +536.3 | 0.000    | —           | —       | finish       |  |
|   13 |   +549.5 | 0.000    | —           | —       | finish       |  |
|   14 |   +551.6 | 0.000    | —           | —       | finish       |  |
|   15 |   +440.2 | 0.000    | —           | —       | finish       |  |
|   16 |   +447.0 | 0.000    | —           | —       | finish       |  |
|   17 |   +614.3 | 0.000    | —           | —       | finish       |  |
|   18 |   +542.6 | 0.000    | —           | —       | finish       |  |
|   19 |   +551.3 | 0.000    | —           | —       | finish       |  |
|   20 |   +466.6 | 0.000    | —           | —       | finish       |  |
|   21 |   +366.2 | 0.000    | —           | —       | finish       |  |
|   22 |   +450.0 | 0.000    | —           | —       | finish       |  |
|   23 |   +663.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |   +479.7 | 0.000    | —           | —       | finish       |  |
|   25 |   +528.0 | 0.000    | —           | —       | finish       |  |
|   26 |   +373.8 | 0.000    | —           | —       | finish       |  |
|   27 |   +414.4 | 0.000    | —           | —       | finish       |  |
|   28 |   +465.4 | 0.000    | —           | —       | finish       |  |
|   29 |   +413.1 | 0.000    | —           | —       | finish       |  |
|   30 |   +531.9 | 0.000    | —           | —       | finish       |  |
|   31 |   +428.9 | 0.000    | —           | —       | finish       |  |
|   32 |   +349.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +388.8 | 0.000    | —           | —       | finish       |  |
|   34 |   +470.5 | 0.000    | —           | —       | finish       |  |
|   35 |   +501.7 | 0.000    | —           | —       | finish       |  |
|   36 |   +577.6 | 0.000    | —           | —       | finish       |  |
|   37 |   +459.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +433.8 | 0.000    | —           | —       | finish       |  |
|   39 |   +414.9 | 0.000    | —           | —       | finish       |  |
|   40 |   +460.2 | 0.000    | —           | —       | finish       |  |
|   41 |   +488.7 | 0.000    | —           | —       | finish       |  |
|   42 |   +449.0 | 0.000    | —           | —       | finish       |  |
|   43 |   +395.0 | 0.000    | —           | —       | finish       |  |
|   44 |   +308.8 | 0.000    | —           | —       | finish       |  |
|   45 |   +447.6 | 0.000    | —           | —       | finish       |  |
|   46 |   +459.6 | 0.000    | —           | —       | finish       |  |
|   47 |   +518.7 | 0.000    | —           | —       | finish       |  |
|   48 |   +499.2 | 0.000    | —           | —       | finish       |  |
|   49 |   +422.8 | 0.000    | —           | —       | finish       |  |
|   50 |   +473.2 | 0.000    | —           | —       | finish       |  |

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

