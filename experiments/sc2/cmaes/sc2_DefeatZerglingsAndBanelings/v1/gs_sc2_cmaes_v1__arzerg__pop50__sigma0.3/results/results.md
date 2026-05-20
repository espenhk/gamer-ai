# Experiment: gs_sc2_cmaes_v1__arzerg__pop50__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-13 07:04:53
- **End:** 2026-05-13 19:36:24
- **Total runtime:** 12h 31m 31.0s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 31m 30.0s |

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
| population_size | 50 |
| initial_sigma | 0.3 |
| policy_params | {'eval_episodes': 5, 'population_size': 50, 'initial_sigma': 0.3} |

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

Best reward: **+722.6**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +283.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +244.3 | 0.000    | —           | —       | finish       |  |
|    3 |   +297.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +558.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +537.4 | 0.000    | —           | —       | finish       |  |
|    6 |   +517.7 | 0.000    | —           | —       | finish       |  |
|    7 |   +389.1 | 0.000    | —           | —       | finish       |  |
|    8 |   +370.9 | 0.000    | —           | —       | finish       |  |
|    9 |   +456.3 | 0.000    | —           | —       | finish       |  |
|   10 |   +374.4 | 0.000    | —           | —       | finish       |  |
|   11 |   +521.8 | 0.000    | —           | —       | finish       |  |
|   12 |   +564.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   13 |   +613.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +616.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |   +505.2 | 0.000    | —           | —       | finish       |  |
|   16 |   +471.9 | 0.000    | —           | —       | finish       |  |
|   17 |   +567.6 | 0.000    | —           | —       | finish       |  |
|   18 |   +481.9 | 0.000    | —           | —       | finish       |  |
|   19 |   +543.2 | 0.000    | —           | —       | finish       |  |
|   20 |   +517.2 | 0.000    | —           | —       | finish       |  |
|   21 |   +532.4 | 0.000    | —           | —       | finish       |  |
|   22 |   +483.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +624.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |   +489.9 | 0.000    | —           | —       | finish       |  |
|   25 |   +593.0 | 0.000    | —           | —       | finish       |  |
|   26 |   +439.8 | 0.000    | —           | —       | finish       |  |
|   27 |   +455.3 | 0.000    | —           | —       | finish       |  |
|   28 |   +414.6 | 0.000    | —           | —       | finish       |  |
|   29 |   +642.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   30 |   +425.9 | 0.000    | —           | —       | finish       |  |
|   31 |   +531.9 | 0.000    | —           | —       | finish       |  |
|   32 |   +535.9 | 0.000    | —           | —       | finish       |  |
|   33 |   +571.8 | 0.000    | —           | —       | finish       |  |
|   34 |   +482.0 | 0.000    | —           | —       | finish       |  |
|   35 |   +465.9 | 0.000    | —           | —       | finish       |  |
|   36 |   +502.1 | 0.000    | —           | —       | finish       |  |
|   37 |   +619.5 | 0.000    | —           | —       | finish       |  |
|   38 |   +619.6 | 0.000    | —           | —       | finish       |  |
|   39 |   +536.1 | 0.000    | —           | —       | finish       |  |
|   40 |   +502.2 | 0.000    | —           | —       | finish       |  |
|   41 |   +722.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   42 |   +516.4 | 0.000    | —           | —       | finish       |  |
|   43 |   +490.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +554.1 | 0.000    | —           | —       | finish       |  |
|   45 |   +471.6 | 0.000    | —           | —       | finish       |  |
|   46 |   +436.4 | 0.000    | —           | —       | finish       |  |
|   47 |   +520.1 | 0.000    | —           | —       | finish       |  |
|   48 |   +417.0 | 0.000    | —           | —       | finish       |  |
|   49 |   +530.7 | 0.000    | —           | —       | finish       |  |
|   50 |   +495.1 | 0.000    | —           | —       | finish       |  |

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

