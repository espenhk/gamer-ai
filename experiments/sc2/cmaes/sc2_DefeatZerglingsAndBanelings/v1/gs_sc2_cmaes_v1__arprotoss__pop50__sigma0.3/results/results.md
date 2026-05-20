# Experiment: gs_sc2_cmaes_v1__arprotoss__pop50__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-16 00:50:53
- **End:** 2026-05-16 13:25:39
- **Total runtime:** 12h 34m 46.6s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 34m 45.5s |

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

Best reward: **+754.0**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +418.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +358.5 | 0.000    | —           | —       | finish       |  |
|    3 |   +349.2 | 0.000    | —           | —       | finish       |  |
|    4 |   +336.4 | 0.000    | —           | —       | finish       |  |
|    5 |   +333.2 | 0.000    | —           | —       | finish       |  |
|    6 |   +470.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +350.5 | 0.000    | —           | —       | finish       |  |
|    8 |   +424.2 | 0.000    | —           | —       | finish       |  |
|    9 |   +395.3 | 0.000    | —           | —       | finish       |  |
|   10 |   +434.8 | 0.000    | —           | —       | finish       |  |
|   11 |   +428.6 | 0.000    | —           | —       | finish       |  |
|   12 |   +424.0 | 0.000    | —           | —       | finish       |  |
|   13 |   +653.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +482.9 | 0.000    | —           | —       | finish       |  |
|   15 |   +413.8 | 0.000    | —           | —       | finish       |  |
|   16 |   +335.1 | 0.000    | —           | —       | finish       |  |
|   17 |   +499.9 | 0.000    | —           | —       | finish       |  |
|   18 |   +514.8 | 0.000    | —           | —       | finish       |  |
|   19 |   +523.2 | 0.000    | —           | —       | finish       |  |
|   20 |   +693.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   21 |   +674.6 | 0.000    | —           | —       | finish       |  |
|   22 |   +521.1 | 0.000    | —           | —       | finish       |  |
|   23 |   +467.8 | 0.000    | —           | —       | finish       |  |
|   24 |   +553.8 | 0.000    | —           | —       | finish       |  |
|   25 |   +445.0 | 0.000    | —           | —       | finish       |  |
|   26 |   +603.1 | 0.000    | —           | —       | finish       |  |
|   27 |   +491.3 | 0.000    | —           | —       | finish       |  |
|   28 |   +421.1 | 0.000    | —           | —       | finish       |  |
|   29 |   +566.1 | 0.000    | —           | —       | finish       |  |
|   30 |   +488.6 | 0.000    | —           | —       | finish       |  |
|   31 |   +512.7 | 0.000    | —           | —       | finish       |  |
|   32 |   +621.2 | 0.000    | —           | —       | finish       |  |
|   33 |   +535.1 | 0.000    | —           | —       | finish       |  |
|   34 |   +500.2 | 0.000    | —           | —       | finish       |  |
|   35 |   +574.6 | 0.000    | —           | —       | finish       |  |
|   36 |   +610.3 | 0.000    | —           | —       | finish       |  |
|   37 |   +545.7 | 0.000    | —           | —       | finish       |  |
|   38 |   +562.7 | 0.000    | —           | —       | finish       |  |
|   39 |   +525.8 | 0.000    | —           | —       | finish       |  |
|   40 |   +592.9 | 0.000    | —           | —       | finish       |  |
|   41 |   +436.5 | 0.000    | —           | —       | finish       |  |
|   42 |   +455.2 | 0.000    | —           | —       | finish       |  |
|   43 |   +503.1 | 0.000    | —           | —       | finish       |  |
|   44 |   +754.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   45 |   +458.8 | 0.000    | —           | —       | finish       |  |
|   46 |   +567.5 | 0.000    | —           | —       | finish       |  |
|   47 |   +495.6 | 0.000    | —           | —       | finish       |  |
|   48 |   +545.8 | 0.000    | —           | —       | finish       |  |
|   49 |   +573.5 | 0.000    | —           | —       | finish       |  |
|   50 |   +534.4 | 0.000    | —           | —       | finish       |  |

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

