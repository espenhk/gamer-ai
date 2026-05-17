# Experiment: gs_sc2_cmaes_v1__arzerg__pop25__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-12 01:17:11
- **End:** 2026-05-12 06:56:58
- **Total runtime:** 5h 39m 46.6s

| Phase | Duration |
|-------|----------|
| Greedy | 5h 39m 45.6s |

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
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 25, 'initial_sigma': 0.1} |

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

Best reward: **+657.4**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +183.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +333.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +138.8 | 0.000    | —           | —       | finish       |  |
|    4 |   +293.5 | 0.000    | —           | —       | finish       |  |
|    5 |   +278.3 | 0.000    | —           | —       | finish       |  |
|    6 |   +388.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +360.6 | 0.000    | —           | —       | finish       |  |
|    8 |   +330.4 | 0.000    | —           | —       | finish       |  |
|    9 |   +504.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |   +373.2 | 0.000    | —           | —       | finish       |  |
|   11 |   +470.6 | 0.000    | —           | —       | finish       |  |
|   12 |   +303.9 | 0.000    | —           | —       | finish       |  |
|   13 |   +544.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +333.3 | 0.000    | —           | —       | finish       |  |
|   15 |   +266.7 | 0.000    | —           | —       | finish       |  |
|   16 |   +382.3 | 0.000    | —           | —       | finish       |  |
|   17 |   +451.5 | 0.000    | —           | —       | finish       |  |
|   18 |   +657.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   19 |   +327.8 | 0.000    | —           | —       | finish       |  |
|   20 |   +342.2 | 0.000    | —           | —       | finish       |  |
|   21 |   +378.2 | 0.000    | —           | —       | finish       |  |
|   22 |   +376.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +343.2 | 0.000    | —           | —       | finish       |  |
|   24 |   +495.9 | 0.000    | —           | —       | finish       |  |
|   25 |   +365.0 | 0.000    | —           | —       | finish       |  |
|   26 |   +469.6 | 0.000    | —           | —       | finish       |  |
|   27 |   +343.3 | 0.000    | —           | —       | finish       |  |
|   28 |   +547.0 | 0.000    | —           | —       | finish       |  |
|   29 |   +595.4 | 0.000    | —           | —       | finish       |  |
|   30 |   +421.8 | 0.000    | —           | —       | finish       |  |
|   31 |   +411.1 | 0.000    | —           | —       | finish       |  |
|   32 |   +423.2 | 0.000    | —           | —       | finish       |  |
|   33 |   +489.6 | 0.000    | —           | —       | finish       |  |
|   34 |   +413.9 | 0.000    | —           | —       | finish       |  |
|   35 |   +448.8 | 0.000    | —           | —       | finish       |  |
|   36 |   +451.6 | 0.000    | —           | —       | finish       |  |
|   37 |   +330.7 | 0.000    | —           | —       | finish       |  |
|   38 |   +493.5 | 0.000    | —           | —       | finish       |  |
|   39 |   +426.0 | 0.000    | —           | —       | finish       |  |
|   40 |   +470.4 | 0.000    | —           | —       | finish       |  |
|   41 |   +422.0 | 0.000    | —           | —       | finish       |  |
|   42 |   +455.0 | 0.000    | —           | —       | finish       |  |
|   43 |   +463.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +436.5 | 0.000    | —           | —       | finish       |  |
|   45 |   +352.4 | 0.000    | —           | —       | finish       |  |
|   46 |   +483.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +439.3 | 0.000    | —           | —       | finish       |  |
|   48 |   +562.7 | 0.000    | —           | —       | finish       |  |
|   49 |   +352.4 | 0.000    | —           | —       | finish       |  |
|   50 |   +396.8 | 0.000    | —           | —       | finish       |  |

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

