# Experiment: gs_sc2_cmaes_v1__arprotoss__pop50__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-16 13:25:54
- **End:** 2026-05-17 01:58:55
- **Total runtime:** 12h 33m 00.4s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 32m 59.4s |

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

Best reward: **+744.4**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +395.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +381.2 | 0.000    | —           | —       | finish       |  |
|    3 |   +421.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +517.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +380.1 | 0.000    | —           | —       | finish       |  |
|    6 |   +410.9 | 0.000    | —           | —       | finish       |  |
|    7 |   +329.8 | 0.000    | —           | —       | finish       |  |
|    8 |   +626.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +425.5 | 0.000    | —           | —       | finish       |  |
|   10 |   +611.1 | 0.000    | —           | —       | finish       |  |
|   11 |   +373.3 | 0.000    | —           | —       | finish       |  |
|   12 |   +480.7 | 0.000    | —           | —       | finish       |  |
|   13 |   +483.2 | 0.000    | —           | —       | finish       |  |
|   14 |   +486.2 | 0.000    | —           | —       | finish       |  |
|   15 |   +373.0 | 0.000    | —           | —       | finish       |  |
|   16 |   +374.7 | 0.000    | —           | —       | finish       |  |
|   17 |   +502.7 | 0.000    | —           | —       | finish       |  |
|   18 |   +514.4 | 0.000    | —           | —       | finish       |  |
|   19 |   +411.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +460.8 | 0.000    | —           | —       | finish       |  |
|   21 |   +452.6 | 0.000    | —           | —       | finish       |  |
|   22 |   +375.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +456.9 | 0.000    | —           | —       | finish       |  |
|   24 |   +444.7 | 0.000    | —           | —       | finish       |  |
|   25 |   +496.5 | 0.000    | —           | —       | finish       |  |
|   26 |   +497.6 | 0.000    | —           | —       | finish       |  |
|   27 |   +442.1 | 0.000    | —           | —       | finish       |  |
|   28 |   +453.8 | 0.000    | —           | —       | finish       |  |
|   29 |   +470.2 | 0.000    | —           | —       | finish       |  |
|   30 |   +581.1 | 0.000    | —           | —       | finish       |  |
|   31 |   +427.5 | 0.000    | —           | —       | finish       |  |
|   32 |   +463.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +530.2 | 0.000    | —           | —       | finish       |  |
|   34 |   +489.9 | 0.000    | —           | —       | finish       |  |
|   35 |   +590.6 | 0.000    | —           | —       | finish       |  |
|   36 |   +578.2 | 0.000    | —           | —       | finish       |  |
|   37 |   +528.7 | 0.000    | —           | —       | finish       |  |
|   38 |   +547.0 | 0.000    | —           | —       | finish       |  |
|   39 |   +545.0 | 0.000    | —           | —       | finish       |  |
|   40 |   +585.2 | 0.000    | —           | —       | finish       |  |
|   41 |   +577.2 | 0.000    | —           | —       | finish       |  |
|   42 |   +731.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   43 |   +706.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +731.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   45 |   +575.7 | 0.000    | —           | —       | finish       |  |
|   46 |   +519.3 | 0.000    | —           | —       | finish       |  |
|   47 |   +714.1 | 0.000    | —           | —       | finish       |  |
|   48 |   +744.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   49 |   +601.2 | 0.000    | —           | —       | finish       |  |
|   50 |   +617.4 | 0.000    | —           | —       | finish       |  |

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

