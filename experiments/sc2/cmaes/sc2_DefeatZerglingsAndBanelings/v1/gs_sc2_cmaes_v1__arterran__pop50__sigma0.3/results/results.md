# Experiment: gs_sc2_cmaes_v1__arterran__pop50__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-10 13:38:14
- **End:** 2026-05-11 01:59:04
- **Total runtime:** 12h 20m 50.0s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 20m 49.0s |

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

Best reward: **+730.4**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +470.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +191.4 | 0.000    | —           | —       | finish       |  |
|    3 |   +425.2 | 0.000    | —           | —       | finish       |  |
|    4 |   +211.7 | 0.000    | —           | —       | finish       |  |
|    5 |   +463.1 | 0.000    | —           | —       | finish       |  |
|    6 |   +517.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +365.6 | 0.000    | —           | —       | finish       |  |
|    8 |   +585.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +280.5 | 0.000    | —           | —       | finish       |  |
|   10 |   +348.9 | 0.000    | —           | —       | finish       |  |
|   11 |   +374.3 | 0.000    | —           | —       | finish       |  |
|   12 |   +390.8 | 0.000    | —           | —       | finish       |  |
|   13 |   +379.4 | 0.000    | —           | —       | finish       |  |
|   14 |   +455.5 | 0.000    | —           | —       | finish       |  |
|   15 |   +367.0 | 0.000    | —           | —       | finish       |  |
|   16 |   +385.1 | 0.000    | —           | —       | finish       |  |
|   17 |   +346.3 | 0.000    | —           | —       | finish       |  |
|   18 |   +372.2 | 0.000    | —           | —       | finish       |  |
|   19 |   +403.1 | 0.000    | —           | —       | finish       |  |
|   20 |   +503.5 | 0.000    | —           | —       | finish       |  |
|   21 |   +525.6 | 0.000    | —           | —       | finish       |  |
|   22 |   +421.0 | 0.000    | —           | —       | finish       |  |
|   23 |   +487.3 | 0.000    | —           | —       | finish       |  |
|   24 |   +384.5 | 0.000    | —           | —       | finish       |  |
|   25 |   +464.4 | 0.000    | —           | —       | finish       |  |
|   26 |   +514.8 | 0.000    | —           | —       | finish       |  |
|   27 |   +429.7 | 0.000    | —           | —       | finish       |  |
|   28 |   +425.9 | 0.000    | —           | —       | finish       |  |
|   29 |   +522.7 | 0.000    | —           | —       | finish       |  |
|   30 |   +431.7 | 0.000    | —           | —       | finish       |  |
|   31 |   +461.7 | 0.000    | —           | —       | finish       |  |
|   32 |   +465.3 | 0.000    | —           | —       | finish       |  |
|   33 |   +554.0 | 0.000    | —           | —       | finish       |  |
|   34 |   +647.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   35 |   +644.3 | 0.000    | —           | —       | finish       |  |
|   36 |   +490.3 | 0.000    | —           | —       | finish       |  |
|   37 |   +483.0 | 0.000    | —           | —       | finish       |  |
|   38 |   +730.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   39 |   +531.9 | 0.000    | —           | —       | finish       |  |
|   40 |   +694.8 | 0.000    | —           | —       | finish       |  |
|   41 |   +629.6 | 0.000    | —           | —       | finish       |  |
|   42 |   +559.4 | 0.000    | —           | —       | finish       |  |
|   43 |   +620.1 | 0.000    | —           | —       | finish       |  |
|   44 |   +612.5 | 0.000    | —           | —       | finish       |  |
|   45 |   +633.8 | 0.000    | —           | —       | finish       |  |
|   46 |   +659.6 | 0.000    | —           | —       | finish       |  |
|   47 |   +674.7 | 0.000    | —           | —       | finish       |  |
|   48 |   +614.2 | 0.000    | —           | —       | finish       |  |
|   49 |   +649.0 | 0.000    | —           | —       | finish       |  |
|   50 |   +641.5 | 0.000    | —           | —       | finish       |  |

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

