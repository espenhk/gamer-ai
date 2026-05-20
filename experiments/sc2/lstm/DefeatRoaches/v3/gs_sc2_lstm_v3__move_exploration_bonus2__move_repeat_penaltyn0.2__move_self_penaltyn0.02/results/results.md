# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.2__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 13:09:43
- **End:** 2026-05-08 13:52:34
- **Total runtime:** 42m 50.9s

| Phase | Duration |
|-------|----------|
| Greedy | 42m 49.9s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatRoaches |
| map_name | DefeatRoaches |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | random |
| n_sims | 50 |
| policy_type | lstm |
| obs_spec_preset | rich |
| enable_belief | True |
| hidden_size | 128 |
| initial_sigma | 0.1 |
| policy_params | {'population_size': 20, 'hidden_size': 128, 'initial_sigma': 0.1} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 1.0 |
| win_bonus | 20.0 |
| loss_penalty | 0.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 1.0 |
| move_exploration_bonus | 2.0 |
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1733.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +558.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +546.5 | 0.000    | —           | —       | finish       |  |
|    3 |   +584.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +856.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1107.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |   +629.2 | 0.000    | —           | —       | finish       |  |
|    7 |   +903.3 | 0.000    | —           | —       | finish       |  |
|    8 |  +1723.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1289.8 | 0.000    | —           | —       | finish       |  |
|   10 |  +1687.3 | 0.000    | —           | —       | finish       |  |
|   11 |   +727.0 | 0.000    | —           | —       | finish       |  |
|   12 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   14 |  +1733.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   16 |  +1613.6 | 0.000    | —           | —       | finish       |  |
|   17 |  +1703.0 | 0.000    | —           | —       | finish       |  |
|   18 |  +1459.9 | 0.000    | —           | —       | finish       |  |
|   19 |  +1682.3 | 0.000    | —           | —       | finish       |  |
|   20 |  +1262.6 | 0.000    | —           | —       | finish       |  |
|   21 |   +963.4 | 0.000    | —           | —       | finish       |  |
|   22 |   +783.5 | 0.000    | —           | —       | finish       |  |
|   23 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1685.2 | 0.000    | —           | —       | finish       |  |
|   25 |  +1214.2 | 0.000    | —           | —       | finish       |  |
|   26 |  +1225.9 | 0.000    | —           | —       | finish       |  |
|   27 |  +1164.4 | 0.000    | —           | —       | finish       |  |
|   28 |   +988.2 | 0.000    | —           | —       | finish       |  |
|   29 |  +1595.0 | 0.000    | —           | —       | finish       |  |
|   30 |  +1332.4 | 0.000    | —           | —       | finish       |  |
|   31 |  +1299.6 | 0.000    | —           | —       | finish       |  |
|   32 |  +1220.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +702.2 | 0.000    | —           | —       | finish       |  |
|   34 |   +885.3 | 0.000    | —           | —       | finish       |  |
|   35 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   36 |  +1504.3 | 0.000    | —           | —       | finish       |  |
|   37 |  +1442.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1110.1 | 0.000    | —           | —       | finish       |  |
|   39 |   +967.0 | 0.000    | —           | —       | finish       |  |
|   40 |  +1244.2 | 0.000    | —           | —       | finish       |  |
|   41 |  +1633.2 | 0.000    | —           | —       | finish       |  |
|   42 |  +1361.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1648.4 | 0.000    | —           | —       | finish       |  |
|   44 |  +1501.8 | 0.000    | —           | —       | finish       |  |
|   45 |   +857.2 | 0.000    | —           | —       | finish       |  |
|   46 |   +763.0 | 0.000    | —           | —       | finish       |  |
|   47 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1696.6 | 0.000    | —           | —       | finish       |  |
|   49 |  +1198.2 | 0.000    | —           | —       | finish       |  |
|   50 |  +1335.5 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Reward trajectory](reward_trajectory.png)

