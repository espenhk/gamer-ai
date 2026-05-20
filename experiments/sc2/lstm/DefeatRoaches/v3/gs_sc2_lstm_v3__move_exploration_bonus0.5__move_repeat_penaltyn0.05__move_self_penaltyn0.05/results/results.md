# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.05__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 03:40:06
- **End:** 2026-05-08 04:26:55
- **Total runtime:** 46m 49.4s

| Phase | Duration |
|-------|----------|
| Greedy | 46m 48.4s |

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
| move_exploration_bonus | 0.5 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1741.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1422.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1700.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +991.3 | 0.000    | —           | —       | finish       |  |
|    4 |  +1233.9 | 0.000    | —           | —       | finish       |  |
|    5 |  +1732.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1395.5 | 0.000    | —           | —       | finish       |  |
|    7 |  +1104.5 | 0.000    | —           | —       | finish       |  |
|    8 |  +1678.5 | 0.000    | —           | —       | finish       |  |
|    9 |   +998.5 | 0.000    | —           | —       | finish       |  |
|   10 |   +596.0 | 0.000    | —           | —       | finish       |  |
|   11 |  +1291.9 | 0.000    | —           | —       | finish       |  |
|   12 |  +1701.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1599.3 | 0.000    | —           | —       | finish       |  |
|   14 |  +1621.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1639.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1446.7 | 0.000    | —           | —       | finish       |  |
|   17 |  +1702.1 | 0.000    | —           | —       | finish       |  |
|   18 |  +1046.2 | 0.000    | —           | —       | finish       |  |
|   19 |  +1440.5 | 0.000    | —           | —       | finish       |  |
|   20 |  +1542.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1708.9 | 0.000    | —           | —       | finish       |  |
|   22 |  +1733.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   23 |  +1741.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |  +1711.3 | 0.000    | —           | —       | finish       |  |
|   25 |  +1694.9 | 0.000    | —           | —       | finish       |  |
|   26 |  +1694.9 | 0.000    | —           | —       | finish       |  |
|   27 |  +1688.5 | 0.000    | —           | —       | finish       |  |
|   28 |  +1264.1 | 0.000    | —           | —       | finish       |  |
|   29 |  +1461.5 | 0.000    | —           | —       | finish       |  |
|   30 |  +1463.3 | 0.000    | —           | —       | finish       |  |
|   31 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   32 |  +1733.3 | 0.000    | —           | —       | finish       |  |
|   33 |  +1718.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1719.3 | 0.000    | —           | —       | finish       |  |
|   35 |  +1706.4 | 0.000    | —           | —       | finish       |  |
|   36 |  +1718.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1707.7 | 0.000    | —           | —       | finish       |  |
|   38 |  +1576.3 | 0.000    | —           | —       | finish       |  |
|   39 |  +1663.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1265.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1167.2 | 0.000    | —           | —       | finish       |  |
|   42 |  +1699.1 | 0.000    | —           | —       | finish       |  |
|   43 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1510.9 | 0.000    | —           | —       | finish       |  |
|   46 |  +1710.1 | 0.000    | —           | —       | finish       |  |
|   47 |  +1708.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1257.1 | 0.000    | —           | —       | finish       |  |
|   49 |  +1687.7 | 0.000    | —           | —       | finish       |  |
|   50 |  +1696.5 | 0.000    | —           | —       | finish       |  |

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

