# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.2__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 05:14:37
- **End:** 2026-05-08 06:01:02
- **Total runtime:** 46m 24.8s

| Phase | Duration |
|-------|----------|
| Greedy | 46m 23.8s |

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
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1733.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +745.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1703.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1648.9 | 0.000    | —           | —       | finish       |  |
|    4 |  +1351.8 | 0.000    | —           | —       | finish       |  |
|    5 |   +794.4 | 0.000    | —           | —       | finish       |  |
|    6 |  +1712.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |  +1729.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |  +1492.9 | 0.000    | —           | —       | finish       |  |
|    9 |  +1706.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1364.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1254.7 | 0.000    | —           | —       | finish       |  |
|   12 |  +1509.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1560.7 | 0.000    | —           | —       | finish       |  |
|   14 |  +1373.3 | 0.000    | —           | —       | finish       |  |
|   15 |  +1667.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1643.3 | 0.000    | —           | —       | finish       |  |
|   17 |  +1556.1 | 0.000    | —           | —       | finish       |  |
|   18 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   19 |  +1724.9 | 0.000    | —           | —       | finish       |  |
|   20 |  +1733.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   21 |  +1471.5 | 0.000    | —           | —       | finish       |  |
|   22 |  +1612.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1681.3 | 0.000    | —           | —       | finish       |  |
|   24 |  +1651.3 | 0.000    | —           | —       | finish       |  |
|   25 |  +1667.3 | 0.000    | —           | —       | finish       |  |
|   26 |  +1600.9 | 0.000    | —           | —       | finish       |  |
|   27 |  +1547.3 | 0.000    | —           | —       | finish       |  |
|   28 |  +1661.9 | 0.000    | —           | —       | finish       |  |
|   29 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   30 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   31 |  +1720.9 | 0.000    | —           | —       | finish       |  |
|   32 |  +1537.3 | 0.000    | —           | —       | finish       |  |
|   33 |  +1675.3 | 0.000    | —           | —       | finish       |  |
|   34 |  +1671.3 | 0.000    | —           | —       | finish       |  |
|   35 |  +1675.3 | 0.000    | —           | —       | finish       |  |
|   36 |  +1663.3 | 0.000    | —           | —       | finish       |  |
|   37 |  +1456.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1484.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1708.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   42 |  +1483.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1667.3 | 0.000    | —           | —       | finish       |  |
|   44 |  +1674.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1651.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1672.9 | 0.000    | —           | —       | finish       |  |
|   47 |  +1633.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1702.5 | 0.000    | —           | —       | finish       |  |
|   49 |  +1682.5 | 0.000    | —           | —       | finish       |  |
|   50 |  +1716.1 | 0.000    | —           | —       | finish       |  |

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

