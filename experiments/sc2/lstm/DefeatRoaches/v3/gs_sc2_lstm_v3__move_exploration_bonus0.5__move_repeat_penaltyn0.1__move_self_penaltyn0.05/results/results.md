# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.1__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 01:09:26
- **End:** 2026-05-08 02:04:01
- **Total runtime:** 54m 35.1s

| Phase | Duration |
|-------|----------|
| Greedy | 54m 34.0s |

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
| move_repeat_penalty | -0.1 |
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1734.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1251.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +607.5 | 0.000    | —           | —       | finish       |  |
|    3 |   +899.3 | 0.000    | —           | —       | finish       |  |
|    4 |  +1529.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1023.8 | 0.000    | —           | —       | finish       |  |
|    6 |  +1052.3 | 0.000    | —           | —       | finish       |  |
|    7 |  +1095.7 | 0.000    | —           | —       | finish       |  |
|    8 |  +1708.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1702.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1698.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1230.5 | 0.000    | —           | —       | finish       |  |
|   12 |  +1668.1 | 0.000    | —           | —       | finish       |  |
|   13 |  +1469.3 | 0.000    | —           | —       | finish       |  |
|   14 |  +1637.5 | 0.000    | —           | —       | finish       |  |
|   15 |  +1518.1 | 0.000    | —           | —       | finish       |  |
|   16 |  +1734.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   17 |  +1731.3 | 0.000    | —           | —       | finish       |  |
|   18 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1239.3 | 0.000    | —           | —       | finish       |  |
|   20 |  +1362.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1661.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1615.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1699.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1722.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1683.3 | 0.000    | —           | —       | finish       |  |
|   26 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   27 |  +1708.5 | 0.000    | —           | —       | finish       |  |
|   28 |  +1519.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1646.3 | 0.000    | —           | —       | finish       |  |
|   30 |  +1694.5 | 0.000    | —           | —       | finish       |  |
|   31 |  +1702.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1710.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1708.9 | 0.000    | —           | —       | finish       |  |
|   34 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   35 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   36 |  +1677.3 | 0.000    | —           | —       | finish       |  |
|   37 |  +1246.7 | 0.000    | —           | —       | finish       |  |
|   38 |  +1319.3 | 0.000    | —           | —       | finish       |  |
|   39 |  +1699.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1708.9 | 0.000    | —           | —       | finish       |  |
|   41 |  +1696.5 | 0.000    | —           | —       | finish       |  |
|   42 |  +1718.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1702.9 | 0.000    | —           | —       | finish       |  |
|   44 |  +1712.9 | 0.000    | —           | —       | finish       |  |
|   45 |  +1389.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1531.9 | 0.000    | —           | —       | finish       |  |
|   47 |  +1706.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1707.7 | 0.000    | —           | —       | finish       |  |
|   49 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   50 |  +1729.3 | 0.000    | —           | —       | finish       |  |

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

