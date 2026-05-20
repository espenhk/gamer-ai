# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.2__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 23:44:03
- **End:** 2026-05-08 00:25:12
- **Total runtime:** 41m 09.0s

| Phase | Duration |
|-------|----------|
| Greedy | 41m 08.0s |

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
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1692.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1558.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1649.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +593.1 | 0.000    | —           | —       | finish       |  |
|    4 |  +1015.6 | 0.000    | —           | —       | finish       |  |
|    5 |  +1259.9 | 0.000    | —           | —       | finish       |  |
|    6 |  +1584.8 | 0.000    | —           | —       | finish       |  |
|    7 |  +1387.0 | 0.000    | —           | —       | finish       |  |
|    8 |  +1575.8 | 0.000    | —           | —       | finish       |  |
|    9 |  +1663.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |   +817.7 | 0.000    | —           | —       | finish       |  |
|   11 |  +1157.7 | 0.000    | —           | —       | finish       |  |
|   12 |   +408.6 | 0.000    | —           | —       | finish       |  |
|   13 |  +1391.4 | 0.000    | —           | —       | finish       |  |
|   14 |  +1692.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |  +1666.2 | 0.000    | —           | —       | finish       |  |
|   16 |  +1681.2 | 0.000    | —           | —       | finish       |  |
|   17 |  +1638.2 | 0.000    | —           | —       | finish       |  |
|   18 |  +1650.1 | 0.000    | —           | —       | finish       |  |
|   19 |  +1665.4 | 0.000    | —           | —       | finish       |  |
|   20 |  +1094.4 | 0.000    | —           | —       | finish       |  |
|   21 |  +1621.8 | 0.000    | —           | —       | finish       |  |
|   22 |  +1303.3 | 0.000    | —           | —       | finish       |  |
|   23 |  +1271.4 | 0.000    | —           | —       | finish       |  |
|   24 |  +1679.4 | 0.000    | —           | —       | finish       |  |
|   25 |  +1516.7 | 0.000    | —           | —       | finish       |  |
|   26 |  +1657.2 | 0.000    | —           | —       | finish       |  |
|   27 |  +1673.2 | 0.000    | —           | —       | finish       |  |
|   28 |  +1601.2 | 0.000    | —           | —       | finish       |  |
|   29 |  +1643.0 | 0.000    | —           | —       | finish       |  |
|   30 |  +1633.0 | 0.000    | —           | —       | finish       |  |
|   31 |  +1089.2 | 0.000    | —           | —       | finish       |  |
|   32 |  +1115.0 | 0.000    | —           | —       | finish       |  |
|   33 |  +1231.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1215.0 | 0.000    | —           | —       | finish       |  |
|   35 |  +1614.9 | 0.000    | —           | —       | finish       |  |
|   36 |  +1677.7 | 0.000    | —           | —       | finish       |  |
|   37 |  +1673.2 | 0.000    | —           | —       | finish       |  |
|   38 |  +1619.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1667.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1657.7 | 0.000    | —           | —       | finish       |  |
|   41 |  +1671.3 | 0.000    | —           | —       | finish       |  |
|   42 |  +1681.0 | 0.000    | —           | —       | finish       |  |
|   43 |  +1470.8 | 0.000    | —           | —       | finish       |  |
|   44 |  +1305.0 | 0.000    | —           | —       | finish       |  |
|   45 |  +1571.2 | 0.000    | —           | —       | finish       |  |
|   46 |  +1016.0 | 0.000    | —           | —       | finish       |  |
|   47 |  +1245.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1327.1 | 0.000    | —           | —       | finish       |  |
|   49 |  +1103.4 | 0.000    | —           | —       | finish       |  |
|   50 |  +1583.4 | 0.000    | —           | —       | finish       |  |

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

