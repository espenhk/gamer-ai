# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.05__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 02:49:29
- **End:** 2026-05-08 03:39:53
- **Total runtime:** 50m 24.0s

| Phase | Duration |
|-------|----------|
| Greedy | 50m 23.0s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1694.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1663.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +498.8 | 0.000    | —           | —       | finish       |  |
|    3 |  +1006.1 | 0.000    | —           | —       | finish       |  |
|    4 |  +1677.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1694.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1490.5 | 0.000    | —           | —       | finish       |  |
|    7 |   +555.5 | 0.000    | —           | —       | finish       |  |
|    8 |   +940.6 | 0.000    | —           | —       | finish       |  |
|    9 |  +1022.5 | 0.000    | —           | —       | finish       |  |
|   10 |   +581.7 | 0.000    | —           | —       | finish       |  |
|   11 |  +1654.6 | 0.000    | —           | —       | finish       |  |
|   12 |  +1129.9 | 0.000    | —           | —       | finish       |  |
|   13 |  +1407.7 | 0.000    | —           | —       | finish       |  |
|   14 |   +991.3 | 0.000    | —           | —       | finish       |  |
|   15 |  +1348.9 | 0.000    | —           | —       | finish       |  |
|   16 |  +1501.7 | 0.000    | —           | —       | finish       |  |
|   17 |  +1035.7 | 0.000    | —           | —       | finish       |  |
|   18 |   +947.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1259.7 | 0.000    | —           | —       | finish       |  |
|   20 |  +1222.5 | 0.000    | —           | —       | finish       |  |
|   21 |  +1160.3 | 0.000    | —           | —       | finish       |  |
|   22 |  +1558.9 | 0.000    | —           | —       | finish       |  |
|   23 |  +1210.3 | 0.000    | —           | —       | finish       |  |
|   24 |  +1546.5 | 0.000    | —           | —       | finish       |  |
|   25 |  +1282.5 | 0.000    | —           | —       | finish       |  |
|   26 |  +1234.5 | 0.000    | —           | —       | finish       |  |
|   27 |  +1562.5 | 0.000    | —           | —       | finish       |  |
|   28 |  +1186.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1181.3 | 0.000    | —           | —       | finish       |  |
|   30 |  +1210.9 | 0.000    | —           | —       | finish       |  |
|   31 |  +1630.5 | 0.000    | —           | —       | finish       |  |
|   32 |  +1654.9 | 0.000    | —           | —       | finish       |  |
|   33 |  +1126.9 | 0.000    | —           | —       | finish       |  |
|   34 |  +1616.1 | 0.000    | —           | —       | finish       |  |
|   35 |  +1555.7 | 0.000    | —           | —       | finish       |  |
|   36 |  +1273.3 | 0.000    | —           | —       | finish       |  |
|   37 |  +1210.5 | 0.000    | —           | —       | finish       |  |
|   38 |  +1450.5 | 0.000    | —           | —       | finish       |  |
|   39 |   +566.2 | 0.000    | —           | —       | finish       |  |
|   40 |  +1282.8 | 0.000    | —           | —       | finish       |  |
|   41 |   +775.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1551.7 | 0.000    | —           | —       | finish       |  |
|   43 |  +1644.7 | 0.000    | —           | —       | finish       |  |
|   44 |  +1592.1 | 0.000    | —           | —       | finish       |  |
|   45 |  +1490.1 | 0.000    | —           | —       | finish       |  |
|   46 |  +1459.7 | 0.000    | —           | —       | finish       |  |
|   47 |  +1371.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1395.7 | 0.000    | —           | —       | finish       |  |
|   49 |  +1100.1 | 0.000    | —           | —       | finish       |  |
|   50 |   +661.7 | 0.000    | —           | —       | finish       |  |

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

