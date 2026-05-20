# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.1__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 00:25:25
- **End:** 2026-05-08 01:09:14
- **Total runtime:** 43m 48.6s

| Phase | Duration |
|-------|----------|
| Greedy | 43m 47.6s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1730.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +768.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1256.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1649.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +647.9 | 0.000    | —           | —       | finish       |  |
|    5 |   +881.7 | 0.000    | —           | —       | finish       |  |
|    6 |  +1704.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |  +1165.3 | 0.000    | —           | —       | finish       |  |
|    8 |  +1730.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1171.3 | 0.000    | —           | —       | finish       |  |
|   10 |  +1712.9 | 0.000    | —           | —       | finish       |  |
|   11 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   12 |  +1107.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1625.1 | 0.000    | —           | —       | finish       |  |
|   14 |   +851.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1341.7 | 0.000    | —           | —       | finish       |  |
|   16 |  +1217.7 | 0.000    | —           | —       | finish       |  |
|   17 |  +1258.9 | 0.000    | —           | —       | finish       |  |
|   18 |  +1051.2 | 0.000    | —           | —       | finish       |  |
|   19 |  +1219.8 | 0.000    | —           | —       | finish       |  |
|   20 |  +1254.5 | 0.000    | —           | —       | finish       |  |
|   21 |  +1687.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1131.3 | 0.000    | —           | —       | finish       |  |
|   23 |  +1516.9 | 0.000    | —           | —       | finish       |  |
|   24 |  +1537.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1402.1 | 0.000    | —           | —       | finish       |  |
|   26 |  +1186.3 | 0.000    | —           | —       | finish       |  |
|   27 |  +1068.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1689.1 | 0.000    | —           | —       | finish       |  |
|   29 |  +1308.7 | 0.000    | —           | —       | finish       |  |
|   30 |  +1068.1 | 0.000    | —           | —       | finish       |  |
|   31 |  +1376.4 | 0.000    | —           | —       | finish       |  |
|   32 |  +1556.2 | 0.000    | —           | —       | finish       |  |
|   33 |  +1586.6 | 0.000    | —           | —       | finish       |  |
|   34 |  +1129.5 | 0.000    | —           | —       | finish       |  |
|   35 |  +1331.5 | 0.000    | —           | —       | finish       |  |
|   36 |  +1490.0 | 0.000    | —           | —       | finish       |  |
|   37 |  +1268.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1428.8 | 0.000    | —           | —       | finish       |  |
|   39 |  +1553.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1384.5 | 0.000    | —           | —       | finish       |  |
|   41 |  +1619.4 | 0.000    | —           | —       | finish       |  |
|   42 |  +1309.1 | 0.000    | —           | —       | finish       |  |
|   43 |  +1547.3 | 0.000    | —           | —       | finish       |  |
|   44 |  +1406.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1504.7 | 0.000    | —           | —       | finish       |  |
|   46 |  +1540.1 | 0.000    | —           | —       | finish       |  |
|   47 |  +1360.3 | 0.000    | —           | —       | finish       |  |
|   48 |  +1550.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1269.1 | 0.000    | —           | —       | finish       |  |
|   50 |  +1587.7 | 0.000    | —           | —       | finish       |  |

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

