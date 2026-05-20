# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.1__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 02:04:14
- **End:** 2026-05-08 02:49:16
- **Total runtime:** 45m 02.1s

| Phase | Duration |
|-------|----------|
| Greedy | 45m 01.1s |

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
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1734.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1330.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1087.3 | 0.000    | —           | —       | finish       |  |
|    3 |   +835.3 | 0.000    | —           | —       | finish       |  |
|    4 |   +386.9 | 0.000    | —           | —       | finish       |  |
|    5 |  +1710.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1726.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |  +1734.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |  +1718.5 | 0.000    | —           | —       | finish       |  |
|    9 |  +1710.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   11 |  +1658.6 | 0.000    | —           | —       | finish       |  |
|   12 |  +1202.6 | 0.000    | —           | —       | finish       |  |
|   13 |  +1229.8 | 0.000    | —           | —       | finish       |  |
|   14 |  +1697.8 | 0.000    | —           | —       | finish       |  |
|   15 |  +1707.0 | 0.000    | —           | —       | finish       |  |
|   16 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   17 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   19 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   20 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   21 |  +1386.6 | 0.000    | —           | —       | finish       |  |
|   22 |  +1138.3 | 0.000    | —           | —       | finish       |  |
|   23 |  +1385.7 | 0.000    | —           | —       | finish       |  |
|   24 |   +866.0 | 0.000    | —           | —       | finish       |  |
|   25 |  +1682.0 | 0.000    | —           | —       | finish       |  |
|   26 |  +1723.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   28 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   30 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   31 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   32 |  +1476.4 | 0.000    | —           | —       | finish       |  |
|   33 |  +1209.4 | 0.000    | —           | —       | finish       |  |
|   34 |  +1651.1 | 0.000    | —           | —       | finish       |  |
|   35 |   +940.9 | 0.000    | —           | —       | finish       |  |
|   36 |  +1697.2 | 0.000    | —           | —       | finish       |  |
|   37 |  +1722.8 | 0.000    | —           | —       | finish       |  |
|   38 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   39 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   40 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   41 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   42 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   43 |   +999.4 | 0.000    | —           | —       | finish       |  |
|   44 |  +1305.7 | 0.000    | —           | —       | finish       |  |
|   45 |  +1218.5 | 0.000    | —           | —       | finish       |  |
|   46 |  +1376.7 | 0.000    | —           | —       | finish       |  |
|   47 |  +1666.2 | 0.000    | —           | —       | finish       |  |
|   48 |  +1722.0 | 0.000    | —           | —       | finish       |  |
|   49 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   50 |  +1734.5 | 0.000    | —           | —       | finish       |  |

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

