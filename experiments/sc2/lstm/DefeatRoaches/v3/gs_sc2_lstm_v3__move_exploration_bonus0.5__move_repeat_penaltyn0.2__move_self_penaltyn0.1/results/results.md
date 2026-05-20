# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.2__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 22:24:54
- **End:** 2026-05-07 23:02:36
- **Total runtime:** 37m 42.1s

| Phase | Duration |
|-------|----------|
| Greedy | 37m 36.2s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1702.1**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1702.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +994.3 | 0.000    | —           | —       | finish       |  |
|    3 |  +1291.7 | 0.000    | —           | —       | finish       |  |
|    4 |  +1106.0 | 0.000    | —           | —       | finish       |  |
|    5 |   +830.5 | 0.000    | —           | —       | finish       |  |
|    6 |   +795.4 | 0.000    | —           | —       | finish       |  |
|    7 |  +1548.6 | 0.000    | —           | —       | finish       |  |
|    8 |  +1133.9 | 0.000    | —           | —       | finish       |  |
|    9 |  +1613.7 | 0.000    | —           | —       | finish       |  |
|   10 |  +1620.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1676.9 | 0.000    | —           | —       | finish       |  |
|   12 |  +1664.1 | 0.000    | —           | —       | finish       |  |
|   13 |  +1666.3 | 0.000    | —           | —       | finish       |  |
|   14 |  +1146.5 | 0.000    | —           | —       | finish       |  |
|   15 |  +1685.7 | 0.000    | —           | —       | finish       |  |
|   16 |  +1696.1 | 0.000    | —           | —       | finish       |  |
|   17 |  +1676.9 | 0.000    | —           | —       | finish       |  |
|   18 |  +1573.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1639.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +668.9 | 0.000    | —           | —       | finish       |  |
|   21 |   +586.9 | 0.000    | —           | —       | finish       |  |
|   22 |  +1662.5 | 0.000    | —           | —       | finish       |  |
|   23 |  +1673.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1672.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1400.9 | 0.000    | —           | —       | finish       |  |
|   26 |  +1188.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1475.7 | 0.000    | —           | —       | finish       |  |
|   28 |  +1588.9 | 0.000    | —           | —       | finish       |  |
|   29 |  +1668.9 | 0.000    | —           | —       | finish       |  |
|   30 |  +1651.3 | 0.000    | —           | —       | finish       |  |
|   31 |  +1660.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1551.3 | 0.000    | —           | —       | finish       |  |
|   33 |  +1642.7 | 0.000    | —           | —       | finish       |  |
|   34 |  +1646.1 | 0.000    | —           | —       | finish       |  |
|   35 |  +1447.3 | 0.000    | —           | —       | finish       |  |
|   36 |   +722.9 | 0.000    | —           | —       | finish       |  |
|   37 |  +1663.3 | 0.000    | —           | —       | finish       |  |
|   38 |  +1152.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1675.3 | 0.000    | —           | —       | finish       |  |
|   40 |  +1683.9 | 0.000    | —           | —       | finish       |  |
|   41 |  +1293.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1675.3 | 0.000    | —           | —       | finish       |  |
|   43 |  +1666.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1656.9 | 0.000    | —           | —       | finish       |  |
|   45 |  +1656.9 | 0.000    | —           | —       | finish       |  |
|   46 |  +1644.1 | 0.000    | —           | —       | finish       |  |
|   47 |  +1289.1 | 0.000    | —           | —       | finish       |  |
|   48 |   +960.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1235.3 | 0.000    | —           | —       | finish       |  |
|   50 |   +807.2 | 0.000    | —           | —       | finish       |  |

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

