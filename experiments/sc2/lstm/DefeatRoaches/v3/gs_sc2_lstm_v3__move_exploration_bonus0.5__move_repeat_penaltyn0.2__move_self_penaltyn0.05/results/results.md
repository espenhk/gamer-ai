# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.2__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 23:02:49
- **End:** 2026-05-07 23:43:51
- **Total runtime:** 41m 01.4s

| Phase | Duration |
|-------|----------|
| Greedy | 41m 00.4s |

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
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1710.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1493.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +795.3 | 0.000    | —           | —       | finish       |  |
|    3 |   +903.2 | 0.000    | —           | —       | finish       |  |
|    4 |  +1106.9 | 0.000    | —           | —       | finish       |  |
|    5 |  +1656.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1256.8 | 0.000    | —           | —       | finish       |  |
|    7 |  +1682.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |  +1679.9 | 0.000    | —           | —       | finish       |  |
|    9 |  +1682.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1501.7 | 0.000    | —           | —       | finish       |  |
|   11 |  +1536.5 | 0.000    | —           | —       | finish       |  |
|   12 |  +1639.9 | 0.000    | —           | —       | finish       |  |
|   13 |  +1667.3 | 0.000    | —           | —       | finish       |  |
|   14 |  +1710.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |  +1667.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1682.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1681.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1674.5 | 0.000    | —           | —       | finish       |  |
|   19 |  +1456.1 | 0.000    | —           | —       | finish       |  |
|   20 |   +309.2 | 0.000    | —           | —       | finish       |  |
|   21 |  +1388.3 | 0.000    | —           | —       | finish       |  |
|   22 |  +1382.3 | 0.000    | —           | —       | finish       |  |
|   23 |  +1381.3 | 0.000    | —           | —       | finish       |  |
|   24 |  +1705.3 | 0.000    | —           | —       | finish       |  |
|   25 |  +1633.3 | 0.000    | —           | —       | finish       |  |
|   26 |  +1682.3 | 0.000    | —           | —       | finish       |  |
|   27 |  +1693.3 | 0.000    | —           | —       | finish       |  |
|   28 |  +1682.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1674.9 | 0.000    | —           | —       | finish       |  |
|   30 |  +1666.9 | 0.000    | —           | —       | finish       |  |
|   31 |  +1576.3 | 0.000    | —           | —       | finish       |  |
|   32 |  +1453.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1466.1 | 0.000    | —           | —       | finish       |  |
|   34 |  +1353.3 | 0.000    | —           | —       | finish       |  |
|   35 |  +1312.3 | 0.000    | —           | —       | finish       |  |
|   36 |  +1688.9 | 0.000    | —           | —       | finish       |  |
|   37 |  +1682.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1696.9 | 0.000    | —           | —       | finish       |  |
|   39 |  +1684.9 | 0.000    | —           | —       | finish       |  |
|   40 |  +1692.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1660.3 | 0.000    | —           | —       | finish       |  |
|   42 |  +1682.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1666.9 | 0.000    | —           | —       | finish       |  |
|   44 |  +1476.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1420.7 | 0.000    | —           | —       | finish       |  |
|   46 |  +1335.3 | 0.000    | —           | —       | finish       |  |
|   47 |  +1708.3 | 0.000    | —           | —       | finish       |  |
|   48 |  +1476.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1692.9 | 0.000    | —           | —       | finish       |  |
|   50 |  +1704.5 | 0.000    | —           | —       | finish       |  |

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

