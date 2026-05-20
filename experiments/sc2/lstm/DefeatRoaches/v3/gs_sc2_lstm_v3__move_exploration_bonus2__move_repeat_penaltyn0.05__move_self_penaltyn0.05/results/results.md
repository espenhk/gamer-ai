# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.05__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 16:43:10
- **End:** 2026-05-08 17:32:28
- **Total runtime:** 49m 18.4s

| Phase | Duration |
|-------|----------|
| Greedy | 49m 17.4s |

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
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1802.2**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +604.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +672.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1308.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |  +1606.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1637.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1631.5 | 0.000    | —           | —       | finish       |  |
|    7 |  +1002.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1679.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1497.9 | 0.000    | —           | —       | finish       |  |
|   10 |  +1511.3 | 0.000    | —           | —       | finish       |  |
|   11 |  +1734.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   12 |  +1479.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   14 |  +1635.7 | 0.000    | —           | —       | finish       |  |
|   15 |  +1718.9 | 0.000    | —           | —       | finish       |  |
|   16 |  +1694.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1726.9 | 0.000    | —           | —       | finish       |  |
|   18 |  +1734.9 | 0.000    | —           | —       | finish       |  |
|   19 |  +1253.8 | 0.000    | —           | —       | finish       |  |
|   20 |  +1686.3 | 0.000    | —           | —       | finish       |  |
|   21 |  +1629.3 | 0.000    | —           | —       | finish       |  |
|   22 |  +1706.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1734.9 | 0.000    | —           | —       | finish       |  |
|   24 |  +1734.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   26 |  +1683.7 | 0.000    | —           | —       | finish       |  |
|   27 |  +1692.5 | 0.000    | —           | —       | finish       |  |
|   28 |   +960.0 | 0.000    | —           | —       | finish       |  |
|   29 |  +1379.7 | 0.000    | —           | —       | finish       |  |
|   30 |  +1657.3 | 0.000    | —           | —       | finish       |  |
|   31 |  +1734.9 | 0.000    | —           | —       | finish       |  |
|   32 |  +1716.5 | 0.000    | —           | —       | finish       |  |
|   33 |  +1685.3 | 0.000    | —           | —       | finish       |  |
|   34 |  +1686.9 | 0.000    | —           | —       | finish       |  |
|   35 |  +1652.5 | 0.000    | —           | —       | finish       |  |
|   36 |  +1716.5 | 0.000    | —           | —       | finish       |  |
|   37 |  +1317.7 | 0.000    | —           | —       | finish       |  |
|   38 |  +1366.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1344.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1802.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   41 |  +1790.9 | 0.000    | —           | —       | finish       |  |
|   42 |  +1790.9 | 0.000    | —           | —       | finish       |  |
|   43 |  +1710.1 | 0.000    | —           | —       | finish       |  |
|   44 |  +1774.9 | 0.000    | —           | —       | finish       |  |
|   45 |  +1750.5 | 0.000    | —           | —       | finish       |  |
|   46 |  +1422.5 | 0.000    | —           | —       | finish       |  |
|   47 |  +1774.9 | 0.000    | —           | —       | finish       |  |
|   48 |  +1707.7 | 0.000    | —           | —       | finish       |  |
|   49 |  +1714.1 | 0.000    | —           | —       | finish       |  |
|   50 |  +1774.9 | 0.000    | —           | —       | finish       |  |

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

