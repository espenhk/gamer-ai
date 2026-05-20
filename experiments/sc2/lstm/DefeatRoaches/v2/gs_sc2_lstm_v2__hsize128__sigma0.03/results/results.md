# Experiment: gs_sc2_lstm_v2__hsize128__sigma0.03

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 20:09:10
- **End:** 2026-05-07 20:44:31
- **Total runtime:** 35m 21.2s

| Phase | Duration |
|-------|----------|
| Greedy | 35m 20.2s |

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
| initial_sigma | 0.03 |
| policy_params | {'population_size': 20, 'hidden_size': 128, 'initial_sigma': 0.03} |

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
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1723.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1446.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1675.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1052.5 | 0.000    | —           | —       | finish       |  |
|    4 |  +1711.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1716.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1588.3 | 0.000    | —           | —       | finish       |  |
|    7 |  +1675.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1701.0 | 0.000    | —           | —       | finish       |  |
|    9 |  +1034.6 | 0.000    | —           | —       | finish       |  |
|   10 |  +1686.9 | 0.000    | —           | —       | finish       |  |
|   11 |  +1714.0 | 0.000    | —           | —       | finish       |  |
|   12 |  +1680.6 | 0.000    | —           | —       | finish       |  |
|   13 |  +1515.1 | 0.000    | —           | —       | finish       |  |
|   14 |  +1686.2 | 0.000    | —           | —       | finish       |  |
|   15 |  +1708.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1710.2 | 0.000    | —           | —       | finish       |  |
|   17 |  +1715.8 | 0.000    | —           | —       | finish       |  |
|   18 |  +1723.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   19 |  +1712.3 | 0.000    | —           | —       | finish       |  |
|   20 |  +1708.4 | 0.000    | —           | —       | finish       |  |
|   21 |  +1708.7 | 0.000    | —           | —       | finish       |  |
|   22 |  +1689.2 | 0.000    | —           | —       | finish       |  |
|   23 |  +1680.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1670.3 | 0.000    | —           | —       | finish       |  |
|   25 |  +1680.0 | 0.000    | —           | —       | finish       |  |
|   26 |  +1683.5 | 0.000    | —           | —       | finish       |  |
|   27 |  +1660.0 | 0.000    | —           | —       | finish       |  |
|   28 |  +1714.0 | 0.000    | —           | —       | finish       |  |
|   29 |  +1708.5 | 0.000    | —           | —       | finish       |  |
|   30 |  +1715.1 | 0.000    | —           | —       | finish       |  |
|   31 |  +1723.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1723.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1671.3 | 0.000    | —           | —       | finish       |  |
|   34 |  +1692.6 | 0.000    | —           | —       | finish       |  |
|   35 |  +1689.5 | 0.000    | —           | —       | finish       |  |
|   36 |  +1696.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1671.8 | 0.000    | —           | —       | finish       |  |
|   38 |  +1652.4 | 0.000    | —           | —       | finish       |  |
|   39 |  +1184.8 | 0.000    | —           | —       | finish       |  |
|   40 |  +1672.8 | 0.000    | —           | —       | finish       |  |
|   41 |  +1672.8 | 0.000    | —           | —       | finish       |  |
|   42 |  +1702.8 | 0.000    | —           | —       | finish       |  |
|   43 |  +1717.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1715.1 | 0.000    | —           | —       | finish       |  |
|   45 |  +1713.5 | 0.000    | —           | —       | finish       |  |
|   46 |  +1711.4 | 0.000    | —           | —       | finish       |  |
|   47 |  +1677.1 | 0.000    | —           | —       | finish       |  |
|   48 |  +1708.6 | 0.000    | —           | —       | finish       |  |
|   49 |  +1405.9 | 0.000    | —           | —       | finish       |  |
|   50 |  +1495.9 | 0.000    | —           | —       | finish       |  |

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

