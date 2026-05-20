# Experiment: gs_sc2_lstm_v2__hsize32__sigma0.03

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 18:07:04
- **End:** 2026-05-07 18:38:05
- **Total runtime:** 31m 01.2s

| Phase | Duration |
|-------|----------|
| Greedy | 30m 58.2s |

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
| hidden_size | 32 |
| initial_sigma | 0.03 |
| policy_params | {'population_size': 20, 'hidden_size': 32, 'initial_sigma': 0.03} |

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

Best reward: **+1740.6**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +631.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1317.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1320.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |  +1168.7 | 0.000    | —           | —       | finish       |  |
|    5 |  +1738.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1508.8 | 0.000    | —           | —       | finish       |  |
|    7 |  +1661.0 | 0.000    | —           | —       | finish       |  |
|    8 |  +1463.5 | 0.000    | —           | —       | finish       |  |
|    9 |  +1649.3 | 0.000    | —           | —       | finish       |  |
|   10 |  +1692.7 | 0.000    | —           | —       | finish       |  |
|   11 |  +1696.6 | 0.000    | —           | —       | finish       |  |
|   12 |  +1698.0 | 0.000    | —           | —       | finish       |  |
|   13 |  +1707.8 | 0.000    | —           | —       | finish       |  |
|   14 |  +1723.1 | 0.000    | —           | —       | finish       |  |
|   15 |  +1044.8 | 0.000    | —           | —       | finish       |  |
|   16 |  +1702.1 | 0.000    | —           | —       | finish       |  |
|   17 |  +1684.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1695.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1703.4 | 0.000    | —           | —       | finish       |  |
|   20 |  +1666.2 | 0.000    | —           | —       | finish       |  |
|   21 |  +1664.0 | 0.000    | —           | —       | finish       |  |
|   22 |  +1720.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1721.4 | 0.000    | —           | —       | finish       |  |
|   24 |  +1709.7 | 0.000    | —           | —       | finish       |  |
|   25 |  +1740.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   26 |  +1719.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1605.4 | 0.000    | —           | —       | finish       |  |
|   28 |  +1636.8 | 0.000    | —           | —       | finish       |  |
|   29 |  +1631.5 | 0.000    | —           | —       | finish       |  |
|   30 |  +1709.7 | 0.000    | —           | —       | finish       |  |
|   31 |  +1713.3 | 0.000    | —           | —       | finish       |  |
|   32 |  +1641.4 | 0.000    | —           | —       | finish       |  |
|   33 |  +1736.9 | 0.000    | —           | —       | finish       |  |
|   34 |  +1702.7 | 0.000    | —           | —       | finish       |  |
|   35 |  +1709.2 | 0.000    | —           | —       | finish       |  |
|   36 |  +1678.4 | 0.000    | —           | —       | finish       |  |
|   37 |  +1739.2 | 0.000    | —           | —       | finish       |  |
|   38 |  +1660.6 | 0.000    | —           | —       | finish       |  |
|   39 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1740.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   41 |  +1565.4 | 0.000    | —           | —       | finish       |  |
|   42 |  +1590.3 | 0.000    | —           | —       | finish       |  |
|   43 |  +1694.4 | 0.000    | —           | —       | finish       |  |
|   44 |  +1501.8 | 0.000    | —           | —       | finish       |  |
|   45 |  +1719.4 | 0.000    | —           | —       | finish       |  |
|   46 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   47 |  +1716.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1711.4 | 0.000    | —           | —       | finish       |  |
|   49 |  +1735.1 | 0.000    | —           | —       | finish       |  |
|   50 |  +1714.3 | 0.000    | —           | —       | finish       |  |

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

