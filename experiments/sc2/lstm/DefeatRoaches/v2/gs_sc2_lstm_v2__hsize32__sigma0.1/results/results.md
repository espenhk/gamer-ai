# Experiment: gs_sc2_lstm_v2__hsize32__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 19:28:14
- **End:** 2026-05-07 20:08:57
- **Total runtime:** 40m 43.0s

| Phase | Duration |
|-------|----------|
| Greedy | 40m 41.9s |

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
| initial_sigma | 0.1 |
| policy_params | {'population_size': 20, 'hidden_size': 32, 'initial_sigma': 0.1} |

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

Best reward: **+1721.4**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1053.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1682.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1721.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |  +1618.7 | 0.000    | —           | —       | finish       |  |
|    5 |  +1677.9 | 0.000    | —           | —       | finish       |  |
|    6 |  +1443.1 | 0.000    | —           | —       | finish       |  |
|    7 |  +1717.9 | 0.000    | —           | —       | finish       |  |
|    8 |  +1684.2 | 0.000    | —           | —       | finish       |  |
|    9 |  +1682.8 | 0.000    | —           | —       | finish       |  |
|   10 |  +1665.9 | 0.000    | —           | —       | finish       |  |
|   11 |  +1683.2 | 0.000    | —           | —       | finish       |  |
|   12 |  +1680.4 | 0.000    | —           | —       | finish       |  |
|   13 |  +1681.9 | 0.000    | —           | —       | finish       |  |
|   14 |  +1682.8 | 0.000    | —           | —       | finish       |  |
|   15 |  +1679.9 | 0.000    | —           | —       | finish       |  |
|   16 |  +1678.5 | 0.000    | —           | —       | finish       |  |
|   17 |  +1678.6 | 0.000    | —           | —       | finish       |  |
|   18 |  +1680.2 | 0.000    | —           | —       | finish       |  |
|   19 |  +1654.1 | 0.000    | —           | —       | finish       |  |
|   20 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   21 |  +1656.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1679.4 | 0.000    | —           | —       | finish       |  |
|   23 |  +1680.5 | 0.000    | —           | —       | finish       |  |
|   24 |  +1658.5 | 0.000    | —           | —       | finish       |  |
|   25 |  +1690.3 | 0.000    | —           | —       | finish       |  |
|   26 |  +1690.2 | 0.000    | —           | —       | finish       |  |
|   27 |  +1659.7 | 0.000    | —           | —       | finish       |  |
|   28 |  +1687.6 | 0.000    | —           | —       | finish       |  |
|   29 |  +1683.9 | 0.000    | —           | —       | finish       |  |
|   30 |  +1686.6 | 0.000    | —           | —       | finish       |  |
|   31 |  +1677.3 | 0.000    | —           | —       | finish       |  |
|   32 |  +1655.8 | 0.000    | —           | —       | finish       |  |
|   33 |  +1680.0 | 0.000    | —           | —       | finish       |  |
|   34 |  +1682.5 | 0.000    | —           | —       | finish       |  |
|   35 |  +1680.5 | 0.000    | —           | —       | finish       |  |
|   36 |  +1677.6 | 0.000    | —           | —       | finish       |  |
|   37 |  +1679.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1689.3 | 0.000    | —           | —       | finish       |  |
|   39 |  +1677.4 | 0.000    | —           | —       | finish       |  |
|   40 |  +1686.7 | 0.000    | —           | —       | finish       |  |
|   41 |  +1675.4 | 0.000    | —           | —       | finish       |  |
|   42 |  +1664.3 | 0.000    | —           | —       | finish       |  |
|   43 |  +1680.4 | 0.000    | —           | —       | finish       |  |
|   44 |  +1528.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1674.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1692.4 | 0.000    | —           | —       | finish       |  |
|   47 |  +1690.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1674.4 | 0.000    | —           | —       | finish       |  |
|   49 |  +1678.2 | 0.000    | —           | —       | finish       |  |
|   50 |  +1677.3 | 0.000    | —           | —       | finish       |  |

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

