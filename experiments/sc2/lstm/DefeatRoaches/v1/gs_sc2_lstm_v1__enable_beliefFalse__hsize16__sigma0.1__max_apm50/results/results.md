# Experiment: gs_sc2_lstm_v1__enable_beliefFalse__hsize16__sigma0.1__max_apm50

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 10:50:42
- **End:** 2026-05-07 11:33:59
- **Total runtime:** 43m 16.3s

| Phase | Duration |
|-------|----------|
| Greedy | 43m 15.3s |

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
| agent_race | random |
| n_sims | 50 |
| policy_type | lstm |
| obs_spec_preset | rich |
| enable_belief | False |
| hidden_size | 16 |
| initial_sigma | 0.1 |
| max_apm | 50 |
| policy_params | {'population_size': 20, 'hidden_size': 16, 'initial_sigma': 0.1} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 1.0 |
| win_bonus | 20.0 |
| loss_penalty | 0.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 1.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1790.1**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1710.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|    3 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|    4 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|    5 |  +1684.1 | 0.000    | —           | —       | finish       |  |
|    6 |  +1662.1 | 0.000    | —           | —       | finish       |  |
|    7 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1670.1 | 0.000    | —           | —       | finish       |  |
|    9 |  +1662.1 | 0.000    | —           | —       | finish       |  |
|   10 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   13 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   14 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   15 |  +1790.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   16 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   17 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   18 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   19 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   20 |  +1654.1 | 0.000    | —           | —       | finish       |  |
|   21 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   23 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   24 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   25 |  +1662.1 | 0.000    | —           | —       | finish       |  |
|   26 |  +1214.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   28 |  +1742.1 | 0.000    | —           | —       | finish       |  |
|   29 |  +1662.1 | 0.000    | —           | —       | finish       |  |
|   30 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   31 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1750.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   34 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   35 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|   36 |  +1710.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1646.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1654.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   42 |  +1670.1 | 0.000    | —           | —       | finish       |  |
|   43 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   44 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   45 |  +1694.1 | 0.000    | —           | —       | finish       |  |
|   46 |  +1742.1 | 0.000    | —           | —       | finish       |  |
|   47 |  +1670.1 | 0.000    | —           | —       | finish       |  |
|   48 |  +1670.1 | 0.000    | —           | —       | finish       |  |
|   49 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   50 |  +1670.1 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Build order](build_order.png)


![Reward trajectory](reward_trajectory.png)

