# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.05__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 15:58:20
- **End:** 2026-05-08 16:42:57
- **Total runtime:** 44m 37.5s

| Phase | Duration |
|-------|----------|
| Greedy | 44m 36.4s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1733.3**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1684.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +388.1 | 0.000    | —           | —       | finish       |  |
|    3 |   +405.3 | 0.000    | —           | —       | finish       |  |
|    4 |  +1697.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1726.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1686.1 | 0.000    | —           | —       | finish       |  |
|    7 |  +1234.3 | 0.000    | —           | —       | finish       |  |
|    8 |  +1176.1 | 0.000    | —           | —       | finish       |  |
|    9 |   +848.0 | 0.000    | —           | —       | finish       |  |
|   10 |  +1055.4 | 0.000    | —           | —       | finish       |  |
|   11 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   12 |  +1718.9 | 0.000    | —           | —       | finish       |  |
|   13 |  +1718.9 | 0.000    | —           | —       | finish       |  |
|   14 |  +1726.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   16 |  +1632.3 | 0.000    | —           | —       | finish       |  |
|   17 |   +672.2 | 0.000    | —           | —       | finish       |  |
|   18 |  +1292.5 | 0.000    | —           | —       | finish       |  |
|   19 |   +860.1 | 0.000    | —           | —       | finish       |  |
|   20 |  +1538.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1705.3 | 0.000    | —           | —       | finish       |  |
|   22 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1709.3 | 0.000    | —           | —       | finish       |  |
|   24 |  +1698.5 | 0.000    | —           | —       | finish       |  |
|   25 |  +1709.7 | 0.000    | —           | —       | finish       |  |
|   26 |  +1721.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1687.1 | 0.000    | —           | —       | finish       |  |
|   28 |  +1628.1 | 0.000    | —           | —       | finish       |  |
|   29 |  +1164.5 | 0.000    | —           | —       | finish       |  |
|   30 |   +587.1 | 0.000    | —           | —       | finish       |  |
|   31 |   +973.7 | 0.000    | —           | —       | finish       |  |
|   32 |  +1727.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   33 |  +1705.3 | 0.000    | —           | —       | finish       |  |
|   34 |  +1733.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   35 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   36 |  +1656.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1710.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1505.3 | 0.000    | —           | —       | finish       |  |
|   39 |  +1614.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1292.1 | 0.000    | —           | —       | finish       |  |
|   41 |   +603.2 | 0.000    | —           | —       | finish       |  |
|   42 |  +1117.3 | 0.000    | —           | —       | finish       |  |
|   43 |  +1362.9 | 0.000    | —           | —       | finish       |  |
|   44 |   +614.4 | 0.000    | —           | —       | finish       |  |
|   45 |  +1008.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1710.5 | 0.000    | —           | —       | finish       |  |
|   47 |  +1628.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1684.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1621.9 | 0.000    | —           | —       | finish       |  |
|   50 |  +1534.5 | 0.000    | —           | —       | finish       |  |

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

