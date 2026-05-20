# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.05__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 17:32:40
- **End:** 2026-05-08 18:09:43
- **Total runtime:** 37m 03.2s

| Phase | Duration |
|-------|----------|
| Greedy | 37m 02.2s |

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
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1796.1**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1709.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1741.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1704.9 | 0.000    | —           | —       | finish       |  |
|    4 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|    5 |  +1527.0 | 0.000    | —           | —       | finish       |  |
|    6 |   +380.2 | 0.000    | —           | —       | finish       |  |
|    7 |  +1150.4 | 0.000    | —           | —       | finish       |  |
|    8 |  +1686.4 | 0.000    | —           | —       | finish       |  |
|    9 |  +1733.0 | 0.000    | —           | —       | finish       |  |
|   10 |  +1604.6 | 0.000    | —           | —       | finish       |  |
|   11 |  +1069.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1672.9 | 0.000    | —           | —       | finish       |  |
|   13 |  +1723.8 | 0.000    | —           | —       | finish       |  |
|   14 |  +1747.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |  +1500.9 | 0.000    | —           | —       | finish       |  |
|   16 |  +1755.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   17 |  +1756.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   18 |  +1756.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   19 |  +1749.5 | 0.000    | —           | —       | finish       |  |
|   20 |  +1758.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   21 |  +1096.9 | 0.000    | —           | —       | finish       |  |
|   22 |  +1738.2 | 0.000    | —           | —       | finish       |  |
|   23 |  +1723.3 | 0.000    | —           | —       | finish       |  |
|   24 |  +1755.7 | 0.000    | —           | —       | finish       |  |
|   25 |  +1755.7 | 0.000    | —           | —       | finish       |  |
|   26 |  +1756.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1748.4 | 0.000    | —           | —       | finish       |  |
|   28 |  +1789.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   29 |  +1782.1 | 0.000    | —           | —       | finish       |  |
|   30 |  +1790.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   31 |  +1787.3 | 0.000    | —           | —       | finish       |  |
|   32 |  +1771.8 | 0.000    | —           | —       | finish       |  |
|   33 |  +1692.8 | 0.000    | —           | —       | finish       |  |
|   34 |  +1780.4 | 0.000    | —           | —       | finish       |  |
|   35 |  +1796.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   36 |  +1788.6 | 0.000    | —           | —       | finish       |  |
|   37 |  +1781.0 | 0.000    | —           | —       | finish       |  |
|   38 |  +1789.5 | 0.000    | —           | —       | finish       |  |
|   39 |  +1789.4 | 0.000    | —           | —       | finish       |  |
|   40 |  +1790.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1787.0 | 0.000    | —           | —       | finish       |  |
|   42 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   43 |  +1780.4 | 0.000    | —           | —       | finish       |  |
|   44 |  +1780.4 | 0.000    | —           | —       | finish       |  |
|   45 |  +1788.6 | 0.000    | —           | —       | finish       |  |
|   46 |  +1788.8 | 0.000    | —           | —       | finish       |  |
|   47 |  +1782.9 | 0.000    | —           | —       | finish       |  |
|   48 |  +1782.3 | 0.000    | —           | —       | finish       |  |
|   49 |  +1789.9 | 0.000    | —           | —       | finish       |  |
|   50 |  +1795.3 | 0.000    | —           | —       | finish       |  |

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

