# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.2__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 11:37:17
- **End:** 2026-05-08 12:16:36
- **Total runtime:** 39m 18.6s

| Phase | Duration |
|-------|----------|
| Greedy | 39m 17.6s |

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
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1799.3**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1709.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1717.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1524.1 | 0.000    | —           | —       | finish       |  |
|    4 |   +365.9 | 0.000    | —           | —       | finish       |  |
|    5 |  +1724.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1463.3 | 0.000    | —           | —       | finish       |  |
|    7 |  +1678.5 | 0.000    | —           | —       | finish       |  |
|    8 |  +1675.3 | 0.000    | —           | —       | finish       |  |
|    9 |  +1702.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1733.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   11 |   +616.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1716.1 | 0.000    | —           | —       | finish       |  |
|   14 |  +1727.3 | 0.000    | —           | —       | finish       |  |
|   15 |  +1727.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1712.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   18 |  +1719.3 | 0.000    | —           | —       | finish       |  |
|   19 |  +1696.9 | 0.000    | —           | —       | finish       |  |
|   20 |  +1712.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1726.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1735.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |  +1719.3 | 0.000    | —           | —       | finish       |  |
|   25 |  +1735.3 | 0.000    | —           | —       | finish       |  |
|   26 |  +1735.3 | 0.000    | —           | —       | finish       |  |
|   27 |  +1720.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1713.7 | 0.000    | —           | —       | finish       |  |
|   29 |  +1703.3 | 0.000    | —           | —       | finish       |  |
|   30 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   31 |  +1714.5 | 0.000    | —           | —       | finish       |  |
|   32 |  +1720.9 | 0.000    | —           | —       | finish       |  |
|   33 |  +1775.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   34 |  +1783.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   35 |  +1789.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   36 |  +1791.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   37 |  +1788.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1789.7 | 0.000    | —           | —       | finish       |  |
|   39 |  +1778.5 | 0.000    | —           | —       | finish       |  |
|   40 |  +1784.1 | 0.000    | —           | —       | finish       |  |
|   41 |  +1780.1 | 0.000    | —           | —       | finish       |  |
|   42 |  +1782.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1756.1 | 0.000    | —           | —       | finish       |  |
|   44 |  +1791.3 | 0.000    | —           | —       | finish       |  |
|   45 |  +1791.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1783.3 | 0.000    | —           | —       | finish       |  |
|   47 |  +1799.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   48 |  +1799.3 | 0.000    | —           | —       | finish       |  |
|   49 |  +1788.9 | 0.000    | —           | —       | finish       |  |
|   50 |  +1786.5 | 0.000    | —           | —       | finish       |  |

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

