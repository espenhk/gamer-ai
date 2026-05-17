# Experiment: gs_sc2_cmaes_v1__arprotoss__pop25__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-14 18:44:26
- **End:** 2026-05-15 00:42:23
- **Total runtime:** 5h 57m 57.7s

| Phase | Duration |
|-------|----------|
| Greedy | 5h 57m 56.7s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatZerglingsAndBanelings |
| map_name | DefeatZerglingsAndBanelings |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | protoss |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 25 |
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 25, 'initial_sigma': 0.1} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 10.0 |
| win_bonus | 1000.0 |
| loss_penalty | -100.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 0.5 |
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+694.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +359.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +212.0 | 0.000    | —           | —       | finish       |  |
|    3 |   +358.9 | 0.000    | —           | —       | finish       |  |
|    4 |   +263.8 | 0.000    | —           | —       | finish       |  |
|    5 |   +352.7 | 0.000    | —           | —       | finish       |  |
|    6 |   +397.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +387.3 | 0.000    | —           | —       | finish       |  |
|    8 |   +539.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +375.8 | 0.000    | —           | —       | finish       |  |
|   10 |   +360.5 | 0.000    | —           | —       | finish       |  |
|   11 |   +450.0 | 0.000    | —           | —       | finish       |  |
|   12 |   +617.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   13 |   +666.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +614.5 | 0.000    | —           | —       | finish       |  |
|   15 |   +431.2 | 0.000    | —           | —       | finish       |  |
|   16 |   +543.5 | 0.000    | —           | —       | finish       |  |
|   17 |   +569.9 | 0.000    | —           | —       | finish       |  |
|   18 |   +651.8 | 0.000    | —           | —       | finish       |  |
|   19 |   +472.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +591.1 | 0.000    | —           | —       | finish       |  |
|   21 |   +407.8 | 0.000    | —           | —       | finish       |  |
|   22 |   +620.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +667.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |   +407.8 | 0.000    | —           | —       | finish       |  |
|   25 |   +320.7 | 0.000    | —           | —       | finish       |  |
|   26 |   +629.5 | 0.000    | —           | —       | finish       |  |
|   27 |   +449.2 | 0.000    | —           | —       | finish       |  |
|   28 |   +524.9 | 0.000    | —           | —       | finish       |  |
|   29 |   +387.8 | 0.000    | —           | —       | finish       |  |
|   30 |   +509.1 | 0.000    | —           | —       | finish       |  |
|   31 |   +459.2 | 0.000    | —           | —       | finish       |  |
|   32 |   +562.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +601.4 | 0.000    | —           | —       | finish       |  |
|   34 |   +487.4 | 0.000    | —           | —       | finish       |  |
|   35 |   +577.2 | 0.000    | —           | —       | finish       |  |
|   36 |   +447.2 | 0.000    | —           | —       | finish       |  |
|   37 |   +418.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +694.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   39 |   +556.0 | 0.000    | —           | —       | finish       |  |
|   40 |   +432.6 | 0.000    | —           | —       | finish       |  |
|   41 |   +602.6 | 0.000    | —           | —       | finish       |  |
|   42 |   +624.3 | 0.000    | —           | —       | finish       |  |
|   43 |   +540.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +401.2 | 0.000    | —           | —       | finish       |  |
|   45 |   +560.4 | 0.000    | —           | —       | finish       |  |
|   46 |   +528.8 | 0.000    | —           | —       | finish       |  |
|   47 |   +576.7 | 0.000    | —           | —       | finish       |  |
|   48 |   +492.3 | 0.000    | —           | —       | finish       |  |
|   49 |   +463.3 | 0.000    | —           | —       | finish       |  |
|   50 |   +569.3 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Skipped frames](skipped_frames.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Build order](build_order.png)


![Reward trajectory](reward_trajectory.png)

