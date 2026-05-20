# Experiment: gs_sc2_cmaes_v1__arterran__pop25__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-09 19:35:28
- **End:** 2026-05-10 01:49:31
- **Total runtime:** 6h 14m 02.9s

| Phase | Duration |
|-------|----------|
| Greedy | 6h 14m 01.9s |

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
| agent_race | terran |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 25 |
| initial_sigma | 0.5 |
| policy_params | {'eval_episodes': 5, 'population_size': 25, 'initial_sigma': 0.5} |

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

Best reward: **+845.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +212.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +262.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +439.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +639.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +522.5 | 0.000    | —           | —       | finish       |  |
|    6 |   +228.6 | 0.000    | —           | —       | finish       |  |
|    7 |   +289.3 | 0.000    | —           | —       | finish       |  |
|    8 |   +372.6 | 0.000    | —           | —       | finish       |  |
|    9 |   +328.0 | 0.000    | —           | —       | finish       |  |
|   10 |   +319.1 | 0.000    | —           | —       | finish       |  |
|   11 |   +386.6 | 0.000    | —           | —       | finish       |  |
|   12 |   +466.6 | 0.000    | —           | —       | finish       |  |
|   13 |   +411.5 | 0.000    | —           | —       | finish       |  |
|   14 |   +417.1 | 0.000    | —           | —       | finish       |  |
|   15 |   +365.1 | 0.000    | —           | —       | finish       |  |
|   16 |   +411.4 | 0.000    | —           | —       | finish       |  |
|   17 |   +486.4 | 0.000    | —           | —       | finish       |  |
|   18 |   +322.3 | 0.000    | —           | —       | finish       |  |
|   19 |   +350.6 | 0.000    | —           | —       | finish       |  |
|   20 |   +614.2 | 0.000    | —           | —       | finish       |  |
|   21 |   +463.4 | 0.000    | —           | —       | finish       |  |
|   22 |   +477.7 | 0.000    | —           | —       | finish       |  |
|   23 |   +492.1 | 0.000    | —           | —       | finish       |  |
|   24 |   +533.9 | 0.000    | —           | —       | finish       |  |
|   25 |   +524.5 | 0.000    | —           | —       | finish       |  |
|   26 |   +439.0 | 0.000    | —           | —       | finish       |  |
|   27 |   +548.9 | 0.000    | —           | —       | finish       |  |
|   28 |   +507.8 | 0.000    | —           | —       | finish       |  |
|   29 |   +845.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   30 |   +417.3 | 0.000    | —           | —       | finish       |  |
|   31 |   +629.6 | 0.000    | —           | —       | finish       |  |
|   32 |   +511.0 | 0.000    | —           | —       | finish       |  |
|   33 |   +468.1 | 0.000    | —           | —       | finish       |  |
|   34 |   +638.7 | 0.000    | —           | —       | finish       |  |
|   35 |   +592.9 | 0.000    | —           | —       | finish       |  |
|   36 |   +424.3 | 0.000    | —           | —       | finish       |  |
|   37 |   +563.6 | 0.000    | —           | —       | finish       |  |
|   38 |   +452.2 | 0.000    | —           | —       | finish       |  |
|   39 |   +601.6 | 0.000    | —           | —       | finish       |  |
|   40 |   +539.9 | 0.000    | —           | —       | finish       |  |
|   41 |   +691.2 | 0.000    | —           | —       | finish       |  |
|   42 |   +539.7 | 0.000    | —           | —       | finish       |  |
|   43 |   +442.7 | 0.000    | —           | —       | finish       |  |
|   44 |   +462.8 | 0.000    | —           | —       | finish       |  |
|   45 |   +578.1 | 0.000    | —           | —       | finish       |  |
|   46 |   +492.7 | 0.000    | —           | —       | finish       |  |
|   47 |   +564.8 | 0.000    | —           | —       | finish       |  |
|   48 |   +472.1 | 0.000    | —           | —       | finish       |  |
|   49 |   +596.5 | 0.000    | —           | —       | finish       |  |
|   50 |   +548.0 | 0.000    | —           | —       | finish       |  |

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

