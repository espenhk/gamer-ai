# Experiment: gs_sc2_cmaes_v1__arprotoss__pop25__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-15 06:47:11
- **End:** 2026-05-15 12:58:37
- **Total runtime:** 6h 11m 25.5s

| Phase | Duration |
|-------|----------|
| Greedy | 6h 11m 24.5s |

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

Best reward: **+644.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +256.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +152.5 | 0.000    | —           | —       | finish       |  |
|    3 |   +132.1 | 0.000    | —           | —       | finish       |  |
|    4 |   +357.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +177.6 | 0.000    | —           | —       | finish       |  |
|    6 |   +262.7 | 0.000    | —           | —       | finish       |  |
|    7 |   +291.2 | 0.000    | —           | —       | finish       |  |
|    8 |   +346.8 | 0.000    | —           | —       | finish       |  |
|    9 |   +390.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |   +372.9 | 0.000    | —           | —       | finish       |  |
|   11 |   +430.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   12 |   +437.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   13 |   +407.5 | 0.000    | —           | —       | finish       |  |
|   14 |   +383.8 | 0.000    | —           | —       | finish       |  |
|   15 |   +398.3 | 0.000    | —           | —       | finish       |  |
|   16 |   +278.9 | 0.000    | —           | —       | finish       |  |
|   17 |   +435.2 | 0.000    | —           | —       | finish       |  |
|   18 |   +365.0 | 0.000    | —           | —       | finish       |  |
|   19 |   +334.8 | 0.000    | —           | —       | finish       |  |
|   20 |   +317.4 | 0.000    | —           | —       | finish       |  |
|   21 |   +377.0 | 0.000    | —           | —       | finish       |  |
|   22 |   +326.1 | 0.000    | —           | —       | finish       |  |
|   23 |   +349.1 | 0.000    | —           | —       | finish       |  |
|   24 |   +380.2 | 0.000    | —           | —       | finish       |  |
|   25 |   +304.0 | 0.000    | —           | —       | finish       |  |
|   26 |   +403.0 | 0.000    | —           | —       | finish       |  |
|   27 |   +438.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   28 |   +358.9 | 0.000    | —           | —       | finish       |  |
|   29 |   +390.1 | 0.000    | —           | —       | finish       |  |
|   30 |   +505.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   31 |   +526.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   32 |   +376.1 | 0.000    | —           | —       | finish       |  |
|   33 |   +377.6 | 0.000    | —           | —       | finish       |  |
|   34 |   +453.9 | 0.000    | —           | —       | finish       |  |
|   35 |   +344.0 | 0.000    | —           | —       | finish       |  |
|   36 |   +451.9 | 0.000    | —           | —       | finish       |  |
|   37 |   +355.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +385.2 | 0.000    | —           | —       | finish       |  |
|   39 |   +514.8 | 0.000    | —           | —       | finish       |  |
|   40 |   +429.6 | 0.000    | —           | —       | finish       |  |
|   41 |   +525.4 | 0.000    | —           | —       | finish       |  |
|   42 |   +644.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   43 |   +555.8 | 0.000    | —           | —       | finish       |  |
|   44 |   +445.7 | 0.000    | —           | —       | finish       |  |
|   45 |   +444.1 | 0.000    | —           | —       | finish       |  |
|   46 |   +403.8 | 0.000    | —           | —       | finish       |  |
|   47 |   +408.4 | 0.000    | —           | —       | finish       |  |
|   48 |   +387.0 | 0.000    | —           | —       | finish       |  |
|   49 |   +513.7 | 0.000    | —           | —       | finish       |  |
|   50 |   +519.9 | 0.000    | —           | —       | finish       |  |

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


![Reward trajectory](reward_trajectory.png)

