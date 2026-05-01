# TMNF AI — Policy Guide

This guide explains every training algorithm available in this project in plain language. You don't need a machine learning background to follow along — the goal is to give you enough intuition to pick the right algorithm and tune it confidently.

---

## Background: What Is a Policy?

The car needs to decide, many times per second, what to do next: steer left or right, press the accelerator, press the brake. A **policy** is the decision-making program that takes the current situation (what the car can sense right now) and outputs those control signals.

Training means iteratively improving that program by letting the car drive, measuring how well it did, and adjusting the policy so it does better next time.

---

## What the Car Can Sense (Observations)

Every policy receives the same vector of numbers describing the car's current state. There are 21 base features (plus optional LIDAR rays):

| Feature | What it means |
|---|---|
| `speed_ms` | How fast the car is moving, in m/s |
| `lateral_offset_m` | How far left/right the car is from the track centreline |
| `vertical_offset_m` | How far above/below the centreline (useful on banked sections) |
| `yaw_error_rad` | The angle between the car's heading and the track direction |
| `pitch_rad` | Whether the nose is pointing up or down |
| `roll_rad` | Whether the car is tilted left or right |
| `track_progress` | What fraction of the lap is complete, 0 → 1 |
| `turning_rate` | The current steering value from TMInterface |
| `wheel_N_contact` | Whether each of the 4 wheels is touching the ground |
| `angular_vel_x/y/z` | How fast the car is rotating around each axis |
| `lookahead_10/25/50_lat` | Lateral offset of the centreline 10, 25, 50 points ahead |
| `lookahead_10/25/50_yaw` | Heading change of the centreline 10, 25, 50 points ahead |

The lookahead features are the car's "windshield view" — they tell it whether a curve is coming up and how sharp it is.

All features are divided by a scale factor before being fed to the policy, so they all live roughly in the range [−1, 1]. This prevents any single feature from dominating just because it has larger numbers.

---

## What the Car Can Do (Actions)

Every policy ultimately outputs three numbers:

| Output | Range | Effect |
|---|---|---|
| `steer` | [−1, 1] | Full left to full right |
| `accel` | [0, 1] | Thresholded at 0.5: accelerate or not |
| `brake` | [0, 1] | Thresholded at 0.5: brake or not |

Policies that use a **discrete action set** (Q-table-based ones) pick from these 9 combinations:

```
brake+left  | brake+straight  | brake+right
coast+left  | coast+straight  | coast+right
accel+left  | accel+straight  | accel+right
```

Policies that use **continuous actions** (linear, neural net) can produce any steering value in [−1, 1].

---

## The Reward Signal

The training loop scores each episode using a weighted sum of several signals. The dominant signals are:

- **Progress reward**: the main signal — proportional to how much further along the track the car got compared to the last step.
- **Centreline penalty**: penalises drifting away from the centreline (quadratic by default, so small offsets are tolerated but large ones are punished heavily).
- **Speed bonus**: small tie-breaker; rewards going faster.
- **Finish bonus**: large one-time reward for completing the lap.
- **Crash termination**: episode ends immediately if the lateral offset exceeds a threshold.

This means policies learn to drive quickly and accurately, not just reach the end.

---

## Episode Warmup

The first few steps of every episode force the car to accelerate straight, regardless of what the policy says. This covers the braking-start phase where the car is barely moving and most sensors are near-zero — updating policy weights during this phase would teach it nothing useful.

---

## Policies

### 1. `hill_climbing` — WeightedLinearPolicy

**The simplest algorithm. A great starting point.**

#### How it works

The policy computes each control output as a weighted sum of all observation features:

```
steer  = w_steer[0]  * speed   + w_steer[1]  * lateral_offset + ...
accel  = w_accel[0]  * speed   + w_accel[1]  * lateral_offset + ...
brake  = w_brake[0]  * speed   + w_brake[1]  * lateral_offset + ...
```

There are three independent sets of weights — one for steering, one for acceleration, one for braking — giving roughly 63 numbers in total (21 features × 3 heads).

Training works like this:

1. Run one episode with the current weights. Record the total reward.
2. Create a **mutant**: copy the weights and add a small random perturbation to some of them.
3. Run an episode with the mutant. If it scored higher, keep the mutant. Otherwise, discard it and go back to step 2.

Repeat. The weights slowly drift toward combinations that make the car drive better.

**Analogy:** Imagine tuning a guitar by ear. You randomly tighten or loosen a peg a tiny amount and check whether it sounds closer to in-tune. If yes, keep the adjustment. If no, undo it and try again. Simple, reliable, but slow.

#### Training phases

`hill_climbing` is the only policy that runs the probe and cold-start phases:

- **Probe**: before training starts, the car drives 6 fixed-action episodes (brake/accel in each of left/straight/right). This establishes a "floor" — any random policy should at least beat this.
- **Cold-start search**: up to `cold_restarts` rounds of random weight initialisation + short hill-climbing. Finds a decent starting point so the main training run doesn't waste time in a bad local minimum.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `mutation_scale` | `0.05` | Standard deviation of the random perturbation. Higher = bigger steps. If training stalls, try increasing; if it's erratic, try decreasing. |
| `mutation_share` | `1.0` | Fraction of weights perturbed each mutation. `1.0` = all weights; `0.3` = 30% chosen at random each step. Lower values give finer control. |
| `n_sims` | `100` | Number of mutation+evaluate cycles to run. |

#### When to use it

- First experiment on a new track.
- When you want interpretable weights (you can open `policy_weights.yaml` and see which features drive each output).
- When compute is limited — it's the lightest algorithm here.

---

### 2. `neural_net` — NeuralNetPolicy

**Like hill climbing, but with a richer model.**

#### How it works

Instead of a single weighted sum, this policy uses a **multi-layer perceptron** (MLP) — a stack of linear transformations with non-linear activations (ReLU) between them. A typical network with `hidden_sizes: [64, 64]` looks like:

```
obs (21) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(3)
              ↑ hidden layer 1              ↑ hidden layer 2     ↑ outputs
```

The extra hidden layers let the policy learn non-linear relationships. For example, a linear policy can't easily represent "steer harder when I'm both fast AND far from the centreline" — the neural net can.

Training is identical to hill climbing: mutate the network weights, run an episode, keep if better.

#### Key hyperparameters

| Parameter | Location | Effect |
|---|---|---|
| `hidden_sizes` | `policy_params` | List of hidden layer widths. `[64, 64]` is a good default. Wider/deeper = more expressive but harder to train by mutation. |
| `mutation_scale` | `training_params` | Same as hill climbing. |

#### When to use it

- When hill climbing plateaus and you suspect the optimal policy is non-linear.
- Note: because network weights are high-dimensional, mutation-based training is inefficient for large networks. Consider `cmaes` or `neural_dqn` for serious neural net training.

---

### 3. `epsilon_greedy` — EpsilonGreedyPolicy

**Tabular Q-learning. The classical RL workhorse.**

#### How it works

Instead of learning weights that compute actions directly, this policy maintains a **Q-table**: a lookup table that stores the estimated total future reward for every (state, action) pair.

```
Q[state, "accel+right"] = 142.7
Q[state, "coast+left"]  = -3.4
Q[state, "accel+left"]  = 98.2
...
```

To pick an action, the policy either:
- **Explores** (with probability ε): picks a random action from the 9 options.
- **Exploits** (with probability 1−ε): picks the action with the highest Q-value for the current state.

After each step, the Q-table is updated using the **Bellman equation**:

```
Q[s, a] += alpha * (reward + gamma * max(Q[s']) - Q[s, a])
```

This says: "the value of taking action `a` in state `s` should equal the reward I just got, plus a discounted estimate of how good the next state is, minus what I originally thought it would be."

Over many episodes, Q-values converge to accurately reflect how much total reward each (state, action) pair leads to.

**The exploration-exploitation trade-off:** early in training, ε is high (lots of random exploration to discover what actions lead where). Over episodes, ε decays so the policy increasingly exploits what it has learned. Eventually it becomes nearly greedy.

#### Why discretise the observation?

A Q-table can't have an entry for every possible floating-point observation — there are infinitely many. Instead, each continuous observation feature is **binned** into `n_bins` buckets. With 21 features and 3 bins each, there are 3²¹ ≈ 10 billion possible states — too many to populate. In practice, the car only visits a small fraction of them, so the table stays manageable.

More bins = finer resolution but sparser coverage. Start with `n_bins: 2` or `3`.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_bins` | `3` | How many buckets per observation dimension. More = finer but table grows exponentially. |
| `epsilon` | `1.0` | Starting exploration rate. 1.0 = fully random at first. |
| `epsilon_decay` | `0.995` | Multiply ε by this after each episode. 0.995 reaches ~0.05 in ~600 episodes. |
| `epsilon_min` | `0.05` | Minimum ε — always explores at least 5% of the time. |
| `alpha` | `0.1` | Learning rate. How much each new experience updates the Q-value. |
| `gamma` | `0.99` | Discount factor. 0.99 = future rewards are nearly as valuable as immediate ones. |

#### When to use it

- When you want a proven, theoretically grounded algorithm.
- Smaller observation spaces or lower `n_bins` settings work best.
- Good for understanding which state features actually matter (Q-values are inspectable).

---

### 4. `mcts` — MCTSPolicy

**Tabular Q-learning with curiosity-driven exploration.**

#### How it works

MCTS here refers to an **online UCT-style Q-learner** — it's less like the tree-search MCTS used in game-playing AI (AlphaGo) and more like tabular Q-learning with a smarter exploration bonus.

Instead of ε-greedy (explore randomly), this policy uses **Upper Confidence Bound for Trees (UCB1)**:

```
score(s, a) = Q[s, a] + c * sqrt(log(visits(s)) / visits(s, a))
```

Pick the action with the highest score. The second term is an **exploration bonus**: it grows for actions that haven't been tried much relative to how often the state has been visited. Actions get "cheaper to explore" as the state is visited more.

After each step, Q-values are updated the same way as epsilon-greedy (Bellman equation). The difference is purely in how exploration is driven: UCB1 is more systematic than random ε-greedy — it ensures every action in a visited state gets tried eventually, without fully random thrashing.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `c` | `1.41` | Exploration constant (≈√2). Higher = explore more aggressively. Lower = exploit sooner. |
| `alpha` | `0.1` | Q-learning rate. |
| `gamma` | `0.99` | Discount factor. |
| `n_bins` | `3` | Observation discretisation bins (same as epsilon_greedy). |

#### When to use it

- When epsilon_greedy's random exploration feels wasteful.
- When you want more consistent coverage of the state space.
- Note: UCB1 keeps visiting less-certain actions even late in training, which can slow convergence compared to epsilon_greedy's decaying ε.

---

### 5. `genetic` — GeneticPolicy

**A population of agents that breed and compete.**

#### How it works

Instead of training a single policy, the genetic algorithm maintains a **population** of `population_size` independent `WeightedLinearPolicy` individuals, each with their own weights.

Each **generation**:

1. **Evaluate**: run one episode per individual, record each one's reward.
2. **Select elites**: keep the top `elite_k` individuals unchanged (elitism — the best always survive).
3. **Breed the rest**: for each non-elite slot in the next generation:
   a. Randomly pick two parents from the elite group.
   b. **Crossover**: for each weight position, randomly inherit from parent A or parent B (uniform crossover).
   c. **Mutate**: add small Gaussian noise to some weights.
4. Replace the non-elite individuals with the new offspring.
5. Record the best individual ever seen (**champion**) across all generations.

**Why populations?** A single agent exploring by mutation can get stuck in a local optimum — a weight configuration that's decent but not great, where every mutation makes it worse. A population maintains diversity: different individuals explore different regions of the weight space simultaneously, and crossover lets good traits from different lineages combine.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | `10` | Number of individuals. More = more diversity but more episodes per generation. |
| `elite_k` | `3` | Number of top individuals that survive unchanged. Higher = more stability but less diversity. |
| `mutation_scale` | `0.1` | Noise added during mutation. |
| `mutation_share` | `1.0` | Fraction of weights mutated per offspring. |

#### When to use it

- When hill climbing gets stuck and you want more exploratory power.
- When you have parallel compute (each individual can be evaluated concurrently).
- The champion policy is saved in the same YAML format as hill climbing, so analytics work identically.

---

### 6. `cmaes` — CMAESPolicy

**The most mathematically sophisticated linear-policy trainer.**

#### How it works

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is like the genetic algorithm, but instead of simple crossover + mutation, it maintains a **multivariate Gaussian distribution** over the full weight space and adapts both the **step size** (how big the random perturbations are) and the **shape** of the search (which directions tend to be promising).

Each generation:

1. **Sample**: draw `population_size` offspring from the current Gaussian `N(mean, σ² · C)`, where `mean` is the current best estimate of good weights, `σ` is the step size, and `C` is the covariance matrix capturing correlations between weight dimensions.
2. **Evaluate**: run each offspring for one episode.
3. **Update**:
   - **New mean**: shift toward the weighted average of the top half of offspring (better ones count more).
   - **Step size adaptation (CSA)**: if the search is making steady progress in one direction, increase σ (take bigger steps); if direction keeps reversing, decrease σ.
   - **Covariance update**: rotate and stretch the search distribution to align with the directions that have been productive (rank-1 + rank-μ update from the Hansen 2016 paper).

**Why is this powerful?** Imagine the weight landscape has a long, narrow valley. Hill climbing would bounce back and forth across the valley walls. CMA-ES learns that the valley is narrow in one dimension and wide in another, and adapts to search primarily along the valley floor — much more efficient.

The search space here is ~63 dimensions (21 observations × 3 heads). CMA-ES is well-suited to 10–200 dimensional problems.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | `20` | λ — offspring per generation. Larger populations cover more ground but need more episodes. |
| `initial_sigma` | `0.3` | Starting step size. CMA-ES adapts this automatically, but a poor initial value can slow early progress. |

There is no `mutation_scale` — σ adapts automatically throughout training.

Total episodes = `n_sims × population_size`.

#### When to use it

- Best general-purpose algorithm for the linear policy (WeightedLinearPolicy weights).
- When hill climbing or genetic plateau early.
- When you have enough compute for larger populations.

---

### 7. `neural_dqn` — NeuralDQNPolicy

**Deep Q-learning with all the stability tricks.**

#### How it works

`neural_dqn` replaces the Q-table (which only works with discretised states) with a neural network that learns to approximate Q-values directly from continuous observations:

```
obs (21) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(9)
                                                         ↑ one Q-value per action
```

The output is a vector of 9 numbers — one predicted "how much reward will I eventually get?" score for each of the 9 discrete actions. The agent picks the action with the highest score (or explores randomly with probability ε).

The key innovations over naive neural Q-learning are:

**Experience replay buffer**: every transition (observation, action, reward, next observation, done) is stored in a circular buffer. Instead of learning from each experience immediately (which is unstable), the network is trained on random mini-batches sampled from the buffer. This breaks the temporal correlation between successive experiences and makes training much more stable.

**Target network**: there are actually two copies of the network — an **online** network (updated every step) and a **target** network (a snapshot, frozen and only updated every `target_update_freq` gradient steps). The target network is used to compute the "what should the Q-value be" side of the Bellman equation. Without it, you'd be chasing a moving target, which causes training to diverge.

**Adam optimizer**: gradient updates use the Adam algorithm, which adapts the learning rate per-parameter based on running estimates of gradient mean and variance. This makes training more robust to poorly-scaled observations.

**Epsilon decay (linear)**: ε starts at 1.0 (fully random), and decreases by a fixed amount per step until it reaches `epsilon_end`. Unlike tabular methods, this decays per *step* rather than per episode.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Architecture. Wider/deeper = more expressive but slower to train. |
| `learning_rate` | `0.001` | Adam step size. Too high = unstable; too low = slow. |
| `batch_size` | `64` | Mini-batch size for each gradient step. |
| `replay_buffer_size` | `10000` | How many past transitions to remember. |
| `min_replay_size` | `500` | Don't start training until this many transitions are in the buffer. |
| `target_update_freq` | `200` | Sync target network every N gradient steps. |
| `epsilon_decay_steps` | `5000` | Number of steps to linearly decay ε from start to end. |
| `epsilon_end` | `0.05` | Minimum exploration rate. |
| `gamma` | `0.99` | Discount factor. |

#### When to use it

- When you want gradient-based training for a neural policy (more sample-efficient than mutation-based `neural_net`).
- When the observation space is too large for a Q-table.
- Requires more tuning than evolutionary methods but can be significantly more powerful with the right hyperparameters.

---

### 8. `reinforce` — REINFORCEPolicy

**Teaching by reflection: update after the whole episode plays out.**

#### How it works

REINFORCE is a **policy gradient** algorithm. Instead of learning Q-values and picking the highest, it directly learns a policy that outputs probabilities over the 9 actions:

```
obs (21) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(9) → softmax → probabilities
```

At each step the policy samples an action according to these probabilities (rather than always picking the highest). This built-in randomness provides exploration.

The clever part happens at the **end of each episode**. For each step t, compute the **discounted return**:

```
G_t = reward_t + γ·reward_{t+1} + γ²·reward_{t+2} + ...
```

This is "how much total reward did I actually collect starting from step t?" Then update the network weights to:
- **Increase** the probability of actions that had high G_t (they led to good outcomes).
- **Decrease** the probability of actions that had low G_t (they led to poor outcomes).

This is equivalent to gradient ascent on the expected return.

**Baseline subtraction**: raw returns G_t are noisy (the same action can lead to high returns on a lucky episode and low returns on an unlucky one). Subtracting a baseline — here, a running mean of past episode totals — gives advantages (G_t − baseline) that are centred near zero. This dramatically reduces variance and makes learning more stable.

**Entropy regularisation**: a bonus term in the gradient update rewards the policy for being more uncertain (higher entropy over its action distribution). Without this, the policy can collapse to always picking the same action early in training and never escape. The coefficient `entropy_coeff` controls how strong this push toward diversity is.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Network architecture. |
| `learning_rate` | `0.001` | Gradient ascent step size. |
| `gamma` | `0.99` | Discount factor for return computation. |
| `entropy_coeff` | `0.01` | Strength of exploration bonus. Increase if the policy collapses to one action; decrease if it won't commit. |
| `baseline` | `"running_mean"` | `"running_mean"` (EMA of episode totals) or `"none"` (no baseline, higher variance). |

#### Compared to DQN

| | REINFORCE | DQN |
|---|---|---|
| When it learns | End of episode | After every step (batch) |
| Exploration | Stochastic policy | ε-greedy |
| Sample efficiency | Lower (whole-episode updates) | Higher (replay buffer) |
| Stability | Can be high variance | Stabilised by target network |
| Conceptual simplicity | Simple gradient ascent | More engineering overhead |

REINFORCE is elegant and theoretically clean. It can struggle with high-variance returns (long episodes where many steps contribute to the final score), but entropy regularisation and baseline subtraction mitigate this significantly.

#### When to use it

- When you want a stochastic policy (useful when the best strategy involves varied behaviour).
- As an alternative to DQN that requires less hyperparameter tuning for stability.
- Can converge more slowly than DQN but is simpler to reason about.

---

### 9. `lstm` — LSTMEvolutionPolicy

**A car with short-term memory, trained by evolution.**

#### How it works

All previous policies are **memoryless**: given the same observation, they always produce the same action (or the same probability distribution over actions). But driving well sometimes requires memory: "I've been drifting right for the last two seconds, I should correct harder." The raw observation features don't encode past history.

An LSTM (**Long Short-Term Memory**) network solves this by maintaining a hidden state (h, c) that persists across steps within an episode. At each step, the new observation is combined with the previous hidden state to produce an updated hidden state and outputs:

```
[h_t, c_t] = LSTM([h_{t-1}, c_{t-1}], obs_t)

steer  = tanh(W_steer · h_t)
accel  = sigmoid(W_accel · h_t) > 0.5
brake  = sigmoid(W_brake · h_t) > 0.5
```

The LSTM has four internal gates that control what information to remember, what to forget, and what to output. This makes it capable of "noticing" patterns that unfold over time, like the car gradually drifting off course over multiple steps.

At the start of each episode, h and c are reset to zero. So the memory is episode-scoped, not cross-episode.

#### Why evolutionary training?

REINFORCE and DQN train neural networks using gradient backpropagation through time (BPTT), which is complex and can be numerically unstable for LSTMs. Instead, this implementation uses an **isotropic Gaussian evolutionary strategy** — essentially a simplified CMA-ES without the full covariance matrix (which would be infeasible for the ~7,000-dimensional LSTM parameter space):

1. **Sample** λ candidate weight vectors from `N(mean, σ² · I)` — Gaussian noise around the current best.
2. **Evaluate** each candidate LSTM for one episode.
3. **Update mean**: shift toward the weighted average of the top-μ candidates.
4. **Adapt σ** using the **1/5 success rule**: if more than 20% of candidates improved on the current champion, increase σ (explore more); otherwise shrink it (focus closer to what's working).

This avoids backpropagation entirely — the LSTM is treated as a black box to be optimised by search.

#### Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `hidden_size` | `32` | Number of LSTM hidden units. More = more memory capacity but larger parameter space and slower evolution. |
| `population_size` | `20` | λ — candidates evaluated per generation. |
| `initial_sigma` | `0.05` | Initial perturbation scale. Much smaller than CMA-ES defaults because the LSTM parameter space is very high-dimensional. |

Total episodes = `n_sims × population_size`.

**Warning**: changing `hidden_size` between runs is incompatible with existing saved weights (the parameter vector changes shape). If you want to resume training, keep `hidden_size` consistent.

#### When to use it

- When you suspect the policy needs to react to temporal patterns (e.g., accumulated drift, remembering a previous corner).
- When memoryless policies (hill climbing, genetic) consistently fail on a particular section where context matters.
- The 7K+ dimensional parameter space means this needs more generations than CMA-ES to converge. Expect to run more `n_sims` or larger populations.

---

## Algorithm Comparison Summary

| `policy_type` | Model | Training method | Action space | Memory | Complexity |
|---|---|---|---|---|---|
| `hill_climbing` | Linear (3 heads) | Mutate-and-keep | Continuous | None | Very low |
| `neural_net` | MLP | Mutate-and-keep | Continuous | None | Low |
| `epsilon_greedy` | Q-table | TD learning | Discrete (9) | None | Low |
| `mcts` | Q-table | TD + UCB1 | Discrete (9) | None | Low |
| `genetic` | Linear population | Evolutionary crossover | Continuous | None | Medium |
| `cmaes` | Linear (flat vec) | CMA-ES | Continuous | None | Medium |
| `neural_dqn` | MLP | DQN + replay + target net | Discrete (9) | None | High |
| `reinforce` | MLP | Policy gradient (MC) | Discrete (9) | None | Medium |
| `lstm` | LSTM | Isotropic Gaussian ES | Continuous | Yes (episode) | High |

---

## Choosing an Algorithm

**Start here:** `hill_climbing`. It's the simplest, fastest, and most interpretable. The saved weights are human-readable YAML. If it plateaus, move to `cmaes` — it's a strict upgrade for the same linear model.

**Want more model capacity?** Try `neural_dqn` or `reinforce`. Both use gradient-based training and can learn non-linear policies. `neural_dqn` is generally more sample-efficient; `reinforce` is simpler to debug.

**Stuck in local optima?** `genetic` adds population diversity to the linear model. `cmaes` adds adaptive search direction — usually the better choice.

**Believe memory matters?** `lstm` is the only option here. Expect it to need more compute than the others.

**For exploration/research:** `epsilon_greedy` and `mcts` are great for understanding what the car is learning, since Q-table entries can be inspected directly.

---

## Config Quick-Reference

Set `policy_type` in `training_params.yaml`. Additional hyperparameters go in `policy_params`:

```yaml
policy_type: cmaes
policy_params:
  population_size: 20
  initial_sigma: 0.3
```

```yaml
policy_type: epsilon_greedy
policy_params:
  n_bins: 3
  epsilon_decay: 0.995
  alpha: 0.1
```

```yaml
policy_type: neural_dqn
policy_params:
  hidden_sizes: [64, 64]
  learning_rate: 0.001
  batch_size: 64
  target_update_freq: 200
  epsilon_decay_steps: 5000
```

```yaml
policy_type: reinforce
policy_params:
  hidden_sizes: [64, 64]
  learning_rate: 0.001
  entropy_coeff: 0.01
  baseline: running_mean
```

```yaml
policy_type: lstm
policy_params:
  hidden_size: 32
  population_size: 20
  initial_sigma: 0.05
```
