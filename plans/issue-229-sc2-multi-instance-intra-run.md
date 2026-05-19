# Plan: SC2 multiple local instances intra-run (issue #229)

Branch: `claude/plan-issue-229-vgcgd`

## Goal

Inside a single experiment, evaluate multiple population members in
parallel on multiple local SC2 binaries.  The training loop today runs
one episode at a time against one PySC2 binary; for population-based
algorithms (`sc2_genetic`, `sc2_cmaes`, `sc2_lstm`, `sc2_cnn`) this is
embarrassingly parallel — each individual is a pure function of its
weight vector → fitness scalar.  Wallclock per generation can drop by
roughly `N×` (minus startup overhead) on a host with `N` SC2 instances
that fit in CPU + RAM.

This is **intra-experiment** parallelism.  It is orthogonal to two
existing capabilities, and we keep both:

1. The `distributed/` coordinator/worker pair distributes **whole
   experiments** across hosts at `ComboSpec` granularity.
2. PR #244 (commit `9fc4087`, "Enable local multi-instance SC2 grid
   search via coordinator-managed worker spawning") added
   `--local-workers N` to `grid_search.py`, which fans out *whole
   experiments* across local worker subprocesses on a single host.

Both of those put one experiment per worker.  Issue #229 is the
remaining case: a single experiment whose **individuals** are split
across local workers.  This matters when you have one config you
actually care about (not a sweep) and want to use the host's full
parallelism on it — exactly the scenario the issue calls out for
population-based algorithms.

One host running this feature can still be a worker inside a
grid-search; `n_workers` (intra) and `--local-workers` (inter) can
coexist as long as their product fits in CPU + RAM.

---

## 1. Current state — confirmed facts

- **One env = one binary.**  `SC2Client._make_sc2_env`
  (`games/sc2/client.py:348`) builds a single `sc2_env.SC2Env` lazily
  on first `reset()`.  `TMNFEnv`-style singleton assumptions don't
  apply — nothing in `games/sc2/env.py` is module-global.

- **Serial per-generation evaluation lives in three places:**
  - `framework/training.py:_greedy_loop_genetic` (`:1003–1140`) —
    `for individual in policy.population: for _ in range(eval_episodes): env.reset(); _run_episode(...)`.
  - `framework/training.py:_greedy_loop_cmaes` (`:820–917`) — same
    pattern over `policy.sample_population()`.
  - `SC2LSTMEvolutionPolicy` and `SC2CNNEvolutionPolicy` use their own
    inline ES loops (`sc2_policies.py:1587–1700`,
    `cnn_policy.py:300–520`) but follow the same
    sample → serial evaluate → update_distribution shape.

- **Fitness function is pure.**  `_run_episode(env, individual, …)`
  takes only `env` and a candidate policy object; the reward
  (`SC2RewardCalculator`, `games/sc2/reward.py`) is computed in-process
  from `info` dicts returned by `env.step()`.  No shared state with
  other individuals.  This is what makes worker parallelism cheap.

- **Policies serialize as flat numpy vectors.**  Every population
  policy has `to_flat()` / `with_flat()`:
  `WeightedLinearPolicy` (`framework/policies.py:155`),
  `SC2MultiHeadLinearPolicy` (`games/sc2/sc2_policies.py:230`),
  `SC2LSTMPolicy` (`sc2_policies.py:1334`),
  `SC2CNNModel` (`cnn_policy.py:183`).
  Shipping `(generation, individual_idx, flat_weights)` to a worker is
  enough to fully reconstruct the candidate.

- **No port management today.**  `_make_sc2_env` never passes ports;
  PySC2 picks them itself.  Multiple binaries on the same host work
  iff PySC2 picks non-colliding ports (it does, via its `Portspicker`
  helper) — confirmed by running it manually; we don't need our own
  port allocator unless we run into races.  Keep a guard for that.

- **Existing distributed code is coarse.**  `ComboSpec`
  (`distributed/protocol.py:27`) is one full experiment per work
  unit; `Coordinator.handle_get_work` pops one combo, posts one
  `ExperimentData` back.  Reusing this for per-individual jobs would
  require breaking the wire format.  Cheaper for this issue: build a
  separate in-process pool, leave `distributed/` untouched.

- **Analytics records one `GreedySimResult` per generation**
  (`framework/analytics.py:157`; populated at
  `training.py:885–903` and `:1107–1126`).  The per-individual rewards
  inside a generation are collapsed to `gen_best = max(rewards)`
  *before* the result row is built — there is no per-individual
  timeline today.  This means **order-of-completion within a
  generation is already invisible to analytics**, so async per-worker
  completion does not break anything as long as we still produce one
  row per generation at the end of that generation.

---

## 2. Design choice — process pool with persistent workers

Three options were considered:

| Option | Pros | Cons |
|---|---|---|
| **A. `multiprocessing.Pool` per call** | minimal code | binary startup is ~10–20 s; paying it every generation kills the gain |
| **B. Persistent worker processes, IPC via stdlib `multiprocessing.Queue`** | binary stays alive across generations; no HTTP overhead; works fully offline | needs care around fork-vs-spawn (PySC2 + absl.flags do not fork-safely) |
| **C. Persistent workers as HTTP children of the local coordinator** | reuses `distributed/` code path | extra complexity (HTTP, auth, payloads) for a single-host case; defers binary cost behind a useless network hop |

**Pick B.**  Persistent workers, `spawn` start method (PySC2 + absl
require it), one queue for jobs in, one queue for results out, one
`SC2Env` per worker held across the run.  Coordinator (main process)
issues `N` jobs per generation, blocks on result queue, sorts by
`individual_idx`, then runs the existing `update_distribution` /
`evaluate_and_evolve` step unchanged.

This keeps the algorithm exactly synchronous **at the generation
boundary** — which is what every ES variant assumes — while
parallelising the dominant cost (episode rollouts) inside the
generation.

The issue mentions "asynchronous, incremental updates as each
instance finishes."  True per-individual async update would require
algorithmic changes (asynchronous CMA-ES, async ES) that are out of
scope.  Generation-level synchronisation gives us 95% of the speed-up
with zero algorithmic risk.  Note this design choice in the issue
when closing.

---

## 3. New module: `framework/parallel_eval.py`

Pure framework code (game-agnostic).  Public surface:

```python
class ParallelEvaluator:
    def __init__(self, n_workers: int, game: GameSpec, config: RunConfig): ...
    def evaluate(self, candidates: list[Candidate]) -> list[EpisodeResult]: ...
    def close(self) -> None: ...

@dataclass
class Candidate:
    individual_idx: int
    flat_weights: np.ndarray
    eval_episodes: int

@dataclass
class EpisodeResult:
    individual_idx: int
    reward: float            # mean over eval_episodes
    info: dict               # info dict from the *last* episode (for logging)
    trace: RunTrace | None   # from the last episode
    total_steps: int         # summed across eval_episodes
```

Internals:

- `__init__` spawns `n_workers` child processes via
  `multiprocessing.get_context("spawn").Process(target=_worker_main,
  args=(job_q, result_q, game_spec, config_dict, worker_id))`.
- `_worker_main` builds `env = game.build_env(...)`, builds a
  template policy via `game.build_policy(...)`, then loops:
  pull `Candidate` from `job_q` → `template.with_flat(c.flat_weights)`
  → run `eval_episodes` episodes → push `EpisodeResult` to
  `result_q`.  Exits on sentinel `None`.
- `evaluate(candidates)` pushes all jobs, gathers `len(candidates)`
  results, returns them sorted by `individual_idx`.
- `close()` sends `n_workers` sentinels, joins.  Also registered via
  `atexit` so a `KeyboardInterrupt` in the main loop tears down
  child binaries.

Pickling: `GameSpec` and `RunConfig` must round-trip through pickle.
`GameSpec` (in `framework/game_api.py`) currently holds methods —
need to confirm it's a plain dataclass or a small class without
unpicklable members; if not, ship the dict needed to reconstruct it
inside the worker (the worker imports its own `GameSpec` factory by
game name — same dispatch already used in `distributed/worker.py`).

---

## 4. Hook into the training loop

Smallest possible change to existing loops:

- Add an optional helper `_evaluate_population(policy, env, evaluator,
  warmup_action, warmup_steps, eval_episodes) -> (rewards, info,
  trace, total_steps)`:
  - if `evaluator is None`: keep current serial logic verbatim
    (the inner `for individual in offspring/population` loop).
  - else: build `[Candidate(idx, ind.to_flat(), eval_episodes) for
    idx, ind in enumerate(offspring)]`, call `evaluator.evaluate(...)`,
    then `rewards = [r.reward for r in results]`,
    `info = results[best_idx].info`, etc.

- Wire it into `_greedy_loop_cmaes` (`training.py:854`),
  `_greedy_loop_genetic` (`:1058`), and the inline loops in
  `SC2LSTMEvolutionPolicy` and `SC2CNNEvolutionPolicy`.  Each call
  site is ~5 lines today.

- `train_rl` (`framework/training.py:1147`) builds the evaluator once
  before dispatching to the greedy loop, passes it through, and
  closes it in a `finally`.  Only constructed when
  `n_workers > 1 and policy_type in {sc2_genetic, sc2_cmaes,
  sc2_lstm, sc2_cnn}`.

Tabular and gradient-based policies (`epsilon_greedy`, `mcts`,
`sc2_reinforce`, `neural_dqn`, `reinforce`) are not population-based
and not in scope.  Their loops keep using `env` directly.

---

## 5. Config

Add to `games/sc2/config/training_params.yaml` (and document in
`CLAUDE.md`):

| Key | Default | Notes |
|---|---|---|
| `n_workers` | `1` | Number of parallel SC2 binaries.  `1` = current serial behaviour, zero overhead. |
| `worker_start_stagger_s` | `5.0` | Sleep between spawning child workers, so PySC2's port-picker and absl.flags init don't race. |
| `worker_warmup_timeout_s` | `90.0` | If a worker can't open its SC2 binary inside this window, fail loudly with its captured stderr. |

Validation: refuse `n_workers > 1` for non-population policies (early
error in `train_rl`, before any binary spawns).  Cap at
`min(n_workers, population_size)` — extra workers would just idle.

Per-host sizing guidance (add to CLAUDE.md): one SC2 binary on Linux
headless mode uses ~1 CPU + ~1.5 GB RSS, so a 16-core / 32 GB box
sustains roughly 8 binaries with overhead for the trainer.

---

## 6. Failure modes & resilience

- **Worker dies mid-episode.**  `evaluate()` waits on the result
  queue with a timeout = `worker_warmup_timeout_s +
  in_game_episode_s × eval_episodes × 2`.  On timeout, mark the
  individual's reward as `-inf` (so ES discards it), log the
  worker_id, restart that worker.  Don't kill the whole run.

- **Worker desync (PySC2 reports `game over` to one binary while
  others are mid-episode).**  Doesn't happen in single-agent /
  vs-bot mode — each binary is its own game.  Documented assumption.

- **`KeyboardInterrupt` in the main process.**  Existing loops
  swallow it (`except KeyboardInterrupt:` at `:914` and `:1137`);
  add a `finally: evaluator.close()` so child binaries are reaped.

- **Reproducibility.**  Each worker needs an independent RNG seed
  to avoid all binaries playing the same map sample.  Derive
  `seed_w = base_seed + worker_id` and pass it in the worker init.
  Document that with `n_workers > 1` the per-individual order in
  the (already collapsed) per-generation log is sorted by
  `individual_idx`, not arrival time.

- **`adaptive_mutation` / patience / early-stop.**  Unchanged —
  these read only generation-level fields (`improved`, `champion_reward`).
  No drift.

---

## 7. Testing

Tests go under `tests/test_parallel_eval.py`:

1. **Pure pool wiring** — fake `GameSpec` whose `build_env` returns a
   `DummyEnv` that scores `policy.flat_sum()`; assert
   `ParallelEvaluator(n_workers=4).evaluate(...)` returns the same
   rewards as a serial loop over the same candidates, in
   `individual_idx` order, for an unsorted submission order.
2. **Worker crash** — `DummyEnv.step` raises on a specific
   `individual_idx`; assert that individual gets `-inf`, others
   complete normally, evaluator stays alive for the next call.
3. **Sentinel shutdown** — `close()` joins all workers within
   `worker_warmup_timeout_s`; no zombies (`Process.is_alive()` all
   false).
4. **Determinism with fixed seed** — two evaluators with the same
   seed return identical rewards for identical candidates.
5. **Integration smoke (skipped unless `RUN_SC2_TESTS=1`)** — actual
   `games/sc2/env.py` build with `n_workers=2` on `MoveToBeacon`,
   2 generations, assert run completes and `experiment_data.json`
   has the expected rows.

Update `tests/README.md`: add a section for `test_parallel_eval.py`
and a sentence in the SC2 "what is and isn't tested" paragraph noting
that the parallel evaluator is unit-tested with a fake env but the
PySC2-binary spawn path is only exercised under the opt-in smoke
test.

---

## 8. Docs & changelog

- `CLAUDE.md` — under `## StarCraft 2`, add an "Intra-run parallel
  evaluation" subsection with the new config keys, sizing guidance,
  and a note that this is generation-synchronous, not
  per-individual async.
- `CHANGELOG.md` — under `## [Unreleased]`, an entry like:
  `**SC2**: population-based policies (sc2_genetic, sc2_cmaes,
  sc2_lstm, sc2_cnn) can now evaluate individuals in parallel across
  multiple local SC2 binaries via the new \`n_workers\` training param
  (issue #229).`

---

## 9. Out of scope (defer to follow-up issues)

- True asynchronous ES (workers post results as they finish, the
  update step doesn't wait for the generation).  Worth its own issue
  with a literature review (async-CMA-ES, A3C-style aggregators).
- Cross-host intra-experiment parallelism.  The persistent-worker
  IPC above is local-only.  Promoting it to HTTP + the existing
  coordinator is a follow-up; the local design intentionally doesn't
  pre-empt that.
- GPU sharing for `sc2_cnn` rollouts — currently CPU numpy; if/when
  the CNN moves to a GPU framework, worker count will be bounded by
  VRAM rather than RAM.
- TMNF parallel workers — TMNF is Windows-bound and TMInterface
  already has a one-game-per-instance assumption; out of scope here.

---

## 10. Implementation order (single PR-sized chunks)

1. `framework/parallel_eval.py` + tests 1–4 (no game code touched
   yet; this is just a pool over a callable).
2. Wire into `_greedy_loop_cmaes` and `_greedy_loop_genetic`
   (gated by `n_workers > 1`).  Verify serial behaviour byte-for-byte
   unchanged at `n_workers=1`.
3. Wire into `SC2LSTMEvolutionPolicy` and `SC2CNNEvolutionPolicy`
   inline loops.
4. SC2 smoke test (test 5), CLAUDE.md + CHANGELOG entries.
5. (Optional, after dogfooding on a real run) — add a `--workers N`
   CLI flag to `main.py` and `grid_search.py` that overrides the
   config value, so quick experimentation doesn't need a config
   edit.

Each step is independently revertable and individually testable.
