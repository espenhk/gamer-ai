# Plan: Q-Table Persistence for Tabular Policies

## Problem

`EpsilonGreedyPolicy` and `MCTSPolicy` both inherit from `QTablePolicy`. They build up
Q-value estimates (`_q_table`) and visit counts (`_n_sa`, `_n_s`) over many episodes,
but `to_cfg()` (in `policies.py` lines ~550 and ~639) saves **only hyperparameters**
(alpha, gamma, epsilon, etc.) — the Q-table itself is discarded on exit.

Tabular methods need hundreds to thousands of episodes to converge. Losing the table on
every restart means the policy re-explores from scratch each time, making it impractical
for the iterative workflow (`python main.py <experiment>`).

## Proposed Solution

Pickle the three table dicts alongside the YAML config:

- YAML (`policy_weights.yaml`) — human-readable hyperparameters, unchanged
- Pickle (`policy_weights_qtable.pkl`) — binary `(q_table, n_sa, n_s)` tuple

On load, if the pickle exists, restore the tables automatically.

## Changes

> **Important:** `policies.py` is a TMNF backward-compatibility shim. The training
> framework imports `EpsilonGreedyPolicy` / `MCTSPolicy` directly from
> `framework/policies.py`, so all logic changes go there and in
> `framework/training.py`. The `from_cfg()` methods in the shim are currently
> unused by the training loop.

### `framework/policies.py` — `QTablePolicy`: add `save()` + `_load_table()`

```python
import pickle  # add at top of file

MAX_QTABLE_ENTRIES = 500_000  # guard against saving enormous tables

# Inside QTablePolicy:

def save(self, path: str) -> None:
    """Save hyperparams to YAML and Q-table to sibling .pkl file."""
    super().save(path)  # writes YAML via to_cfg()
    if len(self._q_table) > MAX_QTABLE_ENTRIES:
        logger.warning(
            "Q-table has %d entries (>%d), skipping pickle.",
            len(self._q_table), MAX_QTABLE_ENTRIES,
        )
        return
    pkl_path = _qtable_pkl_path(path)
    with open(pkl_path, "wb") as f:
        pickle.dump((self._q_table, self._n_sa, self._n_s), f)
    logger.info("Q-table saved: %d states → %s", len(self._q_table), pkl_path)

def _load_table(self, path: str) -> None:
    """Restore Q-table from sibling .pkl if it exists."""
    pkl_path = _qtable_pkl_path(path)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            q_table, n_sa, n_s = pickle.load(f)
        self._q_table = q_table
        self._n_sa    = n_sa
        self._n_s     = n_s
        logger.info("Q-table loaded: %d states from %s", len(q_table), pkl_path)
```

Add a small helper near the top of the file:
```python
def _qtable_pkl_path(yaml_path: str) -> str:
    base, _ = os.path.splitext(yaml_path)
    return base + "_qtable.pkl"
```

### `framework/training.py` — `_make_policy()`: restore table on restart

After constructing the `epsilon_greedy` and `mcts` policies, add a load call:

```python
elif policy_type == "epsilon_greedy":
    policy = EpsilonGreedyPolicy(
        obs_spec=obs_spec,
        discrete_actions=discrete_actions,
        n_bins=policy_params.get("n_bins", 3),
        epsilon=policy_params.get("epsilon", 1.0),
        epsilon_decay=policy_params.get("epsilon_decay", 0.995),
        epsilon_min=policy_params.get("epsilon_min", 0.05),
        alpha=policy_params.get("alpha", 0.1),
        gamma=policy_params.get("gamma", 0.99),
    )
    if os.path.exists(weights_file) and not re_initialize:
        with open(weights_file) as f:
            saved_cfg = yaml.safe_load(f) or {}
        policy._epsilon = float(saved_cfg.get("epsilon", policy._epsilon))
        policy._load_table(weights_file)
    return policy

elif policy_type == "mcts":
    policy = MCTSPolicy(
        obs_spec=obs_spec,
        discrete_actions=discrete_actions,
        c=policy_params.get("c", 1.41),
        alpha=policy_params.get("alpha", 0.1),
        gamma=policy_params.get("gamma", 0.99),
        n_bins=policy_params.get("n_bins", 3),
    )
    if os.path.exists(weights_file) and not re_initialize:
        policy._load_table(weights_file)
    return policy
```

Note: restoring `epsilon` from the saved YAML is important so decay continues
from where it left off rather than restarting from 1.0.

## File Naming

| File | Contents |
|------|----------|
| `experiments/<name>/policy_weights.yaml` | Hyperparameters (unchanged) |
| `experiments/<name>/policy_weights_qtable.pkl` | `(q_table, n_sa, n_s)` pickle |

The `.pkl` file lives next to the YAML so it's automatically found by `_load_table()`.
Both files should be added to `.gitignore` under `experiments/` (already git-ignored).

## Size Concern

With `n_bins=5` and 15 obs dimensions, the theoretical table size is 5^15 ≈ 30 billion
states — but only **visited** states are stored (sparse dict). In practice, a 100-episode
session on a 13-second track at 10× speed visits ~100 × (13 × 100 steps) = 130,000 unique
state bins at most. The `MAX_QTABLE_ENTRIES = 500_000` guard prevents pathological cases
from producing huge pickle files.

## Files to Change

| File | Change |
|------|--------|
| `framework/policies.py` | Add `save()` + `_load_table()` to `QTablePolicy`; add `_qtable_pkl_path()` helper; add `import pickle` |
| `framework/training.py` | Update `_make_policy()` for `epsilon_greedy` and `mcts`: restore epsilon + call `_load_table()` when weights file exists |

## Testing

1. Run `python main.py my_qtable_exp` with `policy_type=epsilon_greedy` for 50 episodes
2. Confirm `experiments/my_qtable_exp/policy_weights_qtable.pkl` exists
3. Restart: `python main.py my_qtable_exp` — confirm the logged Q-table size matches
   (non-zero number of states loaded)
4. Confirm `epsilon` continues decaying from where it left off (saved in YAML)
5. Verify with `policy_type=mcts` also persists and restores correctly
