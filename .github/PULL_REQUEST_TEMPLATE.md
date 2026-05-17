<!--
Thanks for sending a PR! A few quick things before you click "Create":

  - Keep the title short (≤ 70 chars). Use the body for detail.
  - Link the issue this PR closes (e.g. "Closes #123").
  - Tick the checklist below honestly — unchecked boxes are fine, they
    just tell the reviewer what's still missing.
-->

## Summary

<!-- 1–3 bullet points: what changed and why. Focus on the "why". -->

-
-

## Related issues

<!-- "Closes #123", "Refs #456". Use "Closes" so the issue auto-closes on merge. -->

Closes #

## Type of change

<!-- Tick whichever apply. -->

- [ ] Bug fix
- [ ] New feature (no breaking change)
- [ ] Breaking change (existing behaviour or API changes)
- [ ] New game integration (`games/<name>/`)
- [ ] New policy / algorithm
- [ ] Documentation only
- [ ] Refactor / internal cleanup
- [ ] Infrastructure / CI

## How was this tested?

<!-- Be specific. "Ran the training loop for N sims on game X" is much
     more useful than "looks good". Paste commands you ran. -->

```
# e.g. — same unit-test command CI runs:
PYTHONPATH=. poetry run python -m pytest tests/ \
    --ignore=tests/test_env_termination.py \
    --ignore=tests/test_grid_search.py \
    --ignore=tests/integration/

python main.py smoke_test --game car_racing --no-interrupt
```

## Checklist

### Code quality

- [ ] Changes are scoped to the issue — no unrelated refactors mixed in
- [ ] New / changed code follows the surrounding style (no `print` debug noise, no dead code, no commented-out blocks)
- [ ] No accidentally-committed secrets, `.env` files, large binaries, or experiment output under `experiments/`

### Tests

- [ ] Added or updated unit tests for new logic (`tests/`)
- [ ] Cross-platform unit tests pass locally (`PYTHONPATH=. poetry run python -m pytest tests/ --ignore=tests/test_env_termination.py --ignore=tests/test_grid_search.py --ignore=tests/integration/`) — same command the `Tests / test` CI job runs
- [ ] For game-specific changes: ran at least one short end-to-end training run (`python main.py …`) and confirmed it doesn't crash
- [ ] For integration-test-eligible changes (CarRacing / SC2): integration tests are green or there is a clear reason they're skipped

### Documentation

- [ ] Updated `README.md` if user-visible behaviour, install steps, or CLI flags changed
- [ ] Updated `CLAUDE.md` if architecture, config knobs, or training-loop semantics changed
- [ ] Updated the relevant `games/<name>/README.md` for per-game changes
- [ ] Updated `tests/README.md` if tests were added, removed, or substantially changed (required by the repo's test-suite contract)
- [ ] New config keys appear in the relevant `config/*.yaml` master file with a sensible default

### Review

- [ ] Self-reviewed the diff at least once before requesting review
- [ ] Happy for an AI code review (e.g. `/ultrareview` or an automated reviewer) to run on this PR
- [ ] Marked the PR as draft if it isn't ready for human review yet

## Screenshots / plots (optional)

<!-- For training-loop, reward, or analytics changes: drop in the
     before/after plots from experiments/<...>/results/. -->

## Notes for the reviewer

<!-- Anything that would save the reviewer time: known limitations,
     follow-up work, parts of the diff that look weird but are
     intentional, etc. -->
