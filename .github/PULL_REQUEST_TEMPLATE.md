<!--
One template for all PRs.
Keep only the section(s) that apply to this PR and delete the rest.
-->

## Summary

<!-- 1–3 bullets on what changed and why -->

-
-

## Related issues

<!-- e.g. "Closes #123", "Refs #456" -->

Closes #

## Validation

<!-- Commands or manual checks you ran -->

```bash
PYTHONPATH=. python -m pytest tests/ --ignore=tests/test_env_termination.py --ignore=tests/test_grid_search.py --ignore=tests/integration/
```

## Bug fix details (delete this section if not a bug fix)

- Problem:
- Root cause:
- Fix:

## Feature details (delete this section if not a feature)

- User problem:
- Proposed solution:
- Trade-offs / follow-ups:

## Refactor / docs / chore details (delete this section if not applicable)

- What changed:
- Why this is safe:

## Checklist

- [ ] Scope is limited to this issue/PR
- [ ] Tests updated where needed
- [ ] Docs updated where needed
- [ ] No secrets, large binaries, or experiment outputs committed
