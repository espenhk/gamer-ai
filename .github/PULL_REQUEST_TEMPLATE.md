<!--
One template for all PRs.
Keep only the section(s) that apply to this PR and delete the rest.
-->

## Summary

<!-- 1–3 bullets on what changed and why -->

-
-

## Related issues

<!-- Add links like: "Closes #123" / "Refs #456" -->

## Validation

<!-- Commands or manual checks you ran -->

```bash
PYTHONPATH=. poetry run python -m pytest tests/ \
    --ignore=tests/test_env_termination.py \
    --ignore=tests/test_grid_search.py \
    --ignore=tests/integration/
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
### AI usage

- [ ] No AI was used to write this code
- [ ] AI was used to write/generate some or all of this code

**If AI was used:** Please describe which AI tool(s) and model(s) were used (e.g., Claude Opus, ChatGPT, etc.):

<!-- e.g. "Claude Sonnet 4.6 for initial skeleton + types" -->

**Human involvement:**

<!-- Describe what human review/changes happened: validation of AI output, manual edits, testing, etc. -->

## Screenshots / plots (optional)

<!-- For training-loop, reward, or analytics changes: drop in the
     before/after plots from experiments/<...>/results/. -->

## Notes for the reviewer

- [ ] Scope is limited to this issue/PR
- [ ] Tests updated where needed
- [ ] Docs updated where needed
- [ ] No secrets, large binaries, or experiment outputs committed
