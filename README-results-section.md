## Results

### Held-out evaluation

The agent was frozen (git tag `agent-frozen-for-heldout`, commit `<fill in from meta.json: repodoctor_commit>`) and run once on cases it had never been tuned on. The rules for counting results were written down before the run, and nothing was re-run or re-labelled afterwards.

**Configuration.** `nemotron-3.5-lightning-30b-a3b` explores the code. `nemotron-3.5-super-vl-120b-a12b` writes the reproduction test and the fix. A free Nemotron model on OpenRouter is the fallback (it served 3% of model calls). At most 3 fix attempts and 3 reproduction attempts per run, a 900 s limit per run, and no permission to update existing tests. Run on 2026-11-04.

**Cases.** 23 held-out cases were built by script from four pure-Python repositories (`python-humanize/humanize`, `pallets/click`, `andialbrecht/sqlparse`, `prettytable/prettytable`). Four were excluded before the run because the maintainers' tests require names or exact output the issue does not state (`click-3740`, `prettytable-200`, `prettytable-342`, `sqlparse-701`). That leaves 19 cases, run three times each (57 runs). One of them (`click-2012`) is tagged `[scope]`: its tests need the fix to generalize beyond the reported instance. It stays in the headline.

| Metric | Value |
|---|---|
| Resolved runs | **37 / 57 (65%)** |
| Without the `[scope]` case | 37 / 54 (69%) |
| Excluding the 4 runs that never reached the agent (install failed or timed out; decided after seeing the results) | 37 / 53 (70%) |
| Cases solved in at least one of three trials | 14 / 19 |
| Cases solved in all three trials | 9 / 19 |
| Precision (resolved / runs where the agent reported success) | 37 / 45 (82%) |
| Rough 95% interval for the resolve rate, with the 19 cases as the sampling unit | about 45% to 85% |
| Model calls per run (mean) | 12 |
| Time per run (median / mean) | 142 s / 285 s |

A run counts as resolved only if every fail-to-pass and pass-to-pass test from the maintainers' pull request passes on the agent's final code, with the maintainers' test files copied over the agent's tree.

By repository:

| Repository | Resolved runs |
|---|---|
| humanize | 3 / 6 |
| click | 10 / 18 |
| sqlparse | 14 / 18 |
| prettytable | 10 / 15 |

By case:

| Case | Resolved | Other outcomes |
|---|---|---|
| humanize-333 | 0 / 3 | install failed, wrong_fix x2 |
| humanize-152 | 3 / 3 | |
| click-3822 | 3 / 3 | |
| click-2869 | 2 / 3 | fix_failed |
| click-3571 | 0 / 3 | wrong_fix, install timed out x2 |
| click-3242 | 2 / 3 | install timed out |
| click-3487 | 3 / 3 | |
| click-2012 `[scope]` | 0 / 3 | wrong_fix x2, crash |
| sqlparse-601 | 3 / 3 | |
| sqlparse-332 | 3 / 3 | |
| sqlparse-773 | 2 / 3 | run time limit (during the fix step) |
| sqlparse-752 | 3 / 3 | |
| sqlparse-763 | 3 / 3 | |
| sqlparse-562 | 0 / 3 | fix_failed x3 |
| prettytable-474 | 3 / 3 | |
| prettytable-113 | 2 / 3 | no_repro |
| prettytable-202 | 2 / 3 | explore_failed |
| prettytable-247 | 0 / 3 | wrong_fix x3 |
| prettytable-259 | 3 / 3 | |

### What the failures show

- **Fixes tend to stop at the reported instance.** The agent writes its own reproduction test from the issue, and that test also sets how far the fix goes. In `click-3571`, the tests for the `iterate` path pass and the three for the `update` path fail, while the issue's example only iterates. In `prettytable-247` the same three tests fail in all three trials, and their names suggest the reverse direction of the reported override. In `click-2012` the same three tests fail every time. Earlier development cases showed the same pattern. I have not read all of these tests in full, so this is an interpretation of test names and attempt histories. Eight runs ended this way (`wrong_fix`), and the same failing tests came back across trials, so these are systematic and not luck.
- **One fix broke documented behavior.** In `humanize-333` both completed trials failed a test asserting that a too-large finite value still raises `OverflowError`.
- **One case is hard.** `sqlparse-562` failed all three trials within the attempt budget.
- **Infrastructure and robustness problems cost 6 runs.** Four runs never reached the agent: one dependency install failed and three took about 906 s, which matches the install timeout, and were mislabelled as run-time-limit failures by a bug in the harness. The cause is probably a network stall during install, but I did not confirm it. One run crashed because the model returned no structured output and the code did not handle that case, after a single model call that lasted about half an hour. One more run hit the time limit during a very slow fix call. These are defects in the harness and agent, not reasoning failures, and they are counted as failures above.

### How much to trust these numbers

- **The sample is small.** Nineteen cases from four repositories chosen by a script. The interval above is a hand-computed normal approximation and is only a rough guide.
- **It is not comparable to leaderboards.** The scaffold, the models and the cases are different. Compare results from this harness only with each other.
- **The grader is strict.** It checks the behavior the maintainers' tests encode, so a valid alternative fix can be marked wrong. Cases whose tests require undisclosed names or output were excluded before the run, but borderline ones remain.
- **Possible memorization.** Older public issues may be partly known to the models.
- **The development-set numbers were higher** (about 80% on the cases that were not excluded as unfair, in the last development run). Those cases were used to tune the agent and some were removed after seeing failures, so that number is optimistic by construction. The held-out number is the one to quote.
- **This is one run of one version.** Anything changed after the freeze has to be measured on new cases, not on these.

### Reproducing

```powershell
git checkout agent-frozen-for-heldout
uv sync
uv run repodoctor eval-run --split heldout --trials 3 --label heldout-final
uv run repodoctor eval-failures .repodoctor\evals\<run folder>\results.jsonl
```
Results depend on the live models and on GitHub and PyPI being reachable, so an exact repeat is not guaranteed.