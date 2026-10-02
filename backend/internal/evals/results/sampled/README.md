# Sampled eval results

An experiment, not a series. A sampled run reviews every corpus case several
times with the identical prompt and scores what merging those reviews would
have posted. The question: misses and invented findings vary from run to run
on an unchanged prompt, so do several reviews merged by a fixed rule beat one,
and at what cost.

Nothing here is a row in `../README.md`. The shape differs and the numbers are
means over groups of samples, not one run's count.

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... \
      go run ./cmd/evals -context-window 1000000 -samples 5 -timeout 60m \
      -json internal/evals/results/sampled/<date>-<model>-<effort>-s5.json

A case's first sample runs alone, the rest after it with `-parallel` in
flight (default 3): calls sent together would all miss the prefix cache.

## Views

Two findings are the same when they are on the same file within 2 lines. File
and line only: no text comparison, no model.

| View | Reviews | Kept |
|---|---|---|
| `single` | 1 | everything: today's review |
| `agree2` | 2 | what both report |
| `either2` | 2 | what either reports |
| `vote3` | 3 | what 2 of 3 report |
| `tiered3` | 3 | what 2 of 3 report, plus any lone finding rated `issue` |
| `union3` | 3 | what any reports |
| `adaptive` | 2, a third on disagreement | 2 of 2, else 2 of 3 |

Every view is scored over every group the samples can form (5 samples: 10
pairs, 10 triples), so its numbers are means. The groups share samples: the
precision is the sample count's, not the group count's.

- `caught`, `extra`: what one run is expected to score.
- `all-caught`: chance one run catches every seeded bug.
- `controls-clean`: chance one run reports nothing on any clean control.
- `cost`: against one cold review. A view costs one cold call plus warm ones;
  `unpriced` when the gateway priced no sample. The token line shows whether
  the later samples hit the cache at all.

The agreement table counts findings by how many of a case's samples
reported them, real against invented. It is what decides between a vote and
a union: `real` at 1 vote is what a vote discards, `invented` at 1 vote is
what a union posts. A vote also loses any bug the model finds less than
half the time, which is the subtle one.

`-rescore sampled.json` recomputes every view from the samples a report
holds, without a call: a merge rule added later is scored on the same
reviews.

Exit 1 judges `vote3`: some triple missed a seeded bug or invented on a
control. Exit 2: a sample did not complete, so the groups are unequal. The
JSON is written either way.

## Union plus one merge call

    go run ./cmd/evals -confirm sampled.json -confirm-groups 10 -json out.json

Takes a finished sampled report, unions its reviews in groups of three and
asks this run's model to consolidate each group in one call: the review
prompt, unchanged, with a brief from `../../briefs/` and the candidates
appended. The answer is validated like any review. Fewer than two candidates
make no call and the group's findings stand.

- `briefs/merge.txt` (default): merge duplicates, drop nothing.
- `briefs/judge.txt`: may also drop what the code refutes. Kept only to
  reproduce its result.

A sample or consolidation call that fails is drawn again, twice at most, and
the count is recorded (`Redraws`). `-cases a,b` limits any run to the named
cases.

## Mixed votes

    go run ./cmd/evals -mix primary.json,secondary.json -json mix.json

Scores one review from the primary run voted with reviews from the secondary
one (the production level once, a cheaper level or model for the rest), from
two finished reports. No calls. On the same model the secondary calls are
priced warm, which assumes a reasoning level does not change the cached
prefix; no run here made a mixed call, so that is not measured.

## Runs

Cells are `caught / extra / controls-clean`. Every run is unpriced: direct
vendor hosts send no cost header. `extra` counts duplicates of a real bug
anchored more than 2 lines apart, and correct findings worded outside a
case's keywords, as well as inventions; the notes say which.

| File | Model | Effort | Cases | single | vote3 | union3 | Notes |
|---|---|---|---|---|---|---|---|
| `2026-10-02-mimo-v2.5-pro-high-s5-rev5` | mimo-v2.5-pro | high | 12 rev5 | 7.40 / 0.20 / 0.80 | 7.30 / 0.00 / 1.00 | 7.90 / 1.80 / 0.40 | Exit 2: 3 of 5 `resource-leak` samples cut at 98 s (`stream idle for 1m30s`: the host sends headers, then thinks past llmwire's 90 s idle bound). Run before redraws existed. Every completed sample caught its bug, so nothing here shows a recall gain. One invented finding, on `clean-guard-clause`, rated `issue`. |
| `2026-10-02-mimo-v2.5-pro-medium-s5-rev5` | mimo-v2.5-pro | medium | 12 rev5 | 7.20 / 0.20 / 0.80 | 8.00 / 0.00 / 1.00 | 8.00 / 1.50 / 0.40 | Complete, 1 redraw. |
| `2026-10-02-gpt-5.4-mini-high-s5-rev5` | gpt-5.4-mini | high | 12 rev5 | 6.80 / 0.40 / 1.00 | 6.80 / 0.60 / 1.00 | 7.70 / 1.50 / 1.00 | Complete. No invention on any control; every `extra` is a duplicate or a keyword miss on `lock-not-released`. |
| `2026-10-02-gpt-5.4-mini-high-s5-rev6` | gpt-5.4-mini | high | 17 rev6 | 10.00 / 0.20 / 1.00 | 10.30 / 0.00 / 1.00 | 11.80 / 1.80 / 1.00 | Complete. `java-seeded` and `resource-leak` caught in 2 of 5 samples: the vote takes both from 0.40 to 0.30, the union to 0.90. Three correct findings dropped by the validator (`evidence_not_found`), not diagnosed. |

Consolidation over the rev6 mini run, 10 groups per case, consolidated by
gpt-5.4-mini high:

| File | Brief | Calls | Caught / extra | Notes |
|---|---|---|---|---|
| `...-rev6-judge` | `judge.txt` | 3.69 | 11.00 / 0.60 | Dropped the real `java-seeded` finding in 4 of the 9 groups that had it. Duplicates merged in 29 of 30 groups. |
| `...-rev6-merge-run1` | `merge.txt` | 3.69 | 11.30 / 0.40 | Before the lone-candidate rule: one `java-seeded` group, handed a single candidate, returned nothing. |
| `...-rev6-merge` | `merge.txt` | 3.62 | 11.40 / 0.40 | With the rule. Matches the union on every case; the remaining 0.40/0.40 is `lock-not-released` wording outside its keywords, one correct finding per group. Consolidation input 92% cached. |

Read together with `../2026-10-02-gpt-5.5-high-5cases.json`: gpt-5.5 high,
one review each on the five cases mini finds hardest, caught 4 of 5 for
about $0.037 a case; three mini reviews plus the merge cost about $0.035 and
expect about 4.8. On these small diffs that is no saving, which is why
production stays single-pass.
