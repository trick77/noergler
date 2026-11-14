# Merge-conflict eval results

Scores the model's part of a merge-conflict resolution
(`prompts/merge_conflict.txt`): the conflicts code leaves over, one call per
case, the answer read by `merge.ReadAnswer` as production reads it. Never a
row in `../README.md`: another prompt, another corpus
(`../../conflicts/`), another scorer.

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... \
      go run ./cmd/evals -conflicts -samples 5 -parallel 4 -timeout 50m \
      -json internal/evals/results/conflicts/<date>-<model>-<effort>.json

`-cases` buys a few cases, `-usage` logs every call's tokens, `-prompt` scores
a draft of the template. A direct mimo Token Plan key works as
`EVAL_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1`: measured
2026-11-14, ~22 s an answer, ~1.1k prompt and ~1.1k completion tokens.

## Verdicts, per answer

| Verdict | Meaning | Production |
|---|---|---|
| `correct` | every conflict resolved to a text the case accepts | pushes it |
| `declined` | `feasible: false` | answers "resolve it yourself" |
| `rejected` | unparseable, or `merge.ReadAnswer` refuses the ids | answers "not safe" |
| `wrong` | a valid answer whose text the case does not accept; any valid answer on a control | **pushes it** |
| `incomplete` | the call failed, redrawn twice | answers "failed" |

`wrong` is the number that counts: a commit nobody checked, with the
dropped lines listed in the thread as its only warning. Exit 1 on any
`wrong`; exit 2 on an `incomplete`. The JSON is written either way.

Text, not ids, is compared: `T2` and an identical `S3` are one answer. A
case lists every text it accepts; `any_order` accepts its lines in any order
(imports). A control accepts nothing.

What the sample size shows: answers to one case are not independent (a
case draws 0 or 5 of 10 wrong), so count cases. n controls with no
`wrong` bound the share of such cases below about 3/n (rule of three,
95 %): 12 controls (rev3), under about 25 %.

## Corpus

Cases by origin (`mined`, `invented`, `web`) and kind (solve, control);
`cmd/conflictmine` drafts mined cases from a clone's merge commits, with
the merge's own file as the truth. Every case is checked at load: within
production's bounds, at least one conflict the model is asked about, every
accepted text writable by ids; and by test, its conflicts exactly what
`merge.Merge3` shows for the versions it holds.

## Runs

| Date | Model | Effort | Cases | Answers | Correct | Declined | Rejected | Wrong | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | mimo-v2.6-pro | high | 3 of rev1 | 6 (2 each) | 3 | 3 | 0 | 0 | Pilot (`-pilot.json`): timing and tokens (5.8 s an answer at 4 in flight). |
| 2026-11-14 | mimo-v2.6-pro | high | 35 rev1 (20 solve, 15 control) | 350 (10 each) | 138 | 185 | 4 | 23 | **First run.** Controls: 6 wrong of 150. 10 wrong keep base lines one branch deleted (`mined-git-1c1256682a`, `web-gitdocs-fruits-both-removed`), 1 keeps a line deleted on the other side (`inv-delete-vs-modify-log`). 12 are rev1 being too strict: `inv-injection-comment` drops the injected comment line only, `mined-external-java-…` merges each version line three-way where the author kept the PR's. See EVALS.md. |
| 2026-11-14 | mimo-v2.6-pro | high | 12 of rev1 | 60 (5 each) | 39 | 14 | 0 | 7 | `-prompt internal/evals/briefs/merge-conflict-base-lines.txt` (`-draft.json`): base-line rule plus abort conditions. The deletion pattern goes from 10 of 20 wrong to 0 of 10; no case that was right lost a correct answer. Left: the injection case (5) and one each of delete-vs-modify and the version case. |
| 2026-11-14 | mimo-v2.6-pro | high | 32 rev2 (19 solve, 13 control) | 320 (10 each) | 149 | 157 | 12 | 1 | **run2, the landed prompt** (base-line rule alone, `-run2.json`). Solvable: 0 wrong of 190 (rev1: 17 of 200). Controls: 1 of 130, `mined-git-42afcb954b` kept the PR's `string_list_sort_u(list, 0)` and lost target's flag 1. The fruits control: 10 answers keep nothing, which `ReadAnswer` rejects (no push). Delete vs modify declined 10 of 10 without its abort clause. rev3 drops `883a47ef64` (10 correct) and `818eb57007` (10 declined): over production's 10 both-changed paths. |
