# Eval results

One file per run, committed. A score is only meaningful next to the runs
before it, and an uncommitted number is a number nobody can check.

Name a file `<date>-<model>-<effort>.json` and add a row below. A second run
on the same day at the same settings takes a `-runN` suffix, so it cannot
silently overwrite the first. Record the run even when it is worse than the
last one: a regression you can see is the point of the tool.

Runs are comparable only at the same model and effort, over the same corpus,
**and under the same scorer**. A change to any of the four is a new baseline,
not a data point in the old series. The `Cases` column carries the corpus
revision for that reason. Revisions are numbered, not hashed: this repo
squash-merges, so a branch-local hash stops resolving the moment the work
lands.

The scorer changed once, before run6: byte-identical duplicate findings stopped
counting as Extra, and a finding on a clean control started failing the run.
Only one committed row would read differently under it, run2's Extra 1, which
was that duplicate. The JSONs are left as they are; they record what the tool
emitted at the time.

It changed again with anchored findings. Every finding now passes
`inference.ValidateFindings` before it is scored, the same check production
runs before posting. A dropped finding is neither caught nor Extra, and the
JSON lists it under `Dropped` with its reason, and a moved one under `Moved`.
Duplicates now also compare `evidence`.

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... \
      go run ./cmd/evals -context-window 1000000 \
      -json internal/evals/results/<date>-<model>-<effort>.json

Model and effort default to the pair in `../corpus/series.yaml`
(mimo-v2.6-pro, high). The rows up to 2026-10-01 are the mimo-v2.5-pro
series; the 2.6-pro rows start a new one. `EVAL_MODEL` and
`-effort` score another pair; that is a different series. Production runs a
model too costly to eval on, so this series is a stand-in: a false positive
seen in production may not reproduce here.

`-samples N` and `-mix` produce a different report, an experiment on merging
several reviews. Those live in `sampled/` and are never rows here.

| Date | Model | Effort | Cases | Seeded | Caught | Extra | Prompt | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-20 | mimo-v2.5-pro | high | 3 @ efc7bf8 | 2 | 2 | 0 | `prompts/review.txt` @ d576c13 | First run. Both seeded bugs found, nothing invented on the clean control. Originally recorded as d45d47c, which resolves to nothing: the prompt has exactly one commit in its history and every run so far scored that same text. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev2 | 9 | 8 | 1 | `prompts/review.txt` @ d576c13 | run2. Corpus grown to 11. The miss was corpus modelling, fixed in rev3. The extra is a byte-identical duplicate finding, a model defect; see the notes below. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev3 | 8 | 7 | 1 | `prompts/review.txt` @ d576c13 | run3. `resource-leak` reported at the func declaration, one line above the window. Correct review, scored a miss; windows widened in rev4. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev4 | 8 | 8 | 0 | `prompts/review.txt` @ d576c13 | **run4, baseline.** Clean sweep, nothing invented on the three controls. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev4 | 8 | 7 | 1 | `prompts/review.txt` @ d576c13 | **run5, baseline pair.** Same corpus and settings as run4. One miss: the race was reported correctly but at line 39 of a 37-line file. Model defect, not a corpus bug. |
| 2026-09-21 | mimo-v2.5-pro | high | 11 rev4 | 8 | 7 | 3 | `prompts/review.txt` @ d576c13 | **run6, first under the new scorer.** Exit 1 twice over: `resource-leak` cited line 32 of a 29-line file, and the gate fired on `clean-refactor`, the first control finding in six runs. Corpus untouched; see the notes. |
| 2026-09-30 | mimo-v2.6-flash | high | 12 rev5 | 8 | 7 | 1 | `prompts/review.txt` @ d576c13 | **New series: model and corpus both changed.** Exit 2: `clean-rename` came back unparseable, so the score is incomplete. `clean-i18n-key-order` clean. The miss is `off-by-one` anchored at line 13, one above its 14-20 window. |
| 2026-09-30 | mimo-v2.6-flash | high | 12 rev5 | 8 | 7 | 2 | `prompts/review.txt` @ d576c13 | run2. Exit 1. All four controls clean, `clean-i18n-key-order` again. The miss is `lock-not-released` at line 26, past its 16-24 window. The second Extra is `context-not-propagated` reported as two findings, one per call site (lines 14 and 18), the rev1 shape. |
| 2026-09-30 | mimo-v2.5-pro | high | 12 rev5 | 8 | 6 | 0 | `prompts/review.txt` @ d576c13 | **First rev5 run of the default series**, and the first run at the `corpus/series.yaml` default (no `EVAL_MODEL`, no `-effort`). Exit 2: `context-not-propagated` and `lock-not-released` died on `stream idle for 1m30s`, so both "misses" are transport failures and the score is incomplete. All four controls clean, `clean-i18n-key-order` included. |
| 2026-09-30 | mimo-v2.5-pro | high | 12 rev5 | 8 | 6 | 0 | `prompts/review.txt` @ d576c13 | run2. Exit 1. Two real misses, no finding at all on `off-by-one` or `struct-field-race`, both caught in every rev4 run. All four controls clean. By the baseline rule a 6 is a signal. The prompt is unchanged and rev5 only adds a control, so the change is on the model or gateway side, not in the prompt. |
| 2026-09-30 | mimo-v2.5-pro | high | 12 rev5 | 8 | 7 | 0 | `prompts/review.txt`, anchored-findings WIP | **run3, scored by a validator that never landed.** It accepted evidence only from the diff, so it dropped the correct `resource-leak` finding for quoting `resp, err := c.http.Do(req)` from the full file. The miss is that bug, not the model's; the validator was fixed before commit. Not part of any series. |
| 2026-09-30 | mimo-v2.5-pro | high | 12 rev5 | 8 | 8 | 1 | `prompts/review.txt`, anchored findings | **New baseline: line-numbered diff, required `evidence`, `ValidateFindings` before scoring.** The prompt, the schema and the scorer all changed, so the rows above are not comparable. Exit 1: one finding on `clean-guard-clause` whose own text says "a style refactor with no behavior change". It carries no suggestion, so the no-op check cannot see it. Nothing dropped, nothing moved. |
| 2026-09-30 | mimo-v2.5-pro | high | 12 rev5 | 8 | 7 | 0 | `prompts/review.txt`, anchored findings | run5, baseline pair. Exit 1: `context-not-propagated` got no finding at all. All four controls clean. Nothing dropped, nothing moved, and in both runs every finding landed inside its window. Before this, 3 of the 4 misses on rev5 were a right bug on a wrong line. |
| 2026-10-01 | mimo-v2.5-pro | high | 12 rev5 | 8 | 3 | 0 | `prompts/review.txt`, key-order sentence removed | **Prompt drops step 2's sentence on reading keys by name**: too specific for a prompt that covers every language, and the no-op check still drops a reorder-only fix. Exit 2: the gateway was slow, so `resource-leak` stalled and `unchecked-error` and `wrong-error-wrapped` hit the 10-minute run timeout. Of the 9 that completed: all four controls clean, `clean-i18n-key-order` included, and `context-not-propagated` and `struct-field-race` got no finding. |
| 2026-10-01 | mimo-v2.5-pro | high | 12 rev5 | 8 | 7 | 1 | `prompts/review.txt`, key-order sentence removed | run2, `-timeout 25m`. Exit 2: `resource-leak` stalled again. The Extra is on `clean-i18n-key-order`, but it is not the FR/IT swap: it speculates that a future `AddressType` value could break `addressLabel`, quoting only unchanged context lines, and TypeScript type-checks that key. `struct-field-race` moved from line 35 to 32, its evidence. |
| 2026-10-01 | mimo-v2.5-pro | high | 12 rev5 | 8 | 5 | 0 | `prompts/review.txt`, Example 1 no longer invents `fetch_many` | run3: both prompt edits. Exit 2: `lock-not-released`, `resource-leak` and `struct-field-race` stalled on the gateway, so the score is incomplete. All four controls clean. |
| 2026-10-01 | mimo-v2.5-pro | high | 12 rev5 | 8 | 7 | 0 | `prompts/review.txt`, Example 1 no longer invents `fetch_many` | run4, the first complete run on this prompt. Exit 1, and the miss is the validator's: the correct `resource-leak` finding put its own note `// defer resp.Body.Close() is missing here.` among the quoted lines, which matched nothing and dropped it as `evidence_not_found`. Fixed in the same PR (a note or elision line that matches nothing is skipped, like a blank one); scored before the fix, so it is recorded as it ran. All four controls clean. |
| 2026-10-02 | gpt-5.5 | high | 5 of 17 rev6 (`-cases`) | 5 | 4 | 0 | `prompts/review.txt` @ 89e128c | **Different model, partial corpus: not a row of any series.** The one production-model reference, bought case by case: `java-seeded`, `resource-leak`, `struct-field-race`, `lock-not-released`, `buried-behavioural-hunk`, one review each, $0.19 at list price. `resource-leak` got no finding at all. The file is five one-case runs joined. Read next to `sampled/`. |

## What the columns mean

- **Cases**: corpus size and the revision it was scored at. A row over 3
  cases and a row over 11 are not comparable, and neither are two 11-case
  rows if a window moved between them. `rev1` is the corpus as first grown
  to 11, with one expectation on `context-not-propagated`; `rev2` splits
  that into two, one per `context.Background()` call site; `rev3` folds
  them back into one over a window spanning both; `rev4` starts the four
  whole-function windows at the func declaration; `rev5` adds
  `clean-i18n-key-order`, a fourth clean control, taking the corpus to 12;
  `rev6` adds five harder cases (a cross-file stale caller, a behavioural
  change buried among mechanical hunks, one Java and one TypeScript bug, a
  clean two-file signature change), taking it to 17 with 12 seeded. Bump the number when a
  **scored** field changes: `file`, `lines`, `keywords`, `diff`, `content`.
  Prose does not move a score, so an edit to `description` or `why` is not a
  new revision, and rev4 covers such an edit made after run5.
- **Seeded / Caught**: bugs the corpus planted, and how many the review
  reported inside the line window with a matching keyword.
- **Extra**: findings that pinned no seeded bug. Byte-identical repeats are
  collapsed before counting, so a model stuttering does not read as
  invention; the per-case line reports how many were collapsed. Read the
  notes before reading Extra as a false-positive count; see `EVALS.md`.

## Findings

What the runs showed (the baseline spread, why Extra is not the
false-positive count, how models anchor and mis-cite lines) is in
`EVALS.md` at the repo root. This file keeps the run tables and how to
read their columns.

## Running

- This endpoint lists no `max_input_tokens`, so a run needs
  `-context-window 1000000`. Without it `Startup` fails and the tool exits 2
  rather than reporting a score.
