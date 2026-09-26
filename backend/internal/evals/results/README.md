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

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... EVAL_MODEL=<model> \
      go run ./cmd/evals -effort <effort> -context-window 1000000 \
      -json internal/evals/results/<date>-<model>-<effort>.json

Neither model nor effort has a default: take both from the series' rows below
to extend it.

| Date | Model | Effort | Cases | Seeded | Caught | Extra | Prompt | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-20 | mimo-v2.5-pro | high | 3 @ efc7bf8 | 2 | 2 | 0 | `prompts/review.txt` @ d576c13 | First run. Both seeded bugs found, nothing invented on the clean control. Originally recorded as d45d47c, which resolves to nothing: the prompt has exactly one commit in its history and every run so far scored that same text. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev2 | 9 | 8 | 1 | `prompts/review.txt` @ d576c13 | run2. Corpus grown to 11. The miss was corpus modelling, fixed in rev3. The extra is a byte-identical duplicate finding, a model defect; see the notes below. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev3 | 8 | 7 | 1 | `prompts/review.txt` @ d576c13 | run3. `resource-leak` reported at the func declaration, one line above the window. Correct review, scored a miss; windows widened in rev4. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev4 | 8 | 8 | 0 | `prompts/review.txt` @ d576c13 | **run4, baseline.** Clean sweep, nothing invented on the three controls. |
| 2026-09-20 | mimo-v2.5-pro | high | 11 rev4 | 8 | 7 | 1 | `prompts/review.txt` @ d576c13 | **run5, baseline pair.** Same corpus and settings as run4. One miss: the race was reported correctly but at line 39 of a 37-line file. Model defect, not a corpus bug. |
| 2026-09-21 | mimo-v2.5-pro | high | 11 rev4 | 8 | 7 | 3 | `prompts/review.txt` @ d576c13 | **run6, first under the new scorer.** Exit 1 twice over: `resource-leak` cited line 32 of a 29-line file, and the gate fired on `clean-refactor`, the first control finding in six runs. Corpus untouched; see the notes. |

## What the columns mean

- **Cases**: corpus size and the revision it was scored at. A row over 3
  cases and a row over 11 are not comparable, and neither are two 11-case
  rows if a window moved between them. `rev1` is the corpus as first grown
  to 11, with one expectation on `context-not-propagated`; `rev2` splits
  that into two, one per `context.Background()` call site; `rev3` folds
  them back into one over a window spanning both; `rev4` starts the four
  whole-function windows at the func declaration. Bump the number when a
  **scored** field changes: `file`, `lines`, `keywords`, `diff`, `content`.
  Prose does not move a score, so an edit to `description` or `why` is not a
  new revision, and rev4 covers such an edit made after run5.
- **Seeded / Caught**: bugs the corpus planted, and how many the review
  reported inside the line window with a matching keyword.
- **Extra**: findings that pinned no seeded bug. Byte-identical repeats are
  collapsed before counting, so a model stuttering does not read as
  invention; the per-case line reports how many were collapsed. Read the
  notes before reading Extra as a false-positive count; see below.

## The baseline

run4 and run5 are the pair. Same prompt, same model, same effort, same
corpus revision, **8 caught and 7 caught**. That spread is the baseline: a
single later run scoring 7 is inside the noise, and only a run scoring 6 or
less, or a second consecutive 7 after a prompt edit, is a signal.

Every run scored the same prompt text: `prompts/review.txt` has one commit in
its entire history. Every difference between these rows is model
non-determinism, a corpus change or a scorer change, never a prompt change.

**Exit 1 has fired on two of the three rev4 runs with the prompt unchanged.**
It therefore does not mean "the prompt regressed"; it means the model had an
off run, and the cause is named in the row. Treat a single red as noise and a
second consecutive one as the signal, which is the same rule the caught count
already follows. Nothing in the tool enforces that: the exit code stays strict
so a CI job could gate on it later without a threshold to tune.

The three clean controls scored **0 findings in runs 1 through 5** (run1
predates two of them and had only `clean-refactor`). **run6 broke that**: one
finding on `clean-refactor`, the first in six runs, which is what the new gate
exists to catch. The controls are the honest false-positive floor and are what
makes Caught mean anything, so the run is red and stays red in the record.

## Extra is not the false-positive count

Extra has been inflated on a seeded case twice, neither time by invention:

- **`lock-not-released`, run2**: the model emitted the same finding twice,
  byte for byte, both at line 22. The duplicate pins nothing new and counts
  Extra. run5's single Extra is not this; it is the line-39 miss below.
- **`context-not-propagated`, on rev1**: the model reported the two
  `context.Background()` substitutions as two separate findings. rev1 had
  one expectation, which absorbs one of them, so the second honest finding
  scored Extra. rev2 split the expectation in two, and run2 then scored a
  *miss*, because that time the model emitted one finding naming both call
  sites. Both shapes are correct reviews, so rev3 settled on one
  expectation over a window spanning both. That rev1 run is not in this
  directory: it was scored against a corpus state no committed row uses,
  and keeping it would have implied a comparison it cannot support.

Neither is a false positive. The clean controls are.

run6's third Extra is the control finding, and it is counted as invention on
purpose. Its text argues the refactor is *correct* ("No bug is introduced by
the extraction"), which is tempting to excuse. It is not excused: production
posts `severity: suggestion` findings, `severityOrder` only sorts them below
`issue` and nothing drops them, so that paragraph would have been a comment on
the PR. Filtering it out by reading what it says is the judge model the
bluntness rule forbids.

## Notes worth keeping

- `nil-deref` was found at line 19 in runs 1, 2 and 5, at 22 in run3 and at
  20 in run4:
  the dead `if r.Ticket == nil` check, the comment inside it, and the
  dereference. All are honest anchors for the same bug, which is why the
  window spans them. A narrower window scores a correct review as a miss.
- **A whole-function defect is anchored at the func declaration** about as
  often as at the first statement. run3 lost `resource-leak` that way. The
  four whole-function windows now start at the declaration; the
  narrow-statement cases do not, because widening `unchecked-error` to its
  func line would span the entire body and stop being falsifiable.
- **The model can cite a line past the end of the file.** Twice in six runs,
  on two different cases: run5 put the data race at line 39 of a 37-line
  file, run6 put the body leak at line 32 of a 29-line one. Both reviews were
  correct; no window can catch either, and widening one to reach a
  nonexistent line would only make the case unwinnable in the other
  direction. run6's finding also cites "line 24" for a call on line 17 in its
  own prose, so the number is unreliable rather than merely offset.
- This endpoint lists no `max_input_tokens`, so a run needs
  `-context-window 1000000`. Without it `Startup` fails and the tool exits 2
  rather than reporting a score.
