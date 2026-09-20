# Eval results

One file per run, committed. A score is only meaningful next to the runs
before it, and an uncommitted number is a number nobody can check.

Name a file `<date>-<model>-<effort>.json` and add a row below. Record the
run even when it is worse than the last one: a regression you can see is the
point of the tool.

Runs are comparable only at the same model and effort. A change to either is
a new baseline, not a data point in the old series, because a weaker model
scores worse on an unchanged prompt.

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... \
      go run ./cmd/evals -json internal/evals/results/<date>-<model>-<effort>.json

| Date | Model | Effort | Seeded | Caught | Extra | Prompt | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-20 | mimo-v2.5-pro | high | 2 | 2 | 0 | `prompts/review.txt` @ d45d47c | First run. Both seeded bugs found, nothing invented on the clean control. |

## What the columns mean

- **Seeded / Caught**: bugs the corpus planted, and how many the review
  reported inside the line window with a matching keyword.
- **Extra**: findings that pinned no seeded bug. On `clean-refactor` every
  finding is Extra, which is what makes Caught mean something: a model that
  reports everything catches every bug and is useless.

## Notes worth keeping from the first run

- `nil-deref` was found at line **19**, the now-dead `if r.Ticket == nil`
  check, not the dereference at 22. Both are honest anchors for the same
  bug; the window spans both deliberately. A narrower window would have
  scored a correct review as a miss.
- This endpoint lists no `max_input_tokens`, so a run needs
  `-context-window 1000000`. Without it `Startup` fails and the tool exits 2
  rather than reporting a score.
