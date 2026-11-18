# Ticket-check eval results

Scores what the summary shows for a ticket: the review prompt with a ticket
and the compliance instructions, the answer parsed and passed through
`inference.ConfirmCompliance` as `post` does. A requirement is shown only
when the line the model quoted is in the shown code, and never as unmet.
Findings are not scored. Never a row in `../README.md`: another corpus
(`../../compliance/`), another scorer.

    cd backend
    EVAL_BASE_URL=... EVAL_API_KEY=... \
      go run ./cmd/evals -compliance -samples 3 -parallel 3 -timeout 25m \
      -json internal/evals/results/compliance/<date>-<model>-<effort>.json

A direct mimo Token Plan key works as
`EVAL_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1`.

## Verdicts, per criterion and answer

| Verdict | Meaning | Summary |
|---|---|---|
| `shown` | the shown code implements it, and its quote passed | ✅ line |
| `dropped` | implemented, listed, its quote failed (reason in `drop`) | nothing |
| `unlisted` | implemented, never listed as met | nothing |
| `held_back` | not implemented in the shown code, not shown | nothing (right) |
| `false_shown` | not implemented in the shown code, shown anyway | **a wrong ✅** |

`false_shown` is exit 1. The share of implemented criteria `shown` is the
number to watch: too low and the section is empty everywhere, the check off
in all but name. `unmet` counts requirements the model still listed as not
met against the prompt; the summary never shows them.

Cases are invented. A case marks a criterion implemented only when a line
of the shown code shows it; a criterion met outside the PR (another file,
the pipeline) or not at all is marked not shown.

| Date | Model | Effort | Cases × samples | Implemented shown | Dropped | Unlisted | Wrong ✅ | Unmet listed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | mimo-v2.6-pro | high | 6 × 3 | 17 of 21 | 0 | 4 | 0 of 15 | 0 | **First run, rev1.** Exit 0, no redraw. Every quote the model gave passed `ConfirmCompliance`: the rule cost nothing. All 4 unlisted come from 3 answers that returned an empty list although the code implements the criterion (`german-export-masking`, `met-in-unchanged-code`, `partial-missing-header`, one sample each); the other samples of those cases showed them. Never a wrong ✅: the criterion met only outside the PR (`met-outside-pr` AK-1) held back in 3 of 3, never called unmet; the 429 line was never used for the missing Retry-After header. ~105k prompt and ~26k completion tokens, unpriced. |
