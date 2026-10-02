# EVALS.md

What the eval experiments found. Rules for running evals stay in `AGENTS.md`;
numbers and run tables stay in `backend/internal/evals/results/`.

## Multi-review experiment (2026-10-02)

Question: misses vary run to run on an unchanged prompt. Do N reviews merged
by a fixed rule beat one?

Tooling, all in `cmd/evals`, none of it a series run:

- `-samples N`: N reviews per case, scored as single / vote / union / adaptive.
- `-confirm sampled.json`: union of three reviews + one consolidation call.
- `-mix`, `-rescore`, `-cases`, `-usage`.
- Results: `backend/internal/evals/results/sampled/`, never a series row.
- First sample alone, the rest after it: sent together they all miss the
  prefix cache.

Findings (gpt-5.4-mini high, 17 cases, 5 samples each):

- **Vote is wrong.** 2-of-3 scored like one review (10.3 vs 10.0 of 12) and
  dropped every bug found in under half the runs.
- **Union works.** 11.8 of 12, no invention on any control. Cost: the same
  bug posted twice, anchored on different lines.
- **File + line cannot merge duplicates.** Same bug, 3-5 lines apart.
- **A model that may drop findings loses real ones.** Keep-by-default judge
  dropped a real bug in 4 of 9 groups.
- **Merge-only call works.** Duplicates into one, drop nothing, add nothing:
  11.8 of 12, no duplicates. Fewer than two candidates = no call, the lone
  finding stands (handed one candidate, the model lost it once).
- **Self-rated severity is no filter.** An invented finding was rated `issue`
  while calling the change behaviour-preserving.
- **gpt-5.5 high misses too.** 4 of 5 on the hardest cases, nothing on
  `resource-leak`. One review each: thin.

Decision: production stays single-pass. Three mini reviews + merge cost about
what one gpt-5.5 review costs on small diffs (~$0.035 a case), and mini's 400K
window is under the 1M requirement. Worth revisiting if real PRs turn out
input-heavy (mini's input price is 6.67x lower) or a cheaper 1M model appears.

A second model merging findings is acceptable; one judging whether a finding
is true is not.

## Known gaps

- `lock-not-released` keywords miss correct findings; inflates `extra`.
- Validator dropped 3 correct mini findings (`evidence_not_found`), not
  diagnosed.
- Direct mimo host: any review thinking past ~98 s dies on llmwire's 90 s
  idle bound (`stream idle for 1m30s`). Not a gateway stall.
