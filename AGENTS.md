# AGENTS.md

Noergler is a Bitbucket Server PR auto-review bridge backed by an OpenAI-compatible LLM endpoint.

## Testing

- Run tests with `.venv/bin/python -m pytest`.
- Mock HTTP with `respx`; use `unittest.mock.AsyncMock` for async unit mocks.
- Do not call live services from tests.

## Type checking

- Run `.venv/bin/basedpyright` from the repo root. Config lives in `basedpyrightconfig.json`.
- Must be clean (no errors) before committing.

## Database

- Schema changes are managed through Alembic revisions in `alembic/versions/`.
- For any table or column change, add a new migration file; do not edit existing revisions. (Exception already taken: the multi-team change squashed 001..010 into one `001` on the decision that no pre-team database survives; there is no upgrade path from the old chain.)
- The OpenShift init container runs `alembic upgrade head` during deployment.

## Teams (`app/config.py`, `teams.yaml`)

- One instance, N teams. Shared: Bitbucket account, Jira user, DB, gateway + catalog, prompt templates. Per team: inference key, webhook secret, projects, review knobs, Jira prefixes, riptide.
- **Team identity = webhook path + that team's HMAC secret + ownership check.** Never trust `project.key` from the payload alone. Store the authenticated slug (`pr_reviews.team_slug`).
- **One team's fault disables that team only.** Never let a per-team error abort startup; never let a shared-layer error (DB, Bitbucket, Jira, unusable `teams.yaml`) disable just one team.
- Every WARNING/ERROR about a team carries `team=<slug>` via `structlog.contextvars`. Bind at each boundary (webhook route, queue worker, BackgroundTasks handlers via `Reviewer._bind_team`, per-team startup check); the request middleware clears contextvars before BackgroundTasks run.
- Secrets never in `teams.yaml`; `*_env` fields name env vars. `base_url`, `catalog_url` and the prompt templates are instance-only by decision, `extra="forbid"` enforces it.
- Keep the single review worker and single inference lock per client; they protect the shared Bitbucket/Jira accounts. Do not add a per-team worker.

## Riptide emission (optional sink, `app/riptide_client.py`)

- **Best-effort, always.** A slow, rejecting or absent riptide must never fail a webhook, block a review, or change the merge path. `_post` swallows everything and logs.
- **One rollup per PR**, at its terminal outcome (merged / declined / deleted). riptide dedups on `(pr_key, outcome)`, so retries are safe. Never emit PR lifecycle events — riptide already gets those from Bitbucket.
- **The rollup is claimed before the POST**: `riptide_emitted_at` is stamped by the same statement that reads the snapshot, so a rejected or failed emission is **never retried**. Consequence: a payload riptide's strict schema rejects 422s *every* rollup, not just the one carrying a new field, and those PRs are lost. Deploy the riptide side first when adding a field.
- **Unknown cost → omit `total_cost_usd`.** Never send 0 (understates spend), never drop the rollup (outcome, diff size, tokens and runs need no price). Log a warning naming the models so missing pricing stays visible.
- **Declare our own identity** — `reviewer_handle` (`BITBUCKET_USERNAME`) + `reviewer_account_kind`. riptide keeps no bot names of its own; undeclared, our review comments count as a human's and its review-pickup metric collapses to seconds.
- **PR diff size rides on the rollup.** Bitbucket webhooks carry no diff stats, so riptide has no other source for it.

## Cost handling

- **A `None` cost fails open.** An unpriced model, or a gateway not reporting a cost header, must never block a review. The per-PR cost cap skips only *subsequent* auto-runs; the run that overshoots completes.
- The disagree / feedback mechanic was **removed deliberately** (former migration `009`, now folded into the squashed `001`) — replies flagged findings the reviewer could not have prevented. Don't reintroduce it without asking.

## Review prompt layout (`prompts/review.txt`)

- Keep `{files}` BEFORE `{cumulative_pr_diff}` and `{previously_posted_findings}`. Cumulative diff grows on every push and findings accumulate, so they break the endpoint's prefix-cache; files are the most likely bytes to be unchanged across re-reviews of the same PR and must sit in the cached prefix.
- File ordering passed to the LLM must stay content-independent (see `sort_files_by_language_priority` in `app/diff_compression.py`). Do not reintroduce token-count or file-set-derived language priority — the cache breaks if file order shifts when content changes.
