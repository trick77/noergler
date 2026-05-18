# AGENTS.md

Noergler is a Bitbucket Server PR auto-review bridge backed by a Copilot-enabled LLM.

## Testing

- Run tests with `.venv/bin/python -m pytest`.
- Mock HTTP with `respx`; use `unittest.mock.AsyncMock` for async unit mocks.
- Do not call live services from tests.

## Database

- Schema changes are managed through Alembic revisions in `alembic/versions/`.
- For any table or column change, add a new migration file; do not edit existing revisions.
- The OpenShift init container runs `alembic upgrade head` during deployment.

## Review prompt layout (`prompts/review.txt`)

- Keep `{files}` BEFORE `{cumulative_pr_diff}` and `{previously_posted_findings}`. Cumulative diff grows on every push and findings accumulate, so they break Copilot prefix-cache; files are the most likely bytes to be unchanged across re-reviews of the same PR and must sit in the cached prefix.
- File ordering passed to the LLM must stay content-independent (see `sort_files_by_language_priority` in `app/diff_compression.py`). Do not reintroduce token-count or file-set-derived language priority — the cache breaks if file order shifts when content changes.
