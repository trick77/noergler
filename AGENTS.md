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
