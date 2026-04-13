<p>
  <img src="logo.png" alt="noergler" width="360">
</p>

[![Tests](https://github.com/trick77/noergler/actions/workflows/test.yml/badge.svg)](https://github.com/trick77/noergler/actions/workflows/test.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

AI-powered code review bridge for self-hosted Bitbucket Server. The name is German for "Nörgler" (grumbler/complainer).

Brings automated AI code review to on-premise Bitbucket Server installations. Receives PR webhooks, sends diffs to the GitHub Models API, and posts findings back as inline comments plus a summary comment on the PR.

![noergler inline review comment](review.png)

## Features

- Automatic AI-powered code review on PR open/modify
- Incremental reviews — only reviews new changes on push, not the entire PR
- Cross-file context analysis — detects when changed symbols are referenced in other PR files
- Mention-based interaction — ask questions or trigger re-reviews via `@noergler` in PR comments
- Smart context enrichment — fetches full file content, not just diffs, for better AI understanding
- Asymmetric and dynamic diff context expansion with language-aware scope detection
- Token-aware chunking and compression for large PRs
- Jira ticket compliance checking against acceptance criteria
- Project-specific review guidelines via `AGENTS.md`
- Comment deduplication against existing review comments
- Feedback collection and usefulness tracking
- HMAC-SHA256 webhook signature validation
- Corporate CA certificate support

For a detailed description of the review pipeline, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## How it works

1. **Webhook** — Bitbucket Server fires a `pr:opened` or `pr:from_ref_updated` event to the `/webhook` endpoint. The request is validated via HMAC-SHA256.
2. **Diff fetch** — On new PRs, the full diff is fetched. On updates, noergler performs an incremental review covering only changes since the last review (falling back to full review after force-pushes).
3. **Context enrichment** — Full file content is fetched for each reviewable file. Diff hunks are expanded with asymmetric context and language-aware scope detection. Cross-file analysis maps changed symbols to their references in other PR files.
4. **AI review** — Files are grouped into token-aware chunks and sent to the GitHub Models API. The prompt includes file content, diffs, cross-file relationships, repo guidelines (`AGENTS.md`), and Jira ticket context.
5. **Post results** — Findings are deduplicated against existing comments, sorted by severity, capped at the configured limit, and posted as inline comments. A summary comment tracks the reviewed commit for incremental reviews.

## Interacting with noergler

Besides automatic reviews on PR open/modify, you can mention noergler in any PR comment:

- **Ask a question** — `@noergler Why was this endpoint changed?` — noergler replies to your comment with an answer based on the PR diff.
- **Trigger a full review** — `@noergler review` — Runs a full review as if the PR was just opened. Also triggered by `@noergler` with no text, `re-review`, or `rereview`.

The mention trigger name defaults to `noergler` and can be changed via `REVIEW_MENTION_TRIGGER`.

## Quick start

1. Copy `.env.example` to `.env` and fill in the required values:

   ```bash
   cp .env.example .env
   ```

2. Build and run with Podman Compose (no pre-built image is published):

   ```bash
   podman compose up -d
   ```

   Or build and run manually:

   ```bash
   podman build -t noergler -f Containerfile .
   podman run -p 8080:8080 --env-file .env \
     -v ./prompts:/app/prompts:ro \
     noergler
   ```

3. [Configure the Bitbucket webhook](#webhook-setup).

## Configuration

All configuration is driven by environment variables. The four required variables are:

| Variable | Description |
|---|---|
| `BITBUCKET_URL` | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Bitbucket Server API token |
| `BITBUCKET_WEBHOOK_SECRET` | Webhook HMAC secret for signature validation |
| `GITHUB_TOKEN` | GitHub fine-grained access token with `models:read` scope |

See [`.env.example`](.env.example) for all optional settings and their defaults.

### Database (optional)

noergler can optionally persist review data to PostgreSQL. Without a database, the app runs normally — it falls back to parsing Bitbucket comment history for features like incremental reviews and deduplication.

To enable persistence, set the `DATABASE_URL` environment variable:

```bash
DATABASE_URL=postgresql://user:password@host:5432/noergler
```

Both `postgresql://` and `postgres://` URI schemes are accepted.

**What gets stored:**

| Table | Purpose |
|---|---|
| `pr_reviews` | Tracks reviewed PRs, last reviewed commit, summary comment IDs |
| `review_findings` | Individual code findings with file, line, severity, and Bitbucket comment ID |
| `review_statistics` | Review metrics — token usage, finding counts, timing |
| `feedback_events` | User feedback on review comments (agree/disagree) |

**Running migrations:**

Schema is managed with Alembic. Run migrations before first use:

```bash
alembic upgrade head
```

If the database is unreachable at startup, noergler logs a warning and continues without persistence.

## Webhook setup

1. Generate a webhook secret:

   ```bash
   openssl rand -hex 32
   ```

2. Set the generated value as `BITBUCKET_WEBHOOK_SECRET` in your `.env` file.

3. In Bitbucket Server, go to **Repository settings > Webhooks > Create webhook**:
   - **URL:** `https://<host>:8080/webhook`
   - **Secret:** paste the same secret from step 1
   - **Events:** `pr:opened`, `pr:modified`, `pr:comment:added`

All webhook requests must include a valid `X-Hub-Signature` header (HMAC-SHA256). Requests with missing or invalid signatures are rejected.

## Customization

### Review prompt

Edit `prompts/review.txt` to change the review focus, tone, or output format. The prompt template uses `{files}` and `{repo_instructions}` as placeholders. The prompts directory is mounted as a volume, so changes take effect on the next review without rebuilding.

### AGENTS.md

Drop an `AGENTS.md` file in the repository root to provide project-specific review guidelines. noergler automatically picks it up from the PR source branch (falling back to the target branch) and injects the content into the review prompt. Use it to tell the reviewer about project conventions, forbidden patterns, or areas to focus on.

## Running tests

```bash
python -m pytest tests/ -v
```

Tests use pytest + pytest-asyncio with `respx` for HTTP mocking. No external services needed. CI runs via GitHub Actions on push and PR.

---

## Deployment notes

### OpenShift

Kubernetes/OpenShift manifests are provided in the `openshift/` directory. See [openshift/README.md](openshift/README.md) for step-by-step instructions.

### Corporate CA certificates

If you're behind a corporate proxy with custom CA certificates, copy `.crt` or `.pem` files into the `certs/` directory before building. They are added to the container's trust store during the build. The directory ships empty so non-corporate builds are unaffected.

## Health check

```
GET /health → {"status": "ok"}
```

## Project structure

```
app/
  main.py              # FastAPI app, /webhook and /health endpoints
  reviewer.py          # Review orchestrator (diff → AI → comments)
  copilot.py           # GitHub Models API client, token-aware chunking
  context_expansion.py # Asymmetric & dynamic diff context expansion
  cross_file_context.py # Cross-file symbol reference analysis
  diff_compression.py  # Large PR compression and file prioritization
  bitbucket.py         # Bitbucket Server REST API client
  jira.py              # Jira ticket fetching and compliance checking
  models.py            # Pydantic models (webhook payloads, findings)
  config.py            # Environment-based configuration
  feedback.py          # Feedback classification
  db/
    pool.py            # asyncpg connection pool management
    repository.py      # Database operations (upsert, query, insert)
prompts/
  review.txt           # Review prompt template
  mention.txt          # Mention Q&A prompt template
openshift/             # OpenShift/K8s deployment manifests
certs/                 # Custom CA certificates (optional)
tests/                 # pytest test suite
```

