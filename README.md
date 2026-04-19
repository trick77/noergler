<p>
  <img src="logo.png" alt="noergler" width="360">
</p>

[![Tests](https://github.com/trick77/noergler/actions/workflows/test.yaml/badge.svg)](https://github.com/trick77/noergler/actions/workflows/test.yaml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

AI-powered code review bridge for self-hosted Bitbucket Server. The name is German for "Nörgler" (grumbler/complainer).

Brings automated AI code review to on-premise Bitbucket Server installations. Receives PR webhooks, sends diffs to an OpenAI-compatible LLM API, and posts findings back as inline comments plus a summary comment on the PR.

![noergler inline review comment](review.png)

## Features

- Automatic AI-powered code review on PR open/modify
- Incremental reviews — only reviews new changes on push, not the entire PR
- Cross-file context analysis — detects when changed symbols are referenced in other PR files
- Mention-based interaction — ask questions or trigger re-reviews via `@noergler` in PR comments
- Smart context enrichment — fetches full file content, not just diffs, for better AI understanding
- Asymmetric and dynamic diff context expansion with language-aware scope detection
- Token-aware chunking and compression for large PRs
- Prompt-cache-optimised template layout — stable rules and examples first, per-PR variables (tone, ticket, diff) last, so repeated reviews maximise LLM prompt-cache reuse and the diff lands where model recall is strongest
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
4. **AI review** — Files are grouped into token-aware chunks and sent to the configured LLM API (any OpenAI-compatible endpoint). The prompt includes file content, diffs, cross-file relationships, repo guidelines (`AGENTS.md`), and Jira ticket context.
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

All configuration is driven by environment variables. The required variables are:

| Variable | Description |
|---|---|
| `BITBUCKET_URL` | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Bitbucket Server API token |
| `BITBUCKET_WEBHOOK_SECRET` | Webhook HMAC secret for signature validation |
| `BITBUCKET_USERNAME` | Bitbucket service account username (used to identify bot comments) |
| `COPILOT_OAUTH_TOKEN` | Long-lived GitHub OAuth token for a Copilot Business seat (provision via `hack/copilot-provision-token.sh`) |
| `JIRA_URL` | Jira Server/Cloud base URL |
| `JIRA_TOKEN` | Jira API token |
| `DATABASE_URL` | PostgreSQL connection string (see [Database](#database) below) |

See [`.env.example`](.env.example) for all optional settings and their defaults.

### Database

noergler requires PostgreSQL for review state, deduplication, and statistics. The database connection is validated on startup — the app will not start without it.

Create a PostgreSQL database and user, then set `DATABASE_URL`:

```sql
CREATE USER noergler WITH PASSWORD 'changeme';
CREATE DATABASE noergler OWNER noergler;
```

```bash
DATABASE_URL=postgresql://noergler:changeme@localhost:5432/noergler
```

Both `postgresql://` and `postgres://` URI schemes are accepted.

> **Tip:** In production, inject `DATABASE_URL` via container or orchestrator secrets (e.g. Kubernetes Secrets, `--env-file`) rather than storing credentials in plaintext.

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

## Webhook setup

1. Generate a webhook secret:

   ```bash
   openssl rand -hex 32
   ```

2. Set the generated value as `BITBUCKET_WEBHOOK_SECRET` in your `.env` file. This same value must be configured on Bitbucket's side — the service and Bitbucket share the HMAC secret.

### Automated onboarding (recommended)

Use `scripts/onboard_repo.py` to create/update the webhook on one or more repos in a single, idempotent command. It verifies token permissions, reconciles the webhook configuration, and triggers Bitbucket's built-in connectivity test.

```bash
python -m scripts.onboard_repo config.json [--name noergler] [--dry-run] [--env-file PATH]
```

Example `config.json` — repos are grouped under their Bitbucket project:

```json
{
  "bitbucket_url": "https://bitbucket.example.com",
  "webhook_url": "https://noergler.internal/webhook",
  "projects": [
    {
      "project": "PROJ",
      "repos": ["payments-service", "ledger-api"]
    },
    {
      "project": "PLATFORM",
      "repos": ["auth-gateway"]
    }
  ]
}
```

**Credentials.** Secrets are never read from the JSON. `BITBUCKET_TOKEN` (bearer, needs repo read + write on every listed repo) and `BITBUCKET_WEBHOOK_SECRET` are resolved from, in order: process environment → `.env` in the current directory → `--env-file <path>`. Process env wins on conflict. `BITBUCKET_WEBHOOK_SECRET` **must match** the value the running noergler service is using — this script only programs Bitbucket's side; it does not generate a new secret.

**Flags.**
- `--name` — webhook name (default `noergler`). Used to find an existing hook to update instead of creating a duplicate.
- `--dry-run` — print the create/update diff and the body that would be sent, without mutating Bitbucket.
- `--env-file PATH` — additional env file for CI (`/run/secrets/noergler.env` etc.).

**Behaviour.** Each repo runs through three steps: verify read access, create-or-update the webhook (idempotent — re-running prints `already up to date`), then trigger Bitbucket's test endpoint and report the downstream status. A failure on one repo logs and continues to the next; the script exits non-zero iff any repo failed, and prints a final summary table.

**Known limitation — secret-only drift.** Bitbucket's webhook API does not return the stored secret, so the script cannot detect when *only* the secret has changed on one side. If you rotate `BITBUCKET_WEBHOOK_SECRET` on the service side, delete the webhook in Bitbucket (or rename it so this script recreates it) before re-running — otherwise the script will report `already up to date` while Bitbucket continues signing with the old secret.

### Manual setup (fallback)

In Bitbucket Server, go to **Repository settings > Webhooks > Create webhook**:
- **URL:** `https://<host>:8080/webhook`
- **Secret:** the value of `BITBUCKET_WEBHOOK_SECRET`
- **Events:** `pr:opened`, `pr:from_ref_updated`, `pr:comment:added`, `pr:merged`, `pr:deleted`

All webhook requests must include a valid `X-Hub-Signature` header (HMAC-SHA256). Requests with missing or invalid signatures are rejected.

## Customization

### Review prompt

Edit `prompts/review.txt` to change the review focus, tone, or output format. The prompt template uses `{files}` and `{repo_instructions}` as placeholders. The prompts directory is mounted as a volume, so changes take effect on the next review without rebuilding.

### Prompt layout

Both `prompts/review.txt` and `prompts/mention.txt` are ordered deliberately:

1. **Stable prefix** — role, rules, output format, examples, and prompt-injection guardrails. No placeholders, identical on every call.
2. **Per-repo block** — `{repo_instructions}` (the `AGENTS.md` content). Stable across all PRs in the same repo.
3. **Per-PR block** — `{tone}`, `{ticket_context}`, and finally `{files}` (or `{diff}` + `{question}` in the mention template).

Three reasons this matters, and they all push the same layout:

- **Prompt cache reuse** — OpenAI-compatible prompt caching matches on the longest stable prefix. Keeping all variable content at the tail means the entire stable portion is served from cache on every subsequent call.
- **"Lost in the middle"** — long-context LLMs recall the start and end of a prompt better than the middle. Putting the diff (the thing the model must reason about) at the very end of the context gives it the strongest recall.
- **KV-cache eviction on very long contexts** — some long-context implementations evict middle tokens first under pressure. Stable rules at the top are cheap to lose; the diff at the bottom stays intact.

If you edit the templates, preserve this ordering: keep new stable rules above the guardrail section, and any new per-PR placeholders below it, right before `{files}` / `{diff}`.

### AGENTS.md

Drop an `AGENTS.md` file in the repository root to provide project-specific review guidelines. noergler automatically picks it up from the PR source branch (falling back to the target branch) and injects the content into the review prompt. Use it to tell the reviewer about project conventions, forbidden patterns, or areas to focus on.

By default, reviews are **gated** on the presence of `AGENTS.md` — without one, noergler skips the review and posts a short summary comment explaining why. To review PRs without an `AGENTS.md`, set `REVIEW_REQUIRE_AGENTS_MD=false`.

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
  llm_client.py        # OpenAI SDK client for the configured LLM API, token-aware chunking
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

