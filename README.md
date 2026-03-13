<p>
  <img src="logo.png" alt="noergler" width="360">
</p>

[![Tests](https://github.com/trick77/noergler/actions/workflows/test.yml/badge.svg)](https://github.com/trick77/noergler/actions/workflows/test.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

AI-powered code review bridge for self-hosted Bitbucket Server. The name is German for "Nörgler" (grumbler/complainer).

Brings automated AI code review to on-premise Bitbucket Server installations. Receives PR webhooks, sends diffs to the GitHub Models API, and posts findings back as inline comments plus a summary comment on the PR.

![noergler inline review comment](review.png)

## How it works

1. **Webhook** — Bitbucket Server fires a `pr:opened` or `pr:modified` event to the `/webhook` endpoint. The request is validated via HMAC-SHA256.
2. **Diff fetch** — The full PR diff is fetched from Bitbucket, split by file, and filtered (binary files, non-reviewable extensions, and files exceeding the configured line limit are skipped).
3. **Context enrichment** — Full file content is fetched for each reviewable file so the AI has complete context, not just the diff. If an `AGENTS.md` file exists in the repository root, it is loaded and included as project-specific review guidelines.
4. **AI review** — Files are grouped into token-aware chunks and sent to the GitHub Models API with a structured review prompt. Each chunk is reviewed independently.
5. **Post results** — Findings are deduplicated against existing noergler comments, sorted by severity (errors first), capped at the configured limit, and posted as inline comments. A summary comment is added to the PR.

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

All configuration is driven by environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `BITBUCKET_URL` | Yes | — | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Yes | — | Bitbucket Server API token |
| `BITBUCKET_WEBHOOK_SECRET` | Yes | — | Webhook HMAC secret for signature validation |
| `GITHUB_TOKEN` | Yes | — | GitHub fine-grained access token with `models:read` scope |
| `REVIEW_AUTO_REVIEW_AUTHORS` | No | _(empty)_ | Comma-separated list of Bitbucket usernames whose PRs are automatically reviewed. When empty or unset, all PR authors are reviewed. Mention-triggered reviews (`@noergler review`) bypass this check. |
| `REVIEW_MAX_COMMENTS` | No | `25` | Maximum inline comments per review |
| `REVIEW_DIFF_EXTRA_LINES_BEFORE` | No | `3` | Context lines to add before each diff hunk (asymmetric: more before than after aids comprehension) |
| `REVIEW_DIFF_EXTRA_LINES_AFTER` | No | `1` | Context lines to add after each diff hunk |
| `REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT` | No | `8` | Max additional lines to search backwards for enclosing function/class scope |
| `REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT` | No | `true` | Enable dynamic scope expansion (extends context to include enclosing function/class definition) |
| `REVIEW_PROMPT_TEMPLATE` | No | `prompts/review.txt` | Path to the review prompt template |
| `REVIEW_MENTION_TRIGGER` | No | `noergler` | Trigger name for mention-based interactions (used as `@<trigger>` in PR comments) |
| `REVIEW_MENTION_PROMPT_TEMPLATE` | No | `prompts/mention.txt` | Path to the mention Q&A prompt template |
| `COPILOT_MODEL` | No | `openai/gpt-5` | Model ID for the GitHub Models API |
| `COPILOT_API_URL` | No | `https://models.github.ai/inference/chat/completions` | GitHub Models API endpoint |
| `COPILOT_MAX_TOKENS_PER_CHUNK` | No | `80000` | Max tokens per diff chunk sent to the model |
| `SERVER_HOST` | No | `0.0.0.0` | Host to bind the server to |
| `SERVER_PORT` | No | `8080` | Port to bind the server to |

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
  main.py          # FastAPI app, /webhook and /health endpoints
  reviewer.py      # Review orchestrator (diff → AI → comments)
  copilot.py       # GitHub Models API client, token-aware chunking
  context_expansion.py  # Asymmetric & dynamic diff context expansion
  bitbucket.py     # Bitbucket Server REST API client
  models.py        # Pydantic models (webhook payloads, findings)
  config.py        # Environment-based configuration
prompts/
  review.txt       # Review prompt template
  mention.txt      # Mention Q&A prompt template
openshift/         # OpenShift/K8s deployment manifests
certs/             # Custom CA certificates (optional)
tests/             # pytest test suite
```

