# Nitpick

[![Tests](https://github.com/trick77/nitpick/actions/workflows/test.yml/badge.svg)](https://github.com/trick77/nitpick/actions/workflows/test.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

AI-powered code review bridge for Bitbucket Server. Receives PR webhooks, sends the diff to the GitHub Models API for review, and posts findings back as inline comments plus a summary.

## Setup

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `BITBUCKET_URL` | Yes | — | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Yes | — | Bitbucket Server API token |
| `BITBUCKET_WEBHOOK_SECRET` | Yes | — | Webhook HMAC secret for signature validation (see below) |
| `GITHUB_TOKEN` | Yes | — | GitHub fine-grained access token with `models:read` scope |
| `REVIEW_ALLOWED_AUTHORS` | Yes | — | Comma-separated list of Bitbucket usernames to review |
| `REVIEW_CONTEXT_LINES` | No | `20` | Lines of context around each diff hunk |
| `REVIEW_MAX_COMMENTS` | No | `25` | Maximum inline comments per review |
| `REVIEW_MAX_LINES_PER_FILE` | No | `1000` | Skip files exceeding this line count in the diff |
| `REVIEW_PROMPT_TEMPLATE` | No | `prompts/review.txt` | Path to the review prompt template |
| `COPILOT_MODEL` | No | `openai/gpt-4.1` | Model ID for the GitHub Models API |
| `COPILOT_API_URL` | No | `https://models.github.ai/inference/chat/completions` | GitHub Models API endpoint |
| `COPILOT_MAX_TOKENS` | No | `80000` | Max tokens per diff chunk sent to the model |
| `SERVER_HOST` | No | `0.0.0.0` | Host to bind the server to |
| `SERVER_PORT` | No | `8080` | Port to bind the server to |

Only PRs by authors listed in `REVIEW_ALLOWED_AUTHORS` will be reviewed; all others are ignored.

### Corporate CA certificates

If you're behind a corporate proxy with custom CA certificates, copy the `.crt` or `.pem` files into the `certs/` directory before building. They will be added to the container's trust store during the build. The directory ships empty so non-corporate builds are unaffected.

## Run

```bash
podman build -t nitpick -f Containerfile .
podman run -p 8080:8080 --env-file .env \
  -v ./prompts:/app/prompts:ro \
  nitpick
```

Or using the included `compose.yaml`:

```bash
podman compose up -d
```

## Configure Bitbucket webhook

1. Generate a webhook secret:

   ```bash
   openssl rand -hex 32
   ```

2. Set the generated value as `BITBUCKET_WEBHOOK_SECRET` in your `.env` file.

3. In Bitbucket Server, go to **Repository settings → Webhooks → Create webhook**:
   - **URL:** `http://<host>:8080/webhook`
   - **Secret:** paste the same secret from step 1
   - **Events:** `pr:opened`, `pr:modified`

All webhook requests must include a valid `X-Hub-Signature` header (HMAC-SHA256). Requests with missing or invalid signatures are rejected.

## Health check

```
GET /health
```

Returns `{"status": "ok"}` when the service is running.

## Test

```bash
python -m pytest tests/ -v
```

## Customization

Edit `prompts/review.txt` to change the review focus, tone, or output format. The `{diff}` placeholder is replaced with the actual diff at runtime. The prompts directory is mounted into the container, so changes take effect on the next review without rebuilding.
