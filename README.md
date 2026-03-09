# Nitpick

[![Tests](https://github.com/trick77/nitpick/actions/workflows/test.yml/badge.svg)](https://github.com/trick77/nitpick/actions/workflows/test.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

AI-powered code review bridge for Bitbucket Server. Receives PR webhooks, sends the diff to the GitHub Models API for review, and posts findings back as inline comments plus a summary.

## Setup

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `BITBUCKET_URL` | Yes | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Yes | Bitbucket Server API token |
| `BITBUCKET_WEBHOOK_SECRET` | No | Webhook HMAC secret for signature validation |
| `GITHUB_TOKEN` | Yes | GitHub fine-grained access token with `models:read` scope |
| `REVIEW_ALLOWED_AUTHORS` | Yes | Comma-separated list of Bitbucket usernames to review |

Only PRs by authors listed in `REVIEW_ALLOWED_AUTHORS` will be reviewed; all others are ignored.

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

Point a PR webhook at `http://<host>:8080/webhook` with events: `pr:opened`, `pr:modified`.

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
