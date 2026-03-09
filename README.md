# Bitbucket PR Auto-Review Bridge

Automatically reviews pull requests on Bitbucket Server using the GitHub Models API. Reviews trigger only for configured usernames.

## How it works

Bitbucket Server sends a webhook on PR events. The bridge fetches the diff, sends it to the GitHub Models API for review, and posts findings back as inline comments plus a summary comment on the PR.

## Setup

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```
BITBUCKET_URL=https://bitbucket.company.com
BITBUCKET_TOKEN=...
BITBUCKET_WEBHOOK_SECRET=...   # optional, for webhook HMAC validation
GITHUB_TOKEN=...               # GitHub fine-grained access token with models:read scope
REVIEW_ALLOWED_AUTHORS=jan.username,other.user
```

Only PRs by authors listed in `REVIEW_ALLOWED_AUTHORS` will be reviewed; all others are ignored.

## Run

```bash
podman build -t nitpick -f Containerfile .
podman run -p 8080:8080 --env-file .env \
  -v ./prompts:/app/prompts:ro \
  nitpick
```

Or with Compose:

```yaml
services:
  nitpick:
    build:
      context: .
      dockerfile: Containerfile
    ports:
      - "8080:8080"
    env_file: .env
    volumes:
      - ./prompts:/app/prompts:ro
    restart: unless-stopped
```

```bash
docker compose up -d
```

## Configure Bitbucket webhook

Point a PR webhook at `http://<host>:8080/webhook` with events: `pr:opened`, `pr:modified`.

## Test

```bash
python -m pytest tests/ -v
```

## Customization

Edit `prompts/review.txt` to change the review focus, tone, or output format. The `{diff}` placeholder is replaced with the actual diff at runtime. The prompts directory is mounted into the container, so changes take effect on the next review without rebuilding.
