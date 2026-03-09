# Bitbucket PR Auto-Review Bridge

Automatically reviews pull requests on Bitbucket Server using the GitHub Copilot chat completions API. Reviews trigger only for configured usernames.

## How it works

Bitbucket Server sends a webhook on PR events. The bridge fetches the diff, sends it to the Copilot API for review, and posts findings back as inline comments plus a summary comment on the PR.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set required environment variables:

```bash
export BITBUCKET_TOKEN="..."
export BITBUCKET_WEBHOOK_SECRET="..."
export GITHUB_TOKEN="..."   # GitHub Copilot token
```

Edit `config.yaml` to set your Bitbucket base URL, Copilot model, and allowed PR authors:

```yaml
bitbucket:
  base_url: "https://bitbucket.company.com"
  token: "${BITBUCKET_TOKEN}"
  webhook_secret: "${BITBUCKET_WEBHOOK_SECRET}"

copilot:
  model: "gpt-4.1"
  github_token: "${GITHUB_TOKEN}"

review:
  allowed_authors:
    - "your.username"
```

Only PRs by authors listed in `allowed_authors` will be reviewed; all others are ignored.

## Run

```bash
uvicorn app.main:app --reload
```

Or with a container:

```bash
podman build -t bitbucket-review -f Containerfile .
podman run -p 8080:8080 --env-file .env bitbucket-review
```

Or with Compose — create a `compose.yaml`:

```yaml
services:
  nitpick:
    build:
      context: .
      dockerfile: Containerfile
    ports:
      - "8080:8080"
    environment:
      - BITBUCKET_TOKEN=${BITBUCKET_TOKEN}
      - BITBUCKET_WEBHOOK_SECRET=${BITBUCKET_WEBHOOK_SECRET}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
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

Edit `prompts/review.txt` to change the review focus, tone, or output format. The `{diff}` placeholder is replaced with the actual diff at runtime.
