# Configuration

Two layers.

The **instance layer** is environment variables: everything shared by all teams,
plus the defaults a team may override. The **team layer** is `teams.yaml`: one
block per team, holding what is that team's own (its projects, its gateway key,
its webhook secret, its riptide endpoint).

No secret goes in `teams.yaml`. A `*_env` key names the environment variable
that holds the value.

`.env.example` is the working reference for the instance layer: a live line is
required, a commented line is optional and shows its default. This document
explains what the values mean and what happens when they are wrong.

## 1. Instance layer (environment variables)

### Required

| Variable | Meaning |
| --- | --- |
| `BITBUCKET_URL` | Base URL of the Bitbucket Server instance. |
| `BITBUCKET_TOKEN` | HTTP access token of the shared service account. Needs to read repositories and write comments; project admin is **not** needed (self-service onboarding uses the caller's own token instead). |
| `BITBUCKET_USERNAME` | The service account's name. It doubles as the `@mention` trigger, instance-wide: a comment containing `@<BITBUCKET_USERNAME>` reaches noergler. |
| `LLMWIRE_LITELLM_BASE_URL` | Host of the OpenAI-compatible gateway endpoint. llmwire appends `/chat/completions`. |
| `LLMWIRE_LITELLM_MODELS` | `<profile>=<alias>,...`: which llmwire profiles the gateway serves, and the alias each is listed under. See "Models and aliases" below, it is the one setting most likely to be wrong. |
| `OPENAI_MODEL` | Default model as an llmwire **profile id** (`gpt-5.5`), never a gateway alias. Must appear in `LLMWIRE_LITELLM_MODELS`. Teams may override it. |
| `DATABASE_URL` | PostgreSQL DSN. |
| `JIRA_URL` | Base URL of the Jira instance. |
| `JIRA_TOKEN` | Token of a single read-only Jira user; used to fetch tickets for the acceptance-criteria check. |

### Teams

| Variable | Default | Meaning |
| --- | --- | --- |
| `TEAMS_CONFIG` | `teams.yaml` | Path to the teams file. noergler does not start without a readable one. |

Per-team secrets are referenced from `teams.yaml` through `*_env` keys. The
names are yours; the convention is the slug uppercased with `-` turned into `_`:

```
TEAM_PLATFORM_WEBHOOK_SECRET=      # required per team: openssl rand -hex 32
TEAM_PLATFORM_OPENAI_API_KEY=      # required per team: the team's key on the gateway
TEAM_PLATFORM_RIPTIDE_TOKEN=       # only when the team has a riptide: block
```

### Instance defaults, overridable per team

Everything here is a default a team block may replace.

| Variable | Default | Meaning |
| --- | --- | --- |
| `REVIEW_AUTO_REVIEW_AUTHORS` | *(empty)* | Comma-separated author names. Empty reviews every author. |
| `REVIEW_IGNORE_AUTHORS` | *(empty)* | Never auto-reviewed (CI and dependency bots). Wins over the list above. |
| `REVIEW_EXCLUDE_REPOS` | `*-infra` | Repo-slug globs a team's project webhook delivers but noergler ignores. Seeds each team once. |
| `REVIEW_MAX_COMMENTS` | `25` | Cap on inline comments per review. |
| `REVIEW_MAX_FILE_LINES` | `1000` | Files longer than this are reviewed without full file content. |
| `REVIEW_DIFF_EXTRA_LINES_BEFORE` | `3` | Context lines added before each hunk. |
| `REVIEW_DIFF_EXTRA_LINES_AFTER` | `2` | Context lines added after each hunk. |
| `REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT` | `10` | How far above the window to search for the enclosing definition. |
| `REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT` | `true` | Extend context to include the enclosing function or class. |
| `REVIEW_TICKET_COMPLIANCE_CHECK` | `true` | Evaluate the PR against the Jira ticket's acceptance criteria. |
| `REVIEW_REQUIRE_AGENTS_MD` | `true` | Skip the review, with a summary saying why, when the repo has no `AGENTS.md`. |
| `REVIEW_AGENTS_MD_WARN_TOKENS` | `4000` | Warn in the summary above this size. |
| `REVIEW_AGENTS_MD_MAX_TOKENS` | `7000` | Skip the review above this size. |
| `REVIEW_AGENTS_MD_CUSTOM_LINK` | *(empty)* | Extra link in the "too large" summary: `[Title](URL)` or a bare URL. |
| `REVIEW_OPT_OUT_BRANCH_KEYWORD` | `noergloff` | Substring in the source branch name that skips the review. Empty disables it. |
| `REVIEW_MAX_PR_COST_USD` | `5.00` | Once a PR's accumulated cost reaches this, auto-review stops. An `@mention` still works. |
| `JIRA_ACCEPTANCE_CRITERIA_PREFIXES` | `AC,AK,Acceptance Criteria,Acceptance Criterion,Akzeptanzkriterium,Akzeptanzkriterien,DoD,Req` | Prefixes that mark an acceptance-criteria line in a ticket. Matched at a word boundary. |
| `OPENAI_REASONING_EFFORT` | unset | Unset: the model's balanced level, from its llmwire profile. Set: a level the profile lists, checked at team startup before any request; one it does not list disables the team and the error names the accepted ones. The level sent is part of every run's model label. |
| `OPENAI_CONTEXT_WINDOW` | `0` | `0` takes `max_input_tokens` from the gateway's `/v1/models`. Set it when the gateway lists no window, or enforces a lower cap than it advertises. |

### Instance only

Not overridable in a team block. Naming one there disables the team.

| Variable | Default | Meaning |
| --- | --- | --- |
| `REVIEW_PROMPT_TEMPLATE` | `prompts/review.txt` | Review prompt template. |
| `REVIEW_MENTION_PROMPT_TEMPLATE` | `prompts/mention.txt` | Q&A prompt template. |
| `BITBUCKET_MAX_DIFF_BYTES` | `0` | Bytes per PR or compare diff; `0` is unlimited. Over it the PR is skipped, so set it only on a pod too small to rely on `GOMEMLIMIT`. A diff's size is mostly files the review discards. |
| `BITBUCKET_MAX_FILE_BYTES` | `1048576` | 1 MiB per file fetched for context. |
| `CONTEXT_WINDOW_HEADROOM_TOKENS` | `16000` | Trust-curve headroom. |
| `CONTEXT_TRUST_THRESHOLD` | `256000` | Window size trusted in full. |
| `CONTEXT_TRUST_TAIL` | `0.5` | Fraction of the window beyond the threshold that counts. |
| `REVIEW_INFERENCE_CONCURRENCY` | `6` | Inference calls in flight process-wide. Bitbucket work stays on the single worker whatever this is; only the gateway call overlaps. |
| `REVIEW_INFERENCE_CONCURRENCY_PER_TEAM` | `2` | Inference calls in flight for one team, nested inside the global cap so one team's burst cannot take every slot. Must not exceed it. |
| `SERVER_HOST` | `0.0.0.0` | Listen address. |
| `SERVER_PORT` | `8080` | Listen port. |
| `NOERGLER_PUBLIC_URL` | *(empty)* | How Bitbucket reaches this instance. Needed by `POST /onboard/{team}` only, to write the webhook URL. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` (or `WARN`), `ERROR`, any case. |
| `NOERGLER_ENV` | `dev` | The `env` field on every log record. |
| `NOERGLER_VERSION` | *(build stamp)* | Fallback version string when the binary carries no build info. |

The two body caps are enforced while streaming, not after the body is in
memory: one diff is held several times over (raw, split per file, rendered into
the prompt, and as token ids).

### Models and aliases

The single most common misconfiguration.

`OPENAI_MODEL` and a team's `inference.model` are llmwire **profile ids**. The
gateway knows its own **aliases**. `LLMWIRE_LITELLM_MODELS` maps between them:

```
LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5
OPENAI_MODEL=gpt-5.5
```

A profile that is **not** listed there is disabled at startup, deliberately:
llmwire would otherwise route it to `api.openai.com` instead of your gateway.

The context window is resolved per team from the gateway's `/v1/models`, read
for the **alias**, and must be at least 1,000,000 tokens. noergler reviews a PR
in a single call. `OPENAI_CONTEXT_WINDOW` overrides the gateway's answer.

## 2. Team layer (`teams.yaml`)

See `teams.example.yaml` for a fully commented file. The minimum:

```yaml
teams:
  - slug: payments
    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET
    projects:
      - key: PAY
    inference:
      api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY
```

| Key | Required | Meaning |
| --- | --- | --- |
| `slug` | yes | `[a-z0-9-]`, unique. It is the webhook path (`/webhook/payments`), the `team_slug` column and the `team=` log field. |
| `webhook_secret_env` | yes | Env var holding this team's HMAC secret. |
| `inference.api_key_env` | yes | Env var holding this team's gateway key. |
| `projects` | no | A **seed** only: project keys and repo slugs the team claims, written to the database on first start. |
| `inference.model`, `inference.reasoning_effort`, `inference.context_window` | no | Override the instance default. |
| `review.*` | no | Override any `REVIEW_*` default. |
| `jira.acceptance_criteria_prefixes` | no | Override the instance prefixes. |
| `riptide.url`, `riptide.token_env` | no | Present means FinOps forwarding is on. Both required together. |

Decoding is strict: an unknown key disables that team, and only that team.

`projects:` seeds the database once. After the first start the claims live in
`team_claims`, which is what ownership is checked against, and the yaml is no
longer consulted for them. Teams change their claims through
`POST /onboard/{team}`, not by editing the file. A seed that collides with
another team's existing claim disables only the seeding team.

## 3. What can go wrong, and what it affects

The rule: **one team's fault disables that team only. A shared-layer fault stops
the service.**

| Fault | Effect |
| --- | --- |
| Unreadable or invalid `teams.yaml` | Startup aborts. It is the shared layer. |
| Unknown key in a team block | That team is disabled. `team_disabled team=<slug> reason=...` |
| Team's model missing from `LLMWIRE_LITELLM_MODELS` | That team is disabled, rather than being routed to the vendor's own host. |
| Gateway lists a window below 1M for the alias | That team is disabled. Set `OPENAI_CONTEXT_WINDOW` if the gateway understates it. |
| Model cannot reason, lacks strict JSON-schema output, or does not list the configured `reasoning_effort`, or that level switches reasoning off, or no level is set on a model that reasons only when asked and names no balanced level | That team is disabled, from its llmwire profile and before any request. The error names the valid choices. |
| Team's riptide ping returns 401 | That team is disabled. Any other riptide error is a warning. |
| Missing per-team secret env var | That team is disabled. |
| Database, Bitbucket or Jira unreachable at startup | Startup aborts, after reporting every failing check rather than only the first. |
| Jira ticket unreadable during a review | Not an error. The key came off a branch name and may be noise. |
| Gateway does not price a call | The review proceeds and the run is stored with a NULL cost. Cost fails open. |

`teams_ready enabled=[...] disabled=[...]` is logged once at startup and is the
line to alert on, along with `team_disabled`.

## 4. Webhooks

One webhook per team, pointing at `POST /webhook/<slug>`, secured with that
team's secret.

Team identity is the **webhook path plus that team's HMAC secret plus the
ownership check**. The payload's `project.key` alone is never trusted: the
authenticated slug decides. A signed delivery for a repository the team does not
claim is ignored.

Events: `pr:opened`, `pr:modified`, `pr:from_ref_updated`, `pr:merged`,
`pr:declined`, `pr:deleted`, `pr:comment:added`, `pr:comment:deleted`.

`POST /onboard/{team}` sets the webhook up for a team's targets, using the
caller's own Bitbucket token (`X-Bitbucket-Token`) to prove project admin per
target. See `http/` for ready-made requests.

## 5. Secrets

`teams.yaml` holds no secrets, only the names of environment variables. The
webhook secret is per team; generate one with `openssl rand -hex 32`.

The Bitbucket token in `BITBUCKET_TOKEN` is the shared bot's and needs no admin
rights. The admin token used for onboarding is the caller's, passed per request
in `X-Bitbucket-Token`, never logged and never stored.

## 6. Where the files go

| Path | What |
| --- | --- |
| `teams.yaml` | Team layer. Mounted read-only at `/app/teams.yaml` in the container. |
| `prompts/review.txt`, `prompts/mention.txt` | Prompt templates. Mounted read-only at `/app/prompts`. |
| `.env` | Instance layer for compose. Never committed. |

## 7. Database

PostgreSQL. Migrations are embedded in the binary and applied by a separate
subcommand:

```
noergler migrate
```

It is the init container in a deployment and never runs from `serve`. It is
idempotent, guarded by an advisory lock, and applies files in filename order.

The schema records **runs**, not accumulators: per-PR totals are aggregates over
`review_runs`, and a PR's cost is NULL until at least one run is priced.
