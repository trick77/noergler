# Configuration

noergler is configured in two layers:

1. **Instance** (environment variables): everything that is physically one thing, shared by every team, plus the defaults for each team-overridable knob.
2. **Teams** (`teams.yaml`): one block per team. Secrets are never in the file; each `*_env` field names the environment variable holding the value.

Resolution for a team-overridable value: **team block → instance default → built-in default.** Only two fields have no fallback: a team's `webhook_secret_env` and `inference.api_key_env`.

Reference files: [`.env.example`](.env.example) (instance layer) and [`teams.example.yaml`](teams.example.yaml) (team layer). In both, a live line is required and a commented line is optional with its default.

## 1. Instance layer (environment variables)

### Required

| Variable | Purpose |
|---|---|
| `BITBUCKET_URL` | Bitbucket Server base URL |
| `BITBUCKET_TOKEN` | Personal access token of the shared noergler service account. Needs repo read + write (it posts comments) on every onboarded repo |
| `BITBUCKET_USERNAME` | Username of that account. Identifies the bot's own comments and is the `@mention` trigger for every team |
| `OPENAI_BASE_URL` | Base URL of the OpenAI-compatible endpoint (e.g. a LiteLLM proxy). The SDK appends `/chat/completions`; a supplied suffix is stripped. Instance-wide, not team-overridable |
| `MODEL_CATALOG_URL` | Model catalog in LiteLLM's `model_prices_and_context_window.json` format. Every team's model is resolved against it at startup for the context window; its rates are the cost fallback. Instance-wide |
| `JIRA_URL` | Jira base URL |
| `JIRA_TOKEN` | Token of the single Jira user. Read-only use (fetches tickets for acceptance criteria) |
| `DATABASE_URL` | PostgreSQL connection string. Also read by Alembic |

### Teams

| Variable | Default | Purpose |
|---|---|---|
| `TEAMS_CONFIG` | `teams.yaml` | Path to the teams file. noergler does not start without a usable file |
| `TEAM_<SLUG>_WEBHOOK_SECRET` | | One per team: the team's webhook HMAC secret (`openssl rand -hex 32`). Referenced from `teams.yaml` as `webhook_secret_env` |
| `TEAM_<SLUG>_OPENAI_API_KEY` | | One per team: the team's key on the gateway. Referenced as `inference.api_key_env` |
| `TEAM_<SLUG>_RIPTIDE_TOKEN` | | Only for a team with a `riptide:` block. Referenced as `riptide.token_env` |

The `TEAM_<SLUG>_...` names are the convention (slug uppercased, `-` → `_`), not enforced: `teams.yaml` names its variables explicitly. The loader rejects a referenced variable that is missing or empty.

### Instance defaults, overridable per team

AI model (team block `inference:`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_MODEL` | `gpt-5.4` | Model id, looked up verbatim in the catalog (use the gateway's own alias if it publishes one, e.g. `ai-gateway/gpt-5.4`) |
| `OPENAI_REASONING_EFFORT` | `high` | One of `minimal`, `low`, `medium`, `high`. Mandatory: an empty value is a startup error, a model that rejects it disables the team |
| `OPENAI_CONTEXT_WINDOW` | `0` | Explicit context window in tokens; `0` = `max_input_tokens` from the catalog. The resolved window must be ≥ 1,000,000 |

Review behaviour (team block `review:`, same names without the `REVIEW_` prefix, lowercase):

| Variable | Default | Purpose |
|---|---|---|
| `REVIEW_AUTO_REVIEW_AUTHORS` | empty (= everyone) | Comma-separated PR authors that get automatic reviews |
| `REVIEW_MAX_COMMENTS` | `25` | Cap on inline comments per review run |
| `REVIEW_MAX_FILE_LINES` | `1000` | Files longer than this are reviewed from the diff only, without full file context |
| `REVIEW_DIFF_EXTRA_LINES_BEFORE` | `3` | Context lines added before each hunk |
| `REVIEW_DIFF_EXTRA_LINES_AFTER` | `2` | Context lines added after each hunk |
| `REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT` | `10` | How far above a hunk to look for the enclosing function/class |
| `REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT` | `true` | Extend hunks to the enclosing scope |
| `REVIEW_TICKET_COMPLIANCE_CHECK` | `true` | Evaluate the PR against the Jira ticket's acceptance criteria |
| `REVIEW_REQUIRE_AGENTS_MD` | `true` | Skip the review (with a summary explaining why) when the repo has no `AGENTS.md` |
| `REVIEW_AGENTS_MD_WARN_TOKENS` | `4000` | Warn in the summary when `AGENTS.md` exceeds this many tokens |
| `REVIEW_AGENTS_MD_MAX_TOKENS` | `7000` | Skip the review when `AGENTS.md` exceeds this many tokens |
| `REVIEW_AGENTS_MD_CUSTOM_LINK` | empty | Extra link in the "AGENTS.md too large" summary: `[Title](URL)` or a bare URL |
| `REVIEW_OPT_OUT_BRANCH_KEYWORD` | `noergloff` | Source branches containing this substring are not reviewed; empty disables |
| `REVIEW_MAX_PR_COST_USD` | `5.00` | Once a PR's accumulated cost reaches this, automatic reviews stop for it (`@mention` still works). Unknown cost fails open |

Jira (team block `jira:`):

| Variable | Default | Purpose |
|---|---|---|
| `JIRA_ACCEPTANCE_CRITERIA_PREFIXES` | `AC,AK,Acceptance Criteria,Acceptance Criterion,Akzeptanzkriterium,Akzeptanzkriterien,DoD,Req` | Headings recognised as acceptance-criteria sections |

### Instance only

| Variable | Default | Purpose |
|---|---|---|
| `REVIEW_PROMPT_TEMPLATE` | `prompts/review.txt` | Review prompt. One prompt set for all teams |
| `REVIEW_MENTION_PROMPT_TEMPLATE` | `prompts/mention.txt` | Mention Q&A prompt |
| `CONTEXT_WINDOW_HEADROOM_TOKENS` | `16000` | Flat headroom subtracted from the window below the trust threshold |
| `CONTEXT_TRUST_THRESHOLD` | `256000` | Window size beyond which only a fraction of the excess is trusted |
| `CONTEXT_TRUST_TAIL` | `0.5` | That fraction |
| `LOG_LEVEL` | `INFO` | Log level |
| `NOERGLER_ENV` | `dev` | `env` field on every log record |
| `NOERGLER_VERSION` / `OPENSHIFT_BUILD_COMMIT` | `dev` | Version printed at startup; the Containerfile bakes `NOERGLER_VERSION` |
| `SERVER_HOST`, `SERVER_PORT` | `0.0.0.0`, `8080` | Loaded but not used by the app; the container runs uvicorn on `0.0.0.0:8080` |
| `SSL_CERT_FILE` | | Corporate CA bundle, honoured by the HTTP clients |

## 2. Team layer (`teams.yaml`)

```yaml
teams:
  - slug: platform                                    # required
    name: "Platform Engineering"                      # optional, default: the slug
    webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET  # required
    projects:                                         # required, at least one
      - key: PLAT                                     # whole project: one project webhook, future repos included
      - key: INFRA
        repos: [terraform-core, ansible]              # optional: only these repos (one repo webhook each)
    inference:
      api_key_env: TEAM_PLATFORM_OPENAI_API_KEY       # required
      # model, reasoning_effort, context_window       # optional, default: instance OPENAI_*
    # review:  { any review knob }                    # optional, default: instance REVIEW_*
    # jira:    { acceptance_criteria_prefixes: [...] }# optional, default: instance JIRA_ACCEPTANCE_CRITERIA_PREFIXES
    # riptide: { url: ..., token_env: ... }           # optional; both fields or neither
```

| Field | Required | Rules |
|---|---|---|
| `slug` | yes | `^[a-z0-9][a-z0-9-]*$`, unique. Becomes the webhook path (`/webhook/<slug>`), the `pr_reviews.team_slug` value and the `team=` log field |
| `name` | no | Display only |
| `webhook_secret_env` | yes | Env var holding the team's Bitbucket webhook HMAC secret |
| `projects[].key` | yes, ≥ 1 | Bitbucket project keys the team owns |
| `projects[].repos` | no | Restrict a project to these repo slugs; omitted = the whole project. Non-empty when present. Use only for a project shared between teams: a whole-project claim is onboarded with one project webhook and covers every future repo, a `repos:` list needs a `teams.yaml` change per new repo |
| `inference.api_key_env` | yes | Env var holding the team's inference key. Empty value rejected |
| `inference.model`, `.reasoning_effort`, `.context_window` | no | Same validation as the instance variables |
| `review.*` | no | Any review knob from the table above. Unknown keys are rejected |
| `jira.acceptance_criteria_prefixes` | no | |
| `riptide.url`, `riptide.token_env` | both or neither | Present = forwarding on for this team. The token is validated at startup |

Not allowed in a team block (instance-wide by decision; naming them disables the team): `base_url`, `catalog_url`, `review_prompt_template`, `mention_prompt_template`.

**Ownership is exclusive.** A project, or a project/repo pair, belongs to exactly one team. A whole-project claim conflicts with any repo-level claim on the same key. Every team in a conflict is disabled.

## 3. What can go wrong, and what it affects

| Fault | Effect |
|---|---|
| Missing/invalid instance variable | Startup aborts |
| Database, Bitbucket or Jira unreachable at startup | Startup aborts |
| `teams.yaml` missing, unreadable, not valid YAML, no `teams:` list, zero teams, duplicate slug | Startup aborts (no single team owns the fault) |
| A team block fails validation (missing field, unknown key, instance-only key, bad `reasoning_effort`, half a `riptide:` block) | That team is disabled |
| A referenced `*_env` variable is missing or empty | That team is disabled |
| The team's model is not in the catalog, its window is below 1M, or the gateway rejects the key / `reasoning_effort` | That team is disabled |
| Riptide answers 401 to the team's token | That team is disabled |
| Riptide unreachable or answers something odd | Team stays enabled, warning logged; emissions are best-effort |
| Ownership conflict | Every claiming team is disabled |

A disabled team's webhook answers `503` (Bitbucket shows a failed delivery); the reason is in the startup log only. An unknown slug answers `404`. Fixing a team means changing the config and redeploying; nothing is retried at runtime.

Startup log: one `team_disabled team=<slug> reason=...` error per disabled team, then `teams_ready enabled=[...] disabled=[...]` (at `WARNING` when anything is disabled). Every log line about a team carries `team=<slug>` as a JSON field, which Splunk extracts automatically.

Probes: `/health` is liveness and answers `200` while the process is up, with the enabled and disabled slugs in the body. `/ready` is readiness and answers `503` while no team is enabled.

## 4. Webhooks

Each team's repositories send to `https://<noergler>/webhook/<slug>`. The service verifies the `X-Hub-Signature` HMAC-SHA256 against that team's secret, then checks that the PR's project/repo is owned by the team (`403` otherwise). Team identity therefore comes from the path and the signature, never from the payload.

**Who does what.** The noergler admin adds the team to `teams.yaml`, sets its secrets, redeploys and hands the team admin the webhook secret. The team admin creates the webhooks with their own Bitbucket personal access token (project admin) using `scripts/onboard_repo.py`, shipped in the image as `onboard` (see [README](README.md#webhook-setup)). Their `team.json` mirrors the team's `projects:` block:

| `teams.yaml` claim | Webhook created | New repo in the project |
|---|---|---|
| `key: PLAT` (whole project) | one **project** webhook (Bitbucket Data Center 8.8+) | covered automatically, nobody does anything |
| `key: INFRA` + `repos: [...]` | one **repo** webhook per listed repo | noergler admin adds it to `repos:`, team admin re-runs the tool |

**Probe.** Before touching Bitbucket the tool sends a signed request with `X-Event-Key: noergler:probe` to `/webhook/<slug>`. The service verifies the HMAC like a webhook and answers whether the team owns the target, how it is claimed (`whole`, `repos`, `none`) and whether the bot account can read it. A wrong secret is a `401` at this step, so the team admin can tell "wrong secret" from "not in `teams.yaml`" without asking anyone. The probe never enqueues a review.

**Double delivery.** A project webhook plus a leftover repo webhook of the same instance delivers every event twice, and the queue only collapses events that arrive while one is pending. The tool removes such repo hooks under a project hook (`--no-prune` keeps them); `--status` lists them. Hooks named `noergler` that point at *another* instance (intg next to prod) are reported as foreign and never touched; onboard a second instance with `--name`.

## 5. Secrets

Every secret is an environment variable. `teams.yaml` and the ConfigMaps contain names only.

| Secret | Scope | Where it comes from |
|---|---|---|
| `BITBUCKET_TOKEN` | instance | Personal access token of the noergler service account in Bitbucket Server (account → *Manage account → HTTP access tokens*). Needs repository read + write |
| `JIRA_TOKEN` | instance | Personal access token of the Jira user. Read access is enough |
| `DATABASE_URL` | instance | Connection string with the database password embedded |
| `TEAM_<SLUG>_WEBHOOK_SECRET` | per team | Generated by the noergler admin: `openssl rand -hex 32`. The same value goes into the service's environment and, via the team admin's `team.env`, into every webhook of that team (`onboard` copies it there) |
| `TEAM_<SLUG>_OPENAI_API_KEY` | per team | Issued on the LLM gateway (LiteLLM: a virtual key) for that team, so spend is attributed per team. Never the gateway master key |
| `TEAM_<SLUG>_RIPTIDE_TOKEN` | per team, optional | The team's bearer, issued by whoever runs that team's riptide-collector |

Adding a team therefore means: generate the webhook secret, obtain the gateway key (and riptide token if used), put the three variables where the instance reads its environment, add the block to `teams.yaml`, redeploy, hand the webhook secret to the team admin, who runs `onboard` (their own Bitbucket PAT, never the bot token).

**Local / compose:** append the variables to `.env` (gitignored).

**OpenShift:** add them to the `noergler` Secret next to the instance secrets, e.g.

```bash
oc create secret generic noergler \
  --from-literal=BITBUCKET_TOKEN=... \
  --from-literal=JIRA_TOKEN=... \
  --from-literal=DATABASE_URL=... \
  --from-literal=TEAM_PLATFORM_WEBHOOK_SECRET="$(openssl rand -hex 32)" \
  --from-literal=TEAM_PLATFORM_OPENAI_API_KEY=...
```

The Secret is consumed with `envFrom`, so a new key needs no Deployment change, only a restart. In a GitOps setup (Sealed Secrets: plaintext `secrets.env` → `kubeseal` → committed `sealed-secrets.yaml`) the same keys go into `secrets.env`; each team's keys become required as soon as its block references them.

**Rotation.** Webhook secret: set the new value on the service, redeploy, hand it to the team admin, who runs `onboard --remove` and then `onboard` again; Bitbucket's webhook API does not return the stored secret, so a plain re-run would report *already up to date* (see the README's *secret-only drift* note). Gateway key or riptide token: replace the variable, redeploy; a rejected key or token shows up as `team_disabled` at startup, nothing else is affected.

## 6. Where the files go

| Deployment | Instance layer | Teams file |
|---|---|---|
| Local / compose | `.env` (`env_file`), see `compose.yaml` | `./teams.yaml` bind-mounted to `/app/teams.yaml`, `TEAMS_CONFIG=/app/teams.yaml` |
| OpenShift (`openshift/`) | ConfigMap `noergler` + Secret `noergler` via `envFrom` | ConfigMap `noergler-teams` mounted at `/etc/noergler/teams.yaml`; readiness probe on `/ready` |

Both `.env` and `teams.yaml` are gitignored; commit the `.example` files only.

## 7. Database

PostgreSQL, schema managed by Alembic (`alembic upgrade head`, run by the OpenShift init container). The current schema is a single revision `001`; there is no upgrade path from databases created before the multi-team change, they must be recreated. `pr_reviews.team_slug` records which team a review ran for.
