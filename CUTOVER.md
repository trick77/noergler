# Cutover: Python noergler to noergler-go

What has to change in `noergler-infra` to replace the Python service with this
one. The webhook contract, the `teams.yaml` shape and the riptide payload are
unchanged; the differences are the image, the migration step, and three
environment variables.

## 1. Image and commands

| | Python | Go |
| --- | --- | --- |
| Image | `ghcr.io/trick77/noergler` | `ghcr.io/trick77/noergler-go` |
| Entrypoint | `uvicorn app.main:app --host 0.0.0.0 --port 8080` | `["/noergler"]`, default command `serve` |
| Init container | `alembic upgrade head` | `["/noergler","migrate"]` |

The image is built and pushed on merge to master, tagged with the version plus
`major.minor`. Port stays 8080 (`SERVER_PORT`, honoured here; the Python service
read it into config but uvicorn's CLI argument decided the port).

## 2. ConfigMap changes

Rename:

```
OPENAI_BASE_URL   ->   LLMWIRE_LITELLM_BASE_URL
```

Add:

```
LLMWIRE_LITELLM_MODELS: "gpt-5.5=ai-gateway-gpt-5.5"
OPENAI_MODEL: "gpt-5.5"
```

`OPENAI_MODEL` changes meaning. Python sent its value to the gateway verbatim,
so it held the **gateway alias**. Here it is the llmwire **profile id**, and
`LLMWIRE_LITELLM_MODELS` maps the profile onto the alias. The same applies to
every team block's `inference.model`: replace the alias with the profile id and
make sure that profile appears in `LLMWIRE_LITELLM_MODELS`.

**A profile missing from that map disables the team at startup.** This is
deliberate: llmwire would otherwise route the request to `api.openai.com`
instead of the gateway. Check `teams_ready enabled=[...] disabled=[...]` in the
first log lines after the rollout.

Remove:

```
MALLOC_ARENA_MAX      # glibc tuning for the Python image, meaningless here
```

Unchanged: `BITBUCKET_URL`, `BITBUCKET_TOKEN`, `BITBUCKET_USERNAME`, `JIRA_URL`,
`JIRA_TOKEN`, `DATABASE_URL`, `TEAMS_CONFIG`, `NOERGLER_PUBLIC_URL`,
`LOG_LEVEL`, `NOERGLER_ENV`, every `REVIEW_*`, every `TEAM_*` secret, and
`SSL_CERT_FILE`.

`SSL_CERT_FILE` needs no setting of ours: Go's `crypto/x509` reads it when it
builds the system certificate pool.

## 3. Database

The schema is different: this service records **runs**
(`review_runs`, `findings`, `pull_requests`), where Python kept accumulators
(`pr_reviews`, `review_findings`). There is no migration between them.

Use a **new database or a dropped schema.** PR review history restarts: open PRs
lose their last-reviewed pointer and get one full review on their next push,
rather than an incremental one. Nothing else carries over.

`noergler migrate` is idempotent, takes an advisory lock and applies its
embedded files in filename order. Run it as an init container; it never runs
from `serve`.

**`team_claims` seeds from `teams.yaml` only on a first start against an empty
schema.** Since the cutover starts a new one, check that each team's `projects:`
block still lists what that team should own: whatever teams changed through
`POST /onboard/{team}` since the Python deployment went live exists only in the
old database, not in the yaml. Export the old `team_claims` first if in doubt.

## 4. Memory

`GOMEMLIMIT=1500MiB` is set in the image, sized for a 2 Gi pod.

RSS after startup and tokenizer warm-up is about **40 MiB** (measured locally;
`hack/smoke.sh` prints it). The tokenizer vocabulary is compiled into the binary
and warmed at boot, so there is no first-request stall and no network fetch,
unlike tiktoken.

Re-measure under load after the rollout and record it here: a 1M-token prompt is
the case the 1500 MiB limit exists for, and a laptop does not predict the pod.

Bitbucket bodies are capped while streaming (`BITBUCKET_MAX_DIFF_BYTES` 10 MiB,
`BITBUCKET_MAX_FILE_BYTES` 1 MiB) with four file fetches in flight.

## 5. Unchanged contracts

- **Webhooks.** Same path (`POST /webhook/<slug>`), same `X-Hub-Signature`
  HMAC-SHA256 over the raw body, same events. Existing Bitbucket webhooks need
  no change.
- **`teams.yaml`.** Same keys and same meaning, except `inference.model` as
  described above. Decoding is strict, so an unknown key disables that team.
- **riptide.** Same endpoint, same payload fields.
  `reviewer_account_kind: "bot"` and `reviewer_handle` are still always sent, an
  unknown cost is still omitted rather than sent as zero, and there is still one
  rollup per PR.
- **Probes.** `GET /health` and `GET /ready`, same semantics.
- **Log lines to alert on.** `team_disabled team=... reason=...`, `team_ready`,
  `teams_ready enabled=[...] disabled=[...]`, and the `http_request` access line.

## 6. Differences a user might notice

Everything posted to Bitbucket is byte-for-byte identical to the Python
service's output for the same input, verified by `hack/parity.sh`, with these
exceptions:

- **The model line in the summary footnote** reads `gpt-5.5-high` where Python
  read `ai-gateway-gpt-5.5-high`: the llmwire profile id rather than the
  gateway alias, because the alias is the operator's private naming and does not
  belong in a PR comment. The reasoning effort is shown on both.
- **Merged hunks keep one fewer blank context line.** Go takes the overlap off
  the first hunk's added context instead of trimming the second hunk's body,
  which is what made the Python version drop removal lines.
- **Acceptance-criteria prefixes match at a word boundary**, so `AC` no longer
  matches `Actual` and `Req` no longer matches `Request`.
- **No `/docs` endpoint.** It was FastAPI's.
- **The disagree/feedback mechanic is gone**, removed deliberately.

## 7. Rollout

1. Create the new database (or drop the schema).
2. Update the ConfigMap: rename the base URL, add the two model variables,
   remove `MALLOC_ARENA_MAX`, convert each team's `inference.model` to a profile
   id.
3. Point the init container at `["/noergler","migrate"]`.
4. Deploy `ghcr.io/trick77/noergler-go`.
5. Check the first log lines: every shared check `OK`, then `teams_ready` with
   the teams you expect enabled and nothing unexpectedly disabled.
6. `GET /ready` returns 200.
7. Open or push to one PR per team and confirm a review is posted.

Rolling back means redeploying the Python image against its old database. Keep
that database until the new one has run long enough to trust, since nothing
copies back.
