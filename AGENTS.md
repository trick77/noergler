# AGENTS.md

Bitbucket Server PR review bridge; inference via `github.com/trick77/llmwire`
against a LiteLLM gateway. Behaviour: HOW_IT_WORKS.md.

**Before editing an area, read its section in INVARIANTS.md** (teams, money,
prompt and findings, evals, llmwire, adapters, webhook, pipeline, store, ops,
diff engine). Why a rule exists: RATIONALE.md. Changing a rule is a
decision, not a cleanup.

## Commands

- Module in `backend/` (go commands there); `scripts/`, `ui/`, docs and
  `make` at the root. Prompts are `backend/internal/prompts/*.txt`, compiled
  in (go:embed), never configurable.
- Gate = `make check` (ship master; `SHIP_CMD=<binary>` locally). `make hooks`
  per clone. GitHub is never first to find a failure; a Dependabot PR gets
  `make check` before merge.
- Store tests need docker (ship starts Postgres); plain `go test` skips them.
- Commit subjects + PR titles: Go style `area: what` (`review: …`, `store: …`), gated by ship; release notes list them by area.
- Merge ships nothing; release = `gh workflow run Release`, GitHub-hosted
  runners only, never a laptop.
- `./scripts/smoke.sh` boots `serve` against `scripts/fakes`.
- No web framework, no ORM, no logging lib beyond ship's kit.

## Everywhere

- Team identity = webhook path + HMAC secret + ownership; never payload
  `project.key`. One team's fault disables that team only; a shared-layer
  fault never disables one team alone. Every team line carries `team=`.
- No secrets in `teams.yaml`, logs, `/readyz` or the dashboard; a disable
  reason never leaves the logs.
- Cost fails open (unpriced = NULL, review proceeds). DB nano-USD `BIGINT`;
  JSON USD = 3-decimal string, never float; the UI shows cents via
  `money()`, run page included (user, 2026-10-19). Key spend shown, never summed.
- Single worker for every Bitbucket call; only the gateway call leaves it,
  staged onto the bounded pool. Anything staged reports `handedOff`.
- Every finding passes `ValidateFindings` before posting; validation is blunt
  rules, never a judge model. Drops and decisions that change what is
  reviewed, posted or billed get a log line.
- Author text (PR description, replies, disputes) is intent, never evidence,
  never drops a finding.
- `review.txt`: `{files}` before `{cumulative_pr_diff}` and
  `{previously_posted_findings}` (prefix cache). One-pass placeholders, never
  `text/template`.
- RE2 classes are ASCII: write Unicode-aware patterns (INVARIANTS: adapters).
- Webhook check order is fixed; do not tidy it (INVARIANTS: webhook).
- Models are llmwire profile ids in config, never in code. `Chat` only, never
  streaming, never retry, never `ReasoningMinimal()`.
- Bitbucket: never `Client.Timeout`, never follow redirects.
- Never edit an applied migration. `serve` never migrates.
- `team_disabled`, `team_ready`, `teams_ready` are alerted on: never reword.
  Probes are exec, never httpGet.
- Every eval run is committed with its README row; evals are not a gate.
