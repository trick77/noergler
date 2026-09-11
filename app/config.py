import itertools
import logging
import os
import re
import time
from typing import Any

import httpx
import structlog
import yaml
from pydantic import BaseModel, ValidationError, field_validator

from app.http_stats import make_event_hook

# Webhook events the /webhook endpoint in app/main.py dispatches on.
# Kept here so the provisioning script and the service stay in sync.
REQUIRED_WEBHOOK_EVENTS: tuple[str, ...] = (
    "pr:opened",
    "pr:from_ref_updated",
    "pr:comment:added",
    "pr:comment:deleted",
    "pr:merged",
    "pr:declined",
    "pr:deleted",
)


class BitbucketConfig(BaseModel):
    base_url: str
    token: str
    username: str


_REASONING_EFFORT_VALUES = frozenset({"minimal", "low", "medium", "high"})


def model_label(model: str, reasoning_effort: str | None) -> str:
    if reasoning_effort:
        return f"{model}-{reasoning_effort}"
    return model


# --- Model catalog ---------------------------------------------------------
# Context windows come from a catalog in LiteLLM's `model_prices_and_context_
# window.json` format, fetched from `MODEL_CATALOG_URL`. That URL is deployment
# configuration rather than a constant: the public catalog on GitHub is not
# reachable from every environment, and a gateway that proxies models under its
# own names (`ai-gateway/gpt-5.4`) has to publish its own catalog for those names
# to resolve at all. There is no baked-in fallback table and no DB cache: the
# catalog is the single source of truth. Startup resolves the configured model
# against it exactly once and aborts if that fails (see `resolve_or_raise`).
#
# Pricing here is a *fallback only*. The proxy reports the actual cost of each
# call on the response (`x-litellm-response-cost`), computed by the same code
# that bills — including tiered rates above a prompt threshold, prompt-cache read
# rates, and any service tier or margin configured on the gateway. That figure
# always wins. The catalog rates are used only when the proxy reports nothing
# usable, which happens on a deployment LiteLLM itself can't price; without them
# the summary would show no cost at all and the per-PR cap would silently stop
# applying. See `app.llm_client._usd_header` and `resolve_cost_usd`.

# LiteLLM exposes some providers only under prefixed keys. Probe these in order
# so e.g. `claude-sonnet-4.6` resolves to the openrouter entry.
_LITELLM_KEY_PREFIXES: tuple[str, ...] = (
    "",
    "openrouter/anthropic/",
    "vercel_ai_gateway/anthropic/",
)


class ModelCatalogEntry(BaseModel):
    """The catalog facts noergler needs about the configured model."""

    # The id we asked for (the configured `OPENAI_MODEL`).
    model_id: str
    # The catalog key that actually matched. Differs from `model_id` on a
    # provider-prefixed key (`openrouter/anthropic/...`) or a prefix fallback
    # (`gpt-5.4-mini-2025-06-01` -> `gpt-5.4-mini`). Kept distinct because the
    # fallback searches the whole catalog (~3000 ids, including deprecated and
    # regional variants), so a wrong-but-plausible match would otherwise be
    # invisible — it resolves and boots cleanly.
    matched_key: str
    max_input_tokens: int
    # Fallback rates, USD per 1M tokens. Only used when the endpoint reports no
    # usable cost of its own — see `resolve_cost_usd`. None when the catalog
    # entry publishes no pricing.
    input_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None
    output_per_mtok: float | None = None


def _rate(value: object) -> float | None:
    """LiteLLM per-token cost -> USD per 1M tokens, or None if unusable."""
    if value is None:
        return None
    try:
        rate = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return None
    return rate * 1_000_000 if rate >= 0 else None


def _parse_catalog_entry(
    model_id: str, matched_key: str, raw: dict[str, Any]
) -> ModelCatalogEntry | None:
    """Build an entry from one LiteLLM record, or None if it's unusable."""
    log = logging.getLogger(__name__)
    try:
        window = int(raw["max_input_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("malformed LiteLLM max_input_tokens for %s: %s", model_id, exc)
        return None
    if window <= 0:
        log.warning("LiteLLM entry for %s has non-positive max_input_tokens", model_id)
        return None
    input_per_mtok = _rate(raw.get("input_cost_per_token"))
    output_per_mtok = _rate(raw.get("output_cost_per_token"))
    cached_per_mtok = _rate(raw.get("cache_read_input_token_cost"))
    if input_per_mtok is None or output_per_mtok is None:
        # Window is usable on its own; pricing just won't be available as a
        # fallback for this model.
        input_per_mtok = output_per_mtok = cached_per_mtok = None
    elif cached_per_mtok is None:
        # No published cache-read rate: charge cache hits at the full input
        # rate rather than inventing a discount.
        cached_per_mtok = input_per_mtok
    return ModelCatalogEntry(
        model_id=model_id, matched_key=matched_key, max_input_tokens=window,
        input_per_mtok=input_per_mtok,
        cached_input_per_mtok=cached_per_mtok,
        output_per_mtok=output_per_mtok,
    )


def resolve_catalog_entry(
    data: dict[str, Any], model_id: str
) -> ModelCatalogEntry | None:
    """Find `model_id` in a fetched catalog, or None.

    Tries each provider prefix on the exact id first, then falls back to the
    longest catalog key that `model_id` extends, so a dated id like
    `gpt-5.4-mini-2025-06-01` resolves to `gpt-5.4-mini` rather than the
    shorter `gpt-5.4`. The fallback is prefix-aware too: LiteLLM lists some
    families only under a provider prefix, so `claude-sonnet-4.6-20260101` has
    to be matched against the *unprefixed* tail of `openrouter/anthropic/...`.

    A candidate that fails to parse is skipped rather than aborting the search —
    one entry with a null window must not mask a valid entry under a later
    prefix.
    """
    for prefix in _LITELLM_KEY_PREFIXES:
        key = f"{prefix}{model_id}"
        raw = data.get(key)
        if isinstance(raw, dict) and "max_input_tokens" in raw:
            entry = _parse_catalog_entry(model_id, key, raw)
            if entry is not None:
                return entry

    # (catalog key, length of the part `model_id` actually extends) so the
    # longest *model* match wins regardless of how long its provider prefix is.
    candidates: list[tuple[str, int]] = []
    for key, raw in data.items():
        if not isinstance(raw, dict) or "max_input_tokens" not in raw:
            continue
        for prefix in _LITELLM_KEY_PREFIXES:
            if not key.startswith(prefix):
                continue
            base = key[len(prefix):]
            if base and model_id.startswith(base + "-"):
                candidates.append((key, len(base)))
                break
    for key, _ in sorted(candidates, key=lambda c: c[1], reverse=True):
        entry = _parse_catalog_entry(model_id, key, data[key])
        if entry is not None:
            return entry
    return None


# The live entries, one per model any enabled team uses. Installed by
# `resolve_or_raise` at startup, replaced wholesale by the 24h refresher.
# Readers snapshot the reference (atomic under the GIL) so a swap mid-flight
# never tears a lookup.
_ACTIVE_ENTRIES: dict[str, ModelCatalogEntry] = {}


def _swap_active_entry(entry: ModelCatalogEntry) -> None:
    _ACTIVE_ENTRIES[entry.model_id] = entry


def active_entry(model_id: str) -> ModelCatalogEntry | None:
    """The catalog entry for `model_id`, or None before its startup resolve."""
    return _ACTIVE_ENTRIES.get(model_id)


async def fetch_model_catalog(
    url: str, timeout: float = 10.0
) -> dict[str, Any] | None:
    """GET the model catalog once and return the parsed JSON, or None."""
    log = logging.getLogger(__name__)
    # Announce the fetch before making it, not after. The URL is per-deployment
    # now, and a wrong or unroutable one costs the full timeout at startup —
    # without this line those seconds are silent and the first thing the
    # operator sees is a fatal error.
    log.info("Model catalog: fetching %s", url)
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            event_hooks={"request": [make_event_hook("catalog")]},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        # The URL is repeated here because `exc` carries it for a transport or
        # status error but not for a JSON decode failure.
        log.warning("model catalog fetch from %s failed: %s", url, exc)
        return None
    if not isinstance(data, dict):
        log.warning("model catalog at %s is not a JSON object", url)
        return None
    log.info(
        "Model catalog: fetched %d entries in %.1fs",
        len(data), time.monotonic() - started,
    )
    return data


class ModelCatalogError(RuntimeError):
    """The configured model could not be resolved against the model catalog."""


async def resolve_or_raise(
    model_id: str, url: str, timeout: float = 10.0
) -> ModelCatalogEntry:
    """Fetch the catalog and install the entry for `model_id`, or raise.

    Called once at startup. Both failure modes are fatal by design: without a
    catalog entry noergler has no context window to size the review against, and
    silently guessing one is worse than not starting. The 24h refresh is the
    opposite — best-effort, keeping the entry installed here when a later fetch
    fails.
    """
    data = await fetch_model_catalog(url, timeout)
    if data is None:
        raise ModelCatalogError(
            f"could not fetch the model catalog from {url}. noergler reads the "
            "model's context window from it at startup and keeps no local "
            "fallback — check MODEL_CATALOG_URL and network/proxy egress to that host."
        )
    entry = resolve_catalog_entry(data, model_id)
    if entry is None:
        raise ModelCatalogError(
            f"model `{model_id}` is not in the catalog at {url} ({len(data)} entries). "
            "the model (OPENAI_MODEL or the team's inference.model) must be spelled exactly as the catalog spells it — for a "
            "gateway alias, that is the prefixed name the gateway publishes."
        )
    _swap_active_entry(entry)
    return entry


async def refresh_active_entry(
    model_id: str, url: str, timeout: float = 10.0, min_window: int = 0
) -> bool:
    """Re-resolve `model_id` and swap the entry in. Best-effort.

    Returns False and leaves the installed entry untouched on any failure — a
    refresh must never take a running instance down, unlike the startup resolve.

    `min_window` rejects a swap that would drop the context window below the
    floor startup enforces. The catalog is upstream data that can be corrected
    downward (its own history includes a model whose window was listed as a
    pricing threshold, ~4x too low); without this a 24h refresh could quietly
    push a running instance under a bound the rest of the code treats as
    guaranteed. Keeping the older, valid entry is the safer failure.
    """
    log = logging.getLogger(__name__)
    data = await fetch_model_catalog(url, timeout)
    if data is None:
        return False
    entry = resolve_catalog_entry(data, model_id)
    if entry is None:
        log.warning(
            "model catalog refresh: `%s` vanished from the catalog — keeping the "
            "entry loaded at startup", model_id,
        )
        return False
    if min_window and entry.max_input_tokens < min_window:
        log.warning(
            "model catalog refresh: `%s` now reports a %d-token window, below the "
            "required %d — rejecting the update and keeping the entry loaded at "
            "startup", model_id, entry.max_input_tokens, min_window,
        )
        return False
    _swap_active_entry(entry)
    return True


# Turning a model's advertised context window into a usable per-chunk budget.
# A flat headroom (the old 16k) is ~1.5% of a 1M window — useless — so we apply
# a diminishing-trust curve: trust the window fully up to a threshold, then
# count only a fraction of everything beyond it. Large advertised windows are
# the least trustworthy: many endpoints enforce a lower server-side cap and 413
# anything bigger, so they degrade most. All three knobs are
# env-overridable for tuning without a redeploy.
_CONTEXT_WINDOW_HEADROOM_TOKENS = int(os.environ.get("CONTEXT_WINDOW_HEADROOM_TOKENS", "16000"))
_CONTEXT_TRUST_THRESHOLD = int(os.environ.get("CONTEXT_TRUST_THRESHOLD", "256000"))
_CONTEXT_TRUST_TAIL = float(os.environ.get("CONTEXT_TRUST_TAIL", "0.5"))


def usable_context_budget(window: int) -> int:
    """Usable per-chunk token budget for a given context window.

    Below the trust threshold: the full window minus a flat headroom. Above it:
    the threshold plus only `TAIL` of the excess. Examples (T=256k, TAIL=0.5,
    floor=16k): 128k->112k, 272k->264k, 512k->384k, 1.05M->653k.
    """
    if window <= _CONTEXT_TRUST_THRESHOLD:
        usable = window - _CONTEXT_WINDOW_HEADROOM_TOKENS
    else:
        usable = _CONTEXT_TRUST_THRESHOLD + int((window - _CONTEXT_TRUST_THRESHOLD) * _CONTEXT_TRUST_TAIL)
    return max(2000, usable)


class TokenUsage(BaseModel):
    """Token counts and the reported cost for one LLM call.

    `cached` is the subset of `prompt` that hit the provider's prompt cache
    (`usage.prompt_tokens_details.cached_tokens`). It is logged only — nothing
    multiplies it, because the cost below already has the cache discount
    applied by the proxy. Kept because it's the one signal that explains an
    unexpectedly high reported cost.

    `cost_usd` is the proxy's own figure for this call, taken from the
    `x-litellm-response-cost` response header. None means the endpoint didn't
    report one (anything that isn't a LiteLLM proxy) — `resolve_cost_usd` then
    falls back to the catalog rates, and only if those are missing too does the
    run go unpriced with the per-PR cost cap failing open.

    `key_spend_usd` is the total already spent on the API key, from the
    `x-litellm-key-spend` header. Unlike `cost_usd` it is a gauge, not a
    per-call amount: it covers every call made with the key by anyone, is
    already cumulative, and must never be summed, stored as a run cost, or fed
    to the per-PR cap. Display only.
    """

    prompt: int = 0
    cached: int = 0
    completion: int = 0
    cost_usd: float | None = None
    key_spend_usd: float | None = None

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    @property
    def uncached_prompt(self) -> int:
        # Clamp: a proxy reporting cached > prompt must not produce a negative
        # billable count.
        return max(0, self.prompt - self.cached)


def resolve_cost_usd(
    usage: TokenUsage, entry: ModelCatalogEntry | None
) -> tuple[float | None, bool]:
    """(cost, was_reported) for one call.

    Prefers the endpoint's own figure, which is exact — it already accounts for
    tiered rates, cache-read rates, service tier and any gateway margin. Falls
    back to the catalog rates when the endpoint reports nothing usable, which
    happens whenever the proxy itself can't price a deployment (LiteLLM then
    sends the literal string "None"). Above a model's tiered-pricing threshold
    the fallback *understates* the real figure, since it charges the whole
    prompt at the base input rate — a lower bound, not an upper one. A bounded
    number still beats no number: without it the summary shows no cost and the
    per-PR cap silently stops applying.
    """
    # A reported zero on a call that actually consumed tokens means the endpoint
    # is misconfigured, not that the call was free — e.g. a LiteLLM deployment
    # with input/output cost explicitly set to 0, which prices everything at
    # $0.00 and would silently disable the per-PR cap. Fall through to the
    # catalog. A genuinely free model prices at 0 there too, so this can't
    # invent a cost for one.
    if usage.cost_usd is not None and not (usage.cost_usd == 0 and usage.total > 0):
        return usage.cost_usd, True
    price = entry
    if (
        price is None
        or price.input_per_mtok is None
        or price.output_per_mtok is None
        or price.cached_input_per_mtok is None
    ):
        return None, False
    estimated = (
        usage.uncached_prompt * price.input_per_mtok
        + usage.cached * price.cached_input_per_mtok
        + usage.completion * price.output_per_mtok
    ) / 1_000_000
    return estimated, False


class LLMConfig(BaseModel):
    model: str = "gpt-5.4"
    api_key: str
    api_url: str
    # noergler requires a reasoning-capable model, so reasoning_effort is
    # mandatory — an empty value is rejected rather than silently disabling it.
    reasoning_effort: str = "high"
    # Where to fetch the model catalog. Required, with no default: the public
    # LiteLLM catalog is unreachable from some environments and does not list
    # gateway-prefixed names, so every deployment states its own — and intg and
    # prod point at different hosts. `model` is looked up in it verbatim.
    catalog_url: str
    # Explicit context window (tokens). 0 = auto-detect from the catalog. Set
    # this for an endpoint whose real cap differs from what its catalog
    # advertises; the startup guard requires the resolved window to be
    # >= 1,000,000 either way.
    context_window: int = 0

    @field_validator("api_url", mode="after")
    @classmethod
    def strip_chat_completions_suffix(cls, v: str) -> str:
        # The OpenAI SDK appends `/chat/completions` to `base_url`, so strip a
        # user-supplied suffix to avoid doubling it.
        return v.removesuffix("/chat/completions").rstrip("/")

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, v: object) -> str:
        if isinstance(v, str):
            stripped = v.strip().lower()
            if not stripped:
                raise ValueError(
                    "reasoning_effort is required (noergler needs a reasoning-capable "
                    f"model); set one of {sorted(_REASONING_EFFORT_VALUES)}"
                )
            if stripped not in _REASONING_EFFORT_VALUES:
                raise ValueError(
                    f"reasoning_effort must be one of {sorted(_REASONING_EFFORT_VALUES)}, got {v!r}"
                )
            return stripped
        raise ValueError("reasoning_effort must be a string")


class ReviewConfig(BaseModel):
    auto_review_authors: list[str] = []
    max_comments: int = 25
    max_file_lines: int = 1000
    diff_extra_lines_before: int = 3
    diff_extra_lines_after: int = 2
    diff_max_extra_lines_dynamic_context: int = 10
    diff_allow_dynamic_context: bool = True
    review_prompt_template: str = "prompts/review.txt"
    mention_prompt_template: str = "prompts/mention.txt"
    ticket_compliance_check: bool = True
    require_agents_md: bool = True
    agents_md_warn_tokens: int = 4000
    agents_md_max_tokens: int = 7000
    agents_md_custom_link: str = ""
    opt_out_branch_keyword: str = "noergloff"
    max_pr_cost_usd: float = 5.00

    @field_validator("auto_review_authors", mode="before")
    @classmethod
    def parse_comma_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v


class JiraConfig(BaseModel):
    url: str
    token: str
    acceptance_criteria_prefixes: list[str] = ["AC", "AK", "Acceptance Criteria", "Acceptance Criterion", "Akzeptanzkriterium", "Akzeptanzkriterien", "DoD", "Req"]

    @field_validator("acceptance_criteria_prefixes", mode="before")
    @classmethod
    def parse_comma_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class DatabaseConfig(BaseModel):
    url: str


class RiptideConfig(BaseModel):
    """Per-team forwarding to riptide-collector.

    Present on a team only when both `url` and `token` are set: noergler then
    emits the team's PR rollups and validates the bearer at startup. A team
    without a riptide block forwards nothing.
    """

    url: str
    token: str


# --- Teams -------------------------------------------------------------------
# One instance serves many teams. The instance env (`load_config`) carries what
# is physically one thing — the Bitbucket service account, the Jira user, the
# database, the gateway and its catalog — plus the defaults for every
# team-overridable knob. `teams.yaml` (path `TEAMS_CONFIG`) carries one block
# per team; secrets are never in the file, each `*_env` field names the env
# var holding the value.
#
# One team's bad block never affects another: a per-team fault disables that
# team (`AppConfig.disabled`, reason logged with `team=<slug>`) and the
# instance still boots. Only file-level faults — missing file, unparseable,
# zero teams, duplicate slug — abort startup, because none of them can be
# pinned on a single team.

TEAM_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class TeamConfigError(ValueError):
    """A single team's block is unusable. Disables that team only."""


class TeamsFileError(RuntimeError):
    """`teams.yaml` as a whole is unusable. Aborts startup."""


def team_env_prefix(slug: str) -> str:
    """`TEAM_<SLUG>_` — the naming convention for a team's secret env vars.

    Not enforced by the loader (the file names its vars explicitly); shared
    with the onboarding script and the infra helpers so all three agree.
    """
    return f"TEAM_{slug.upper().replace('-', '_')}_"


class ProjectScope(BaseModel, extra="forbid"):
    """A Bitbucket project a team owns, optionally narrowed to some repos."""

    key: str
    repos: list[str] | None = None

    @field_validator("key", mode="after")
    @classmethod
    def non_empty_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("project key must be non-empty")
        return v.strip()

    @field_validator("repos", mode="after")
    @classmethod
    def non_empty_repos(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError("repos must list at least one slug when present")
        return cleaned

    def owns(self, project_key: str, repo_slug: str) -> bool:
        if project_key != self.key:
            return False
        return self.repos is None or repo_slug in self.repos


class TeamInferenceBlock(BaseModel, extra="forbid"):
    """`inference:` in a team block. `base_url`/`catalog_url` are deliberately
    absent: the gateway is instance-wide, so naming them here is a validation
    error (extra="forbid") and disables the team."""

    api_key_env: str
    model: str | None = None
    reasoning_effort: str | None = None
    context_window: int | None = None


class TeamReviewOverrides(BaseModel, extra="forbid"):
    """`review:` in a team block. Same names as `ReviewConfig` minus the two
    prompt-template paths, which are instance-only (one prompt set for all
    teams) — naming them here disables the team."""

    auto_review_authors: list[str] | None = None
    max_comments: int | None = None
    max_file_lines: int | None = None
    diff_extra_lines_before: int | None = None
    diff_extra_lines_after: int | None = None
    diff_max_extra_lines_dynamic_context: int | None = None
    diff_allow_dynamic_context: bool | None = None
    ticket_compliance_check: bool | None = None
    require_agents_md: bool | None = None
    agents_md_warn_tokens: int | None = None
    agents_md_max_tokens: int | None = None
    agents_md_custom_link: str | None = None
    opt_out_branch_keyword: str | None = None
    max_pr_cost_usd: float | None = None


class TeamJiraOverrides(BaseModel, extra="forbid"):
    acceptance_criteria_prefixes: list[str] | None = None


class TeamRiptideBlock(BaseModel, extra="forbid"):
    url: str
    token_env: str


class TeamBlock(BaseModel, extra="forbid"):
    """One raw entry of `teams.yaml`, before secrets and defaults are resolved."""

    slug: str
    name: str | None = None
    webhook_secret_env: str
    projects: list[ProjectScope]
    inference: TeamInferenceBlock
    review: TeamReviewOverrides | None = None
    jira: TeamJiraOverrides | None = None
    riptide: TeamRiptideBlock | None = None

    @field_validator("slug", mode="after")
    @classmethod
    def valid_slug(cls, v: str) -> str:
        if not TEAM_SLUG_RE.match(v):
            raise ValueError(f"slug {v!r} must match {TEAM_SLUG_RE.pattern}")
        return v

    @field_validator("projects", mode="after")
    @classmethod
    def at_least_one_project(cls, v: list[ProjectScope]) -> list[ProjectScope]:
        if not v:
            raise ValueError("projects must list at least one Bitbucket project key")
        return v


class TeamConfig(BaseModel):
    """A fully resolved team: secrets read, defaults merged, ready to use."""

    slug: str
    name: str
    webhook_secret: str
    projects: list[ProjectScope]
    llm: LLMConfig
    review: ReviewConfig
    jira: JiraConfig
    riptide: RiptideConfig | None = None

    def owns(self, project_key: str, repo_slug: str) -> bool:
        return any(p.owns(project_key, repo_slug) for p in self.projects)


class InstanceDefaults(BaseModel):
    """Instance-level values a team block may override."""

    model: str
    reasoning_effort: str
    context_window: int


class AppConfig(BaseModel):
    bitbucket: BitbucketConfig
    # Instance-wide gateway + defaults. `api_key` is empty here: every team
    # brings its own, there is no instance key to fall back to.
    llm: LLMConfig
    review: ReviewConfig = ReviewConfig()
    jira: JiraConfig
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig
    teams_config_path: str
    # Enabled teams by slug; disabled teams by slug with the reason. A slug is
    # in exactly one of the two.
    teams: dict[str, TeamConfig] = {}
    disabled: dict[str, str] = {}

    def team_for(self, project_key: str, repo_slug: str) -> TeamConfig | None:
        for team in self.teams.values():
            if team.owns(project_key, repo_slug):
                return team
        return None


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


def _secret_from_env(field: str, var: str) -> str:
    """Resolve a `*_env` reference. Empty is rejected: an empty inference key
    or webhook secret is a misconfiguration, never a valid value."""
    if not var.strip():
        raise TeamConfigError(f"{field} must name an environment variable")
    value = os.environ.get(var)
    if value is None:
        raise TeamConfigError(f"{field}: environment variable {var} is not set")
    if not value.strip():
        raise TeamConfigError(f"{field}: environment variable {var} is empty")
    return value


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}" if loc else str(err.get("msg", "invalid")))
    return "; ".join(parts)


def resolve_team(raw: dict[str, Any], instance: AppConfig) -> TeamConfig:
    """Turn one raw `teams.yaml` entry into a `TeamConfig`.

    Raises `TeamConfigError` for anything wrong with this block; the caller
    disables the team and keeps going.
    """
    try:
        block = TeamBlock(**raw)
    except ValidationError as exc:
        raise TeamConfigError(_format_validation_error(exc)) from exc

    webhook_secret = _secret_from_env("webhook_secret_env", block.webhook_secret_env)
    api_key = _secret_from_env("inference.api_key_env", block.inference.api_key_env)

    inference = block.inference
    try:
        llm = LLMConfig(
            model=inference.model if inference.model is not None else instance.llm.model,
            api_key=api_key,
            api_url=instance.llm.api_url,
            reasoning_effort=(
                inference.reasoning_effort
                if inference.reasoning_effort is not None
                else instance.llm.reasoning_effort
            ),
            catalog_url=instance.llm.catalog_url,
            context_window=(
                inference.context_window
                if inference.context_window is not None
                else instance.llm.context_window
            ),
        )
    except ValidationError as exc:
        raise TeamConfigError(f"inference: {_format_validation_error(exc)}") from exc

    overrides = block.review.model_dump(exclude_none=True) if block.review else {}
    try:
        review = ReviewConfig(**{**instance.review.model_dump(), **overrides})
    except ValidationError as exc:
        raise TeamConfigError(f"review: {_format_validation_error(exc)}") from exc

    prefixes = (
        block.jira.acceptance_criteria_prefixes
        if block.jira and block.jira.acceptance_criteria_prefixes is not None
        else instance.jira.acceptance_criteria_prefixes
    )
    jira = JiraConfig(
        url=instance.jira.url, token=instance.jira.token,
        acceptance_criteria_prefixes=prefixes,
    )

    riptide: RiptideConfig | None = None
    if block.riptide is not None:
        if not block.riptide.url.strip():
            raise TeamConfigError("riptide.url must be non-empty")
        riptide = RiptideConfig(
            url=block.riptide.url.strip().rstrip("/"),
            token=_secret_from_env("riptide.token_env", block.riptide.token_env),
        )

    return TeamConfig(
        slug=block.slug,
        name=block.name or block.slug,
        webhook_secret=webhook_secret,
        projects=block.projects,
        llm=llm,
        review=review,
        jira=jira,
        riptide=riptide,
    )


def _read_teams_file(path: str) -> list[dict[str, Any]]:
    """Parse `teams.yaml` into raw team dicts. File-level faults raise
    `TeamsFileError` — the instance must not start without a usable file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise TeamsFileError(
            f"teams file {path} not found. noergler does not start without teams: "
            "set TEAMS_CONFIG to a teams.yaml (see teams.example.yaml)."
        ) from None
    except OSError as exc:
        # e.g. IsADirectoryError: a compose bind-mount of a missing host file
        # creates a directory in its place.
        raise TeamsFileError(f"teams file {path} cannot be read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TeamsFileError(f"teams file {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "teams" not in data:
        raise TeamsFileError(f"teams file {path} must be a mapping with a top-level `teams:` list")
    teams = data["teams"]
    if not isinstance(teams, list) or not teams:
        raise TeamsFileError(f"teams file {path}: `teams:` must be a non-empty list")
    for i, item in enumerate(teams):
        if not isinstance(item, dict):
            raise TeamsFileError(f"teams file {path}: teams[{i}] must be a mapping")
    return teams


def load_teams(path: str, instance: AppConfig) -> tuple[dict[str, TeamConfig], dict[str, str]]:
    """Resolve every team in the file. Returns (enabled, disabled-with-reason).

    Duplicate slugs abort (the fault has no single owner). A project or repo
    claimed by more than one team disables every claimant: ownership is
    ambiguous and `pr_reviews` is keyed by project/repo/pr, so two owners
    would double-review and double-charge.
    """
    log = logging.getLogger(__name__)
    raw_teams = _read_teams_file(path)

    slugs = [str(t.get("slug") or "") for t in raw_teams]
    dupes = sorted({s for s in slugs if s and slugs.count(s) > 1})
    if dupes:
        raise TeamsFileError(f"teams file {path}: duplicate slug(s) {dupes}")

    enabled: dict[str, TeamConfig] = {}
    disabled: dict[str, str] = {}
    for i, raw in enumerate(raw_teams):
        slug = str(raw.get("slug") or f"teams[{i}]")
        try:
            enabled[slug] = resolve_team(raw, instance)
        except TeamConfigError as exc:
            disabled[slug] = str(exc)

    # Ownership conflicts: exact project key, or project+repo, claimed twice.
    # A whole-project claim conflicts with any repo-level claim on that key.
    for a, b in itertools.combinations(list(enabled.values()), 2):
        for pa in a.projects:
            for pb in b.projects:
                if pa.key != pb.key:
                    continue
                overlap = (
                    pa.repos is None or pb.repos is None
                    or bool(set(pa.repos) & set(pb.repos))
                )
                if overlap:
                    reason = f"project {pa.key} is also claimed by team {{other}}"
                    disabled.setdefault(a.slug, reason.format(other=b.slug))
                    disabled.setdefault(b.slug, reason.format(other=a.slug))
    for slug in list(disabled):
        enabled.pop(slug, None)

    for slug, reason in disabled.items():
        # Bound, not just in the message: Splunk extracts `team` as a field.
        structlog.contextvars.bind_contextvars(team=slug)
        try:
            log.error("team_disabled team=%s reason=%s", slug, reason)
        finally:
            structlog.contextvars.unbind_contextvars("team")
    return enabled, disabled


_SECRET_FIELDS = {
    "bitbucket": {"token"},
    "llm": {"api_key"},
    "jira": {"token"},
    "database": {"url"},
}

_TEAM_SECRET_FIELDS = {
    "llm": {"api_key"},
    "jira": {"token"},
    "riptide": {"token"},
}


def _log_section(log: logging.Logger, label: str, section: BaseModel, secrets: set[str]) -> None:
    log.info("[%s]", label)
    for field_name in type(section).model_fields:
        value = getattr(section, field_name)
        display = "***" if field_name in secrets else value
        log.info("  %s = %s", field_name, display)


def log_config(config: AppConfig, log: logging.Logger) -> None:
    for section_name in ("bitbucket", "llm", "review", "jira", "server", "database"):
        section = getattr(config, section_name)
        _log_section(log, f"config.{section_name}", section, _SECRET_FIELDS.get(section_name, set()))
    log.info("[config.teams] path = %s", config.teams_config_path)
    for slug, team in config.teams.items():
        log.info("[config.teams.%s] name = %s", slug, team.name)
        log.info("  webhook_secret = ***")
        log.info("  projects = %s", [
            p.key if p.repos is None else f"{p.key}/{{{','.join(p.repos)}}}" for p in team.projects
        ])
        _log_section(log, f"config.teams.{slug}.llm", team.llm, _TEAM_SECRET_FIELDS["llm"])
        _log_section(log, f"config.teams.{slug}.review", team.review, set())
        _log_section(log, f"config.teams.{slug}.jira", team.jira, _TEAM_SECRET_FIELDS["jira"])
        if team.riptide is not None:
            _log_section(log, f"config.teams.{slug}.riptide", team.riptide, _TEAM_SECRET_FIELDS["riptide"])
        else:
            log.info("[config.teams.%s.riptide] disabled", slug)
    for slug, reason in config.disabled.items():
        log.error("[config.teams.%s] DISABLED: %s", slug, reason)


def load_instance_config() -> AppConfig:
    """Layer 1 only: the instance env, no teams resolved yet."""
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url=_env("BITBUCKET_URL"),
            token=_env("BITBUCKET_TOKEN"),
            username=_env("BITBUCKET_USERNAME"),
        ),
        llm=LLMConfig(
            model=_env("OPENAI_MODEL", "gpt-5.4"),
            api_key="",
            api_url=_env("OPENAI_BASE_URL"),
            reasoning_effort=_env("OPENAI_REASONING_EFFORT", "high"),
            catalog_url=_env("MODEL_CATALOG_URL"),
            context_window=int(_env("OPENAI_CONTEXT_WINDOW", "0")),
        ),
        review=ReviewConfig(
            auto_review_authors=[a.strip() for a in _env("REVIEW_AUTO_REVIEW_AUTHORS", "").split(",") if a.strip()],
            max_comments=int(_env("REVIEW_MAX_COMMENTS", "25")),
            max_file_lines=int(_env("REVIEW_MAX_FILE_LINES", "1000")),
            diff_extra_lines_before=int(_env("REVIEW_DIFF_EXTRA_LINES_BEFORE", "3")),
            diff_extra_lines_after=int(_env("REVIEW_DIFF_EXTRA_LINES_AFTER", "2")),
            diff_max_extra_lines_dynamic_context=int(_env("REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT", "10")),
            diff_allow_dynamic_context=_env("REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT", "true").lower() in ("true", "1", "yes"),
            review_prompt_template=_env("REVIEW_PROMPT_TEMPLATE", "prompts/review.txt"),
            mention_prompt_template=_env("REVIEW_MENTION_PROMPT_TEMPLATE", "prompts/mention.txt"),
            ticket_compliance_check=_env("REVIEW_TICKET_COMPLIANCE_CHECK", "true").lower() in ("true", "1", "yes"),
            require_agents_md=_env("REVIEW_REQUIRE_AGENTS_MD", "true").lower() in ("true", "1", "yes"),
            agents_md_warn_tokens=int(_env("REVIEW_AGENTS_MD_WARN_TOKENS", "4000")),
            agents_md_max_tokens=int(_env("REVIEW_AGENTS_MD_MAX_TOKENS", "7000")),
            agents_md_custom_link=_env("REVIEW_AGENTS_MD_CUSTOM_LINK", ""),
            opt_out_branch_keyword=_env("REVIEW_OPT_OUT_BRANCH_KEYWORD", "noergloff"),
            max_pr_cost_usd=float(_env("REVIEW_MAX_PR_COST_USD", "5.00")),
        ),
        jira=JiraConfig(
            url=_env("JIRA_URL"),
            token=_env("JIRA_TOKEN"),
            acceptance_criteria_prefixes=[p.strip() for p in _env("JIRA_ACCEPTANCE_CRITERIA_PREFIXES", "AC,AK,Acceptance Criteria,Acceptance Criterion,Akzeptanzkriterium,Akzeptanzkriterien,DoD,Req").split(",") if p.strip()],
        ),
        server=ServerConfig(
            host=_env("SERVER_HOST", "0.0.0.0"),
            port=int(_env("SERVER_PORT", "8080")),
        ),
        database=DatabaseConfig(
            url=_env("DATABASE_URL"),
        ),
        teams_config_path=_env("TEAMS_CONFIG", "teams.yaml"),
    )


def load_config() -> AppConfig:
    """Instance env plus every team from `TEAMS_CONFIG`."""
    config = load_instance_config()
    teams, disabled = load_teams(config.teams_config_path, config)
    config.teams = teams
    config.disabled = disabled
    return config
