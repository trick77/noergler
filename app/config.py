import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

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
    webhook_secret: str
    username: str


_REASONING_EFFORT_VALUES = frozenset({"minimal", "low", "medium", "high"})


def model_label(model: str, reasoning_effort: str | None) -> str:
    if reasoning_effort:
        return f"{model}-{reasoning_effort}"
    return model


# Per-model token pricing, USD per 1M tokens. Static fallback used until the
# LiteLLM pricing table refresh overrides it (see LITELLM_PRICING_URL /
# pricing_for). We can't bill cached input separately because the LLM response
# doesn't expose cached-token counts — see estimate_cost_usd().
class ModelPrice(BaseModel):
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float


_STATIC_MODEL_PRICING: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4.1":        ModelPrice(input_per_mtok=2.00, cached_input_per_mtok=0.50,  output_per_mtok=8.00),
    "gpt-5-mini":     ModelPrice(input_per_mtok=0.25, cached_input_per_mtok=0.025, output_per_mtok=2.00),
    "gpt-5.2":        ModelPrice(input_per_mtok=1.75, cached_input_per_mtok=0.175, output_per_mtok=14.00),
    "gpt-5.2-codex":  ModelPrice(input_per_mtok=1.75, cached_input_per_mtok=0.175, output_per_mtok=14.00),
    "gpt-5.3-codex":  ModelPrice(input_per_mtok=1.75, cached_input_per_mtok=0.175, output_per_mtok=14.00),
    "gpt-5.4":        ModelPrice(input_per_mtok=2.50, cached_input_per_mtok=0.25,  output_per_mtok=15.00),
    "gpt-5.4-mini":   ModelPrice(input_per_mtok=0.75, cached_input_per_mtok=0.075, output_per_mtok=4.50),
    "gpt-5.4-nano":   ModelPrice(input_per_mtok=0.20, cached_input_per_mtok=0.02,  output_per_mtok=1.25),
    "gpt-5.5":        ModelPrice(input_per_mtok=5.00, cached_input_per_mtok=0.50,  output_per_mtok=30.00),
    # Anthropic
    "claude-haiku-4.5":  ModelPrice(input_per_mtok=1.00, cached_input_per_mtok=0.10, output_per_mtok=5.00),
    "claude-sonnet-4":   ModelPrice(input_per_mtok=3.00, cached_input_per_mtok=0.30, output_per_mtok=15.00),
    "claude-sonnet-4.5": ModelPrice(input_per_mtok=3.00, cached_input_per_mtok=0.30, output_per_mtok=15.00),
    "claude-sonnet-4.6": ModelPrice(input_per_mtok=3.00, cached_input_per_mtok=0.30, output_per_mtok=15.00),
    "claude-opus-4.5":   ModelPrice(input_per_mtok=5.00, cached_input_per_mtok=0.50, output_per_mtok=25.00),
    "claude-opus-4.6":   ModelPrice(input_per_mtok=5.00, cached_input_per_mtok=0.50, output_per_mtok=25.00),
    "claude-opus-4.7":   ModelPrice(input_per_mtok=5.00, cached_input_per_mtok=0.50, output_per_mtok=25.00),
}

# Live pricing table. Initially the static defaults; replaced wholesale
# (atomic reference swap under the GIL) by `_swap_pricing` whenever a refresh
# completes. `pricing_for` snapshots this reference at call time so a swap
# in flight never tears a single lookup.
_MODEL_PRICING: dict[str, ModelPrice] = dict(_STATIC_MODEL_PRICING)


def _swap_pricing(new_table: dict[str, ModelPrice]) -> None:
    global _MODEL_PRICING
    _MODEL_PRICING = new_table  # pyright: ignore[reportConstantRedefinition]


def pricing_for(model: str) -> ModelPrice | None:
    """Return the price entry for a model id, or None if unknown.

    Falls back to a prefix match so dated/suffixed ids resolve to the base
    model entry (mirrors _context_window_for in app/llm_client.py).
    """
    table = _MODEL_PRICING  # snapshot to survive an atomic refresh swap
    if model in table:
        return table[model]
    # Iterate longest base first so `gpt-5.4-mini-2025-06-01` matches
    # `gpt-5.4-mini` instead of falling through to the (3x more expensive)
    # `gpt-5.4` entry.
    for base in sorted(table, key=len, reverse=True):
        if model.startswith(base + "-"):
            return table[base]
    return None


# --- Model context windows -------------------------------------------------
# Total input context window per model, in tokens. Sourced from the same
# LiteLLM catalog as pricing (`max_input_tokens`); these static values are the
# offline-safe fallback used before the first fetch / when LiteLLM is down.
#
# NOTE: 272k for gpt-5.5 / gpt-5.4 was a long-standing bug — 272_000 is OpenAI's
# *pricing threshold* (2x input above it), not the context window. The real
# window is 1_050_000 (LiteLLM `max_input_tokens`). The other entries were and
# remain correct.
_STATIC_MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-5": 272_000,
    "gpt-5-mini": 272_000,
    "gpt-5.3-codex": 272_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.5": 1_050_000,
    "claude-sonnet-4": 200_000,
    "claude-sonnet-4.5": 200_000,
    "claude-opus-4": 200_000,
    "claude-opus-4.1": 200_000,
    "claude-haiku-4.5": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
}

# Live context-window table. Same atomic-swap pattern as `_MODEL_PRICING`:
# starts at the static defaults, replaced wholesale by the refresher.
_MODEL_CONTEXT_WINDOW: dict[str, int] = dict(_STATIC_MODEL_CONTEXT_WINDOW)


def _swap_context_windows(new_table: dict[str, int]) -> None:
    global _MODEL_CONTEXT_WINDOW
    _MODEL_CONTEXT_WINDOW = new_table  # pyright: ignore[reportConstantRedefinition]


def context_window_for(model: str) -> int | None:
    """Return the input context window for a model id, or None if unknown.

    Exact match first, then a longest-base prefix match so dated/suffixed ids
    resolve to the base model entry — mirrors `pricing_for`.
    """
    table = _MODEL_CONTEXT_WINDOW  # snapshot to survive an atomic refresh swap
    if model in table:
        return table[model]
    for base in sorted(table, key=len, reverse=True):
        if model.startswith(base + "-"):
            return table[base]
    return None


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


LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# LiteLLM exposes Anthropic models only under provider-prefixed keys. Probe
# these in order so e.g. `claude-sonnet-4.6` resolves to the openrouter entry.
_LITELLM_KEY_PREFIXES: tuple[str, ...] = (
    "",
    "openrouter/anthropic/",
    "vercel_ai_gateway/anthropic/",
)


def _build_pricing_from_litellm(data: dict[str, Any]) -> dict[str, ModelPrice]:
    """Overlay LiteLLM pricing onto static defaults, returning a fresh dict.

    Only ids present in `_STATIC_MODEL_PRICING` are looked up; ids missing
    from LiteLLM keep their static price.
    """
    log = logging.getLogger(__name__)
    table: dict[str, ModelPrice] = dict(_STATIC_MODEL_PRICING)
    for model_id in _STATIC_MODEL_PRICING:
        entry = None
        for prefix in _LITELLM_KEY_PREFIXES:
            candidate = data.get(f"{prefix}{model_id}")
            if candidate and "input_cost_per_token" in candidate:
                entry = candidate
                break
        if entry is None:
            continue
        try:
            input_per_mtok = float(entry["input_cost_per_token"]) * 1_000_000
            output_per_mtok = float(entry["output_cost_per_token"]) * 1_000_000
            cached_raw = entry.get("cache_read_input_token_cost")
            cached_per_mtok = (
                float(cached_raw) * 1_000_000 if cached_raw is not None
                else input_per_mtok * 0.1
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("malformed LiteLLM entry for %s: %s", model_id, exc)
            continue
        table[model_id] = ModelPrice(
            input_per_mtok=input_per_mtok,
            cached_input_per_mtok=cached_per_mtok,
            output_per_mtok=output_per_mtok,
        )
    return table


def apply_pricing_overlay(entries: dict[str, tuple[float, float, float]]) -> int:
    """Replace the live pricing table by overlaying `entries` on the static
    defaults. Used by the DB hydrator at startup. Returns count overlaid.
    """
    table: dict[str, ModelPrice] = dict(_STATIC_MODEL_PRICING)
    overlaid = 0
    for model_id, (inp, cached, out) in entries.items():
        if model_id not in _STATIC_MODEL_PRICING:
            continue
        table[model_id] = ModelPrice(
            input_per_mtok=inp,
            cached_input_per_mtok=cached,
            output_per_mtok=out,
        )
        overlaid += 1
    _swap_pricing(table)
    return overlaid


def _build_context_windows_from_litellm(data: dict[str, Any]) -> dict[str, int]:
    """Overlay LiteLLM `max_input_tokens` onto static defaults, fresh dict.

    Only ids present in `_STATIC_MODEL_CONTEXT_WINDOW` are looked up; ids missing
    from LiteLLM (or with a malformed/zero value) keep their static window.
    Mirrors `_build_pricing_from_litellm`.
    """
    log = logging.getLogger(__name__)
    table: dict[str, int] = dict(_STATIC_MODEL_CONTEXT_WINDOW)
    for model_id in _STATIC_MODEL_CONTEXT_WINDOW:
        entry = None
        for prefix in _LITELLM_KEY_PREFIXES:
            candidate = data.get(f"{prefix}{model_id}")
            if candidate and "max_input_tokens" in candidate:
                entry = candidate
                break
        if entry is None:
            continue
        try:
            window = int(entry["max_input_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("malformed LiteLLM max_input_tokens for %s: %s", model_id, exc)
            continue
        if window > 0:
            table[model_id] = window
    return table


async def _fetch_litellm_json(timeout: float) -> dict[str, Any] | None:
    """GET the LiteLLM catalog once and return the parsed JSON, or None."""
    log = logging.getLogger(__name__)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(LITELLM_PRICING_URL)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        log.warning("model-meta fetch failed: %s", exc)
        return None


async def fetch_litellm_pricing(timeout: float = 5.0) -> dict[str, ModelPrice] | None:
    """Fetch LiteLLM JSON and return a fresh pricing overlay, or None on failure.

    Async so it can run inside the FastAPI lifespan / background task without
    blocking the event loop.
    """
    data = await _fetch_litellm_json(timeout)
    if data is None:
        return None
    return _build_pricing_from_litellm(data)


async def fetch_litellm_model_meta(
    timeout: float = 5.0,
) -> tuple[dict[str, ModelPrice], dict[str, int]] | None:
    """One LiteLLM fetch → (pricing overlay, context-window overlay), or None.

    Lets the refresher update both tables from a single HTTP round-trip.
    """
    data = await _fetch_litellm_json(timeout)
    if data is None:
        return None
    return _build_pricing_from_litellm(data), _build_context_windows_from_litellm(data)


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Upper-bound USD cost for one LLM call.

    The API doesn't return a cached-tokens breakdown, so all prompt
    tokens are billed at the full input rate. Real bill on follow-up runs
    will be lower because prompt cache hits are charged at the cached rate.
    """
    price = pricing_for(model)
    if price is None:
        return None
    return (
        prompt_tokens * price.input_per_mtok
        + completion_tokens * price.output_per_mtok
    ) / 1_000_000


class LLMConfig(BaseModel):
    model: str = "gpt-5.4"
    api_key: str
    api_url: str
    # noergler requires a reasoning-capable model, so reasoning_effort is
    # mandatory — an empty value is rejected rather than silently disabling it.
    reasoning_effort: str = "high"
    # Explicit context window (tokens). 0 = auto-detect from the LiteLLM table.
    # Set this for custom proxy aliases absent from the table; the startup guard
    # requires the resolved window to be >= 1,000,000.
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
    """Optional forwarding to riptide-collector.

    When both `url` and `token` are non-empty, noergler emits review-cost
    and feedback events to riptide and validates the bearer at startup.
    Empty values disable forwarding entirely.
    """

    url: str = ""
    token: str = ""


class AppConfig(BaseModel):
    bitbucket: BitbucketConfig
    llm: LLMConfig
    review: ReviewConfig = ReviewConfig()
    jira: JiraConfig
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig
    riptide: RiptideConfig = RiptideConfig()


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


_SECRET_FIELDS = {
    "bitbucket": {"token", "webhook_secret"},
    "llm": {"api_key"},
    "jira": {"token"},
    "database": {"url"},
    "riptide": {"token"},
}


def log_config(config: AppConfig, log: logging.Logger) -> None:
    for section_name in ("bitbucket", "llm", "review", "jira", "server", "database", "riptide"):
        section = getattr(config, section_name)
        secrets = _SECRET_FIELDS.get(section_name, set())
        log.info("[config.%s]", section_name)
        for field_name in section.model_fields:
            value = getattr(section, field_name)
            display = "***" if field_name in secrets else value
            log.info("  %s = %s", field_name, display)


def load_config() -> AppConfig:
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url=_env("BITBUCKET_URL"),
            token=_env("BITBUCKET_TOKEN"),
            webhook_secret=_env("BITBUCKET_WEBHOOK_SECRET"),
            username=_env("BITBUCKET_USERNAME"),
        ),
        llm=LLMConfig(
            model=_env("OPENAI_MODEL", "gpt-5.4"),
            api_key=_env("OPENAI_API_KEY"),
            api_url=_env("OPENAI_BASE_URL"),
            reasoning_effort=_env("OPENAI_REASONING_EFFORT", "high"),
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
        riptide=RiptideConfig(
            url=_env("RIPTIDE_URL", ""),
            token=_env("RIPTIDE_TOKEN", ""),
        ),
    )
