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


# --- Model catalog ---------------------------------------------------------
# Pricing and context windows come from LiteLLM's public catalog, fetched over
# the network. There is no baked-in fallback table and no DB cache: the catalog
# is the single source of truth. Startup resolves the configured model against
# it exactly once and aborts if that fails (see `resolve_or_raise`), so the rest
# of the process can assume a priced, sized model.
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# LiteLLM exposes some providers only under prefixed keys. Probe these in order
# so e.g. `claude-sonnet-4.6` resolves to the openrouter entry.
_LITELLM_KEY_PREFIXES: tuple[str, ...] = (
    "",
    "openrouter/anthropic/",
    "vercel_ai_gateway/anthropic/",
)


class ModelCatalogEntry(BaseModel):
    """One resolved LiteLLM catalog entry: prices in USD per 1M tokens.

    `cached_input_per_mtok` is LiteLLM's `cache_read_input_token_cost`, the rate
    charged for prompt-cache hits. It is a real billed rate, not an estimate —
    see `estimate_cost_usd`.
    """

    # The id we asked for (`catalog_model`).
    model_id: str
    # The catalog key that actually matched. Differs from `model_id` on a
    # provider-prefixed key (`openrouter/anthropic/...`) or a prefix fallback
    # (`gpt-5.4-mini-2025-06-01` -> `gpt-5.4-mini`). Kept distinct because the
    # fallback searches the whole catalog (~3000 ids, including deprecated and
    # regional variants at different rates), so a wrong-but-plausible match
    # would otherwise be invisible — it prices and boots cleanly.
    matched_key: str
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    max_input_tokens: int

    @property
    def supports_caching(self) -> bool:
        return self.cached_input_per_mtok < self.input_per_mtok


def _parse_catalog_entry(
    model_id: str, matched_key: str, raw: dict[str, Any]
) -> ModelCatalogEntry | None:
    """Build an entry from one LiteLLM record, or None if it's unusable.

    NOTE on tiered pricing: several large-context models also carry
    `*_cost_per_token_above_272k_tokens` keys charging up to 2x above a prompt
    threshold. Those are deliberately ignored — LiteLLM's JSON does not say
    whether the higher rate applies to the whole request or only to the excess,
    and guessing would trade an honest understatement for a confident wrong
    number. Costs for prompts above the threshold are therefore a lower bound.
    """
    log = logging.getLogger(__name__)
    try:
        input_per_mtok = float(raw["input_cost_per_token"]) * 1_000_000
        output_per_mtok = float(raw["output_cost_per_token"]) * 1_000_000
        window = int(raw["max_input_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("malformed LiteLLM entry for %s: %s", model_id, exc)
        return None
    if window <= 0:
        log.warning("LiteLLM entry for %s has non-positive max_input_tokens", model_id)
        return None
    cached_raw = raw.get("cache_read_input_token_cost")
    try:
        # No cache-read rate published means the model has no prompt cache we
        # can bill separately — charge cache hits at the full input rate rather
        # than inventing a discount.
        cached_per_mtok = (
            float(cached_raw) * 1_000_000 if cached_raw is not None else input_per_mtok
        )
    except (TypeError, ValueError):
        cached_per_mtok = input_per_mtok
    return ModelCatalogEntry(
        model_id=model_id,
        matched_key=matched_key,
        input_per_mtok=input_per_mtok,
        cached_input_per_mtok=cached_per_mtok,
        output_per_mtok=output_per_mtok,
        max_input_tokens=window,
    )


def resolve_catalog_entry(
    data: dict[str, Any], model_id: str
) -> ModelCatalogEntry | None:
    """Find `model_id` in a fetched catalog, or None.

    Tries each provider prefix on the exact id first, then falls back to the
    longest catalog key that `model_id` extends, so a dated id like
    `gpt-5.4-mini-2025-06-01` resolves to `gpt-5.4-mini` rather than the
    shorter, more expensive `gpt-5.4`.
    """
    for prefix in _LITELLM_KEY_PREFIXES:
        key = f"{prefix}{model_id}"
        raw = data.get(key)
        if isinstance(raw, dict) and "input_cost_per_token" in raw:
            return _parse_catalog_entry(model_id, key, raw)
    candidates = [
        key for key in data
        if model_id.startswith(key + "-") and isinstance(data[key], dict)
    ]
    for key in sorted(candidates, key=len, reverse=True):
        raw = data[key]
        if "input_cost_per_token" in raw:
            return _parse_catalog_entry(model_id, key, raw)
    return None


# The live entry for the configured model. Installed by `resolve_or_raise` at
# startup, replaced wholesale by the 24h refresher. Readers snapshot the
# reference (atomic under the GIL) so a swap mid-flight never tears a lookup.
_ACTIVE_ENTRY: ModelCatalogEntry | None = None


def _swap_active_entry(entry: ModelCatalogEntry) -> None:
    global _ACTIVE_ENTRY
    _ACTIVE_ENTRY = entry  # pyright: ignore[reportConstantRedefinition]


def active_entry() -> ModelCatalogEntry | None:
    """The catalog entry for the configured model, or None before startup."""
    return _ACTIVE_ENTRY


async def fetch_model_catalog(timeout: float = 10.0) -> dict[str, Any] | None:
    """GET the LiteLLM catalog once and return the parsed JSON, or None."""
    log = logging.getLogger(__name__)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(LITELLM_PRICING_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("model catalog fetch failed: %s", exc)
        return None
    if not isinstance(data, dict):
        log.warning("model catalog is not a JSON object")
        return None
    return data


class ModelCatalogError(RuntimeError):
    """The configured model could not be resolved against the LiteLLM catalog."""


async def resolve_or_raise(model_id: str, timeout: float = 10.0) -> ModelCatalogEntry:
    """Fetch the catalog and install the entry for `model_id`, or raise.

    Called once at startup. Both failure modes are fatal by design: without a
    catalog entry noergler has no context window to size the review against and
    no rates to price it with, and silently guessing either is worse than not
    starting. The 24h refresh is the opposite — best-effort, keeping the entry
    installed here when a later fetch fails.
    """
    data = await fetch_model_catalog(timeout)
    if data is None:
        raise ModelCatalogError(
            f"could not fetch the model catalog from {LITELLM_PRICING_URL}. "
            "noergler reads pricing and context windows from it at startup and "
            "keeps no local fallback — check network/proxy egress to raw.githubusercontent.com."
        )
    entry = resolve_catalog_entry(data, model_id)
    if entry is None:
        raise ModelCatalogError(
            f"model `{model_id}` is not in the LiteLLM catalog ({len(data)} entries). "
            "Set OPENAI_BASE_MODEL to the upstream model your OPENAI_MODEL alias "
            "maps to, spelled exactly as the catalog spells it."
        )
    _swap_active_entry(entry)
    return entry


async def refresh_active_entry(model_id: str, timeout: float = 10.0) -> bool:
    """Re-resolve `model_id` and swap the entry in. Best-effort.

    Returns False and leaves the installed entry untouched on any failure — a
    refresh must never take a running instance down, unlike the startup resolve.
    """
    log = logging.getLogger(__name__)
    data = await fetch_model_catalog(timeout)
    if data is None:
        return False
    entry = resolve_catalog_entry(data, model_id)
    if entry is None:
        log.warning(
            "model catalog refresh: `%s` vanished from the catalog — keeping the "
            "entry loaded at startup", model_id,
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
    """Token counts for one LLM call.

    `cached` is the subset of `prompt` that hit the provider's prompt cache
    (`usage.prompt_tokens_details.cached_tokens`). Endpoints that don't report
    it leave it at 0, which prices every prompt token at the full input rate —
    the old upper-bound behaviour.
    """

    prompt: int = 0
    cached: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    @property
    def uncached_prompt(self) -> int:
        # Clamp: a proxy reporting cached > prompt must not produce a negative
        # billable count.
        return max(0, self.prompt - self.cached)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt=self.prompt + other.prompt,
            cached=self.cached + other.cached,
            completion=self.completion + other.completion,
        )


def estimate_cost_usd(
    usage: TokenUsage, entry: ModelCatalogEntry | None = None
) -> float | None:
    """USD cost for one LLM call, or None if no catalog entry is installed.

    Cache hits are billed at the model's published `cache_read_input_token_cost`
    rather than the full input rate, so this tracks the real invoice instead of
    the upper bound the old estimate produced. Prompts above a model's tiered
    pricing threshold are still understated — see `_parse_catalog_entry`.
    """
    price = entry if entry is not None else _ACTIVE_ENTRY
    if price is None:
        return None
    return (
        usage.uncached_prompt * price.input_per_mtok
        + usage.cached * price.cached_input_per_mtok
        + usage.completion * price.output_per_mtok
    ) / 1_000_000


class LLMConfig(BaseModel):
    model: str = "gpt-5.4"
    api_key: str
    api_url: str
    # noergler requires a reasoning-capable model, so reasoning_effort is
    # mandatory — an empty value is rejected rather than silently disabling it.
    reasoning_effort: str = "high"
    # Upstream model id this deployment's `model` is an alias for, e.g.
    # `gpt-5.5` for a gateway alias `ai-gateway-gpt-5.5`. Only ever a lookup key
    # for the pricing / context-window tables — never sent on the wire, never
    # shown as the model label (the alias is what actually ran). Empty = the
    # alias is itself a catalog id. Mirrors LiteLLM's own `model_info.base_model`.
    base_model: str = ""
    # Explicit context window (tokens). 0 = auto-detect from the LiteLLM table.
    # Set this for custom proxy aliases absent from the table; the startup guard
    # requires the resolved window to be >= 1,000,000. Usually unnecessary once
    # `base_model` is set, since the base model resolves in the table.
    context_window: int = 0

    @property
    def catalog_model(self) -> str:
        """Model id to look up in the pricing / context-window tables.

        Lookup only — see `base_model`. Everything user-visible or on the wire
        keeps using `model`.
        """
        return self.base_model or self.model

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
            base_model=_env("OPENAI_BASE_MODEL", ""),
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
