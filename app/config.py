import logging
import os

from pydantic import BaseModel, field_validator

# Webhook events the /webhook endpoint in app/main.py dispatches on.
# Kept here so the provisioning script and the service stay in sync.
REQUIRED_WEBHOOK_EVENTS: tuple[str, ...] = (
    "pr:opened",
    "pr:from_ref_updated",
    "pr:comment:added",
    "pr:merged",
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


# GitHub Copilot per-model token pricing, USD per 1M tokens.
# Effective 2026-06-01 (usage-based billing). Source:
# https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
# We can't bill cached input separately because the Copilot LLM response
# doesn't expose cached-token counts — see estimate_cost_usd().
class ModelPrice(BaseModel):
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float


_MODEL_PRICING: dict[str, ModelPrice] = {
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


def pricing_for(model: str) -> ModelPrice | None:
    """Return the price entry for a model id, or None if unknown.

    Falls back to a prefix match so dated/suffixed ids resolve to the base
    model entry (mirrors _context_window_for in app/llm_client.py).
    """
    if model in _MODEL_PRICING:
        return _MODEL_PRICING[model]
    # Iterate longest base first so `gpt-5.4-mini-2025-06-01` matches
    # `gpt-5.4-mini` instead of falling through to the (3x more expensive)
    # `gpt-5.4` entry.
    for base in sorted(_MODEL_PRICING, key=len, reverse=True):
        if model.startswith(base + "-"):
            return _MODEL_PRICING[base]
    return None


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Upper-bound USD cost for one LLM call.

    The Copilot API doesn't return a cached-tokens breakdown, so all prompt
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
    oauth_token: str
    api_url: str = "https://api.business.githubcopilot.com"
    integration_id: str = "vscode-chat"
    # Identity headers Copilot Business/Enterprise validates against. Pin to
    # current GitHub-Copilot-Chat-shaped values (matches the blessed opencode
    # plugin) so we land on the same routing path the official client gets.
    # Bump these together when rotating to a newer working set.
    editor_version: str = "vscode/1.109.2"
    editor_plugin_version: str = "copilot-chat/0.37.5"
    user_agent: str = "GitHubCopilotChat/0.37.5"
    github_api_version: str = "2025-10-01"
    # vscode-abexpcontext experiment-flag blob. Empty by default — opt in via
    # COPILOT_ABEXP_CONTEXT when a working value is needed (the flags rotate
    # so we don't ship a hardcoded default that may go stale).
    abexp_context: str = ""
    reasoning_effort: str | None = "high"

    @field_validator("api_url", mode="after")
    @classmethod
    def strip_chat_completions_suffix(cls, v: str) -> str:
        return v.removesuffix("/chat/completions").rstrip("/")

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip().lower()
            if not stripped:
                return None
            if stripped not in _REASONING_EFFORT_VALUES:
                raise ValueError(
                    f"reasoning_effort must be one of {sorted(_REASONING_EFFORT_VALUES)}, got {v!r}"
                )
            return stripped
        raise ValueError("reasoning_effort must be a string or None")


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
    opt_out_branch_keyword: str = "noergloff"

    @field_validator("auto_review_authors", mode="before")
    @classmethod
    def parse_comma_list(cls, v):
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v


class JiraConfig(BaseModel):
    url: str
    token: str
    acceptance_criteria_prefixes: list[str] = ["AC", "AK", "Acceptance Criteria", "Acceptance Criterion", "Akzeptanzkriterium", "Akzeptanzkriterien", "DoD", "Req"]

    @field_validator("acceptance_criteria_prefixes", mode="before")
    @classmethod
    def parse_comma_list(cls, v):
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class DatabaseConfig(BaseModel):
    url: str


class AnalyticsConfig(BaseModel):
    api_key: str = ""  # empty -> /analytics endpoints return 503


class AppConfig(BaseModel):
    bitbucket: BitbucketConfig
    llm: LLMConfig
    review: ReviewConfig = ReviewConfig()
    jira: JiraConfig
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig
    analytics: AnalyticsConfig = AnalyticsConfig()


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


_SECRET_FIELDS = {
    "bitbucket": {"token", "webhook_secret"},
    "llm": {"oauth_token"},
    "jira": {"token"},
    "database": {"url"},
    "analytics": {"api_key"},
}


def log_config(config: AppConfig, log: logging.Logger) -> None:
    for section_name in ("bitbucket", "llm", "review", "jira", "server", "database", "analytics"):
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
            model=_env("COPILOT_MODEL", "gpt-5.4"),
            oauth_token=_env("COPILOT_OAUTH_TOKEN"),
            api_url=_env("COPILOT_API_URL", "https://api.business.githubcopilot.com"),
            integration_id=_env("COPILOT_INTEGRATION_ID", "vscode-chat"),
            editor_version=_env("COPILOT_EDITOR_VERSION", "vscode/1.109.2"),
            editor_plugin_version=_env("COPILOT_EDITOR_PLUGIN_VERSION", "copilot-chat/0.37.5"),
            user_agent=_env("COPILOT_USER_AGENT", "GitHubCopilotChat/0.37.5"),
            github_api_version=_env("COPILOT_GITHUB_API_VERSION", "2025-10-01"),
            abexp_context=_env("COPILOT_ABEXP_CONTEXT", ""),
            reasoning_effort=os.environ.get("COPILOT_REASONING_EFFORT") or "high",
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
            opt_out_branch_keyword=_env("REVIEW_OPT_OUT_BRANCH_KEYWORD", "noergloff"),
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
        analytics=AnalyticsConfig(
            api_key=_env("ANALYTICS_API_KEY", ""),
        ),
    )
