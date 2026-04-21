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


class LLMConfig(BaseModel):
    model: str = "gpt-5.3-codex"
    oauth_token: str
    api_url: str = "https://api.business.githubcopilot.com"
    integration_id: str = "vscode-chat"
    editor_version: str = "vscode/1.99.0"

    @field_validator("api_url", mode="after")
    @classmethod
    def strip_chat_completions_suffix(cls, v: str) -> str:
        return v.removesuffix("/chat/completions").rstrip("/")


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


class AppConfig(BaseModel):
    bitbucket: BitbucketConfig
    llm: LLMConfig
    review: ReviewConfig = ReviewConfig()
    jira: JiraConfig
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig


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
}


def log_config(config: AppConfig, log: logging.Logger) -> None:
    for section_name in ("bitbucket", "llm", "review", "jira", "server", "database"):
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
            model=_env("COPILOT_MODEL", "gpt-5.3-codex"),
            oauth_token=_env("COPILOT_OAUTH_TOKEN"),
            api_url=_env("COPILOT_API_URL", "https://api.business.githubcopilot.com"),
            integration_id=_env("COPILOT_INTEGRATION_ID", "vscode-chat"),
            editor_version=_env("COPILOT_EDITOR_VERSION", "vscode/1.99.0"),
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
    )
