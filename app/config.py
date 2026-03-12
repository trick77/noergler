import logging
import os

from pydantic import BaseModel, field_validator


class BitbucketConfig(BaseModel):
    base_url: str
    token: str
    webhook_secret: str


class CopilotConfig(BaseModel):
    model: str = "openai/gpt-5"
    github_token: str
    api_url: str = "https://models.github.ai/inference/chat/completions"
    max_tokens_per_chunk: int = 80000


class ReviewConfig(BaseModel):
    auto_review_authors: list[str] = []
    max_comments: int = 25
    max_lines_per_file: int = 1000
    context_lines: int = 0
    review_prompt_template: str = "prompts/review.txt"
    ramsay_authors: list[str] = []
    mention_trigger: str = "noergler"
    mention_prompt_template: str = "prompts/mention.txt"
    optimize_diff_tokens: bool = True

    @field_validator("auto_review_authors", "ramsay_authors", mode="before")
    @classmethod
    def parse_comma_list(cls, v):
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class AppConfig(BaseModel):
    bitbucket: BitbucketConfig
    copilot: CopilotConfig
    review: ReviewConfig = ReviewConfig()
    server: ServerConfig = ServerConfig()


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


_SECRET_FIELDS = {
    "bitbucket": {"token", "webhook_secret"},
    "copilot": {"github_token"},
}


def log_config(config: AppConfig, log: logging.Logger) -> None:
    for section_name in ("bitbucket", "copilot", "review", "server"):
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
        ),
        copilot=CopilotConfig(
            model=_env("COPILOT_MODEL", "openai/gpt-5"),
            github_token=_env("GITHUB_TOKEN"),
            api_url=_env("COPILOT_API_URL", "https://models.github.ai/inference/chat/completions"),
            max_tokens_per_chunk=int(_env("COPILOT_MAX_TOKENS_PER_CHUNK", "80000")),
        ),
        review=ReviewConfig(
            auto_review_authors=_env("REVIEW_AUTO_REVIEW_AUTHORS", ""),
            max_comments=int(_env("REVIEW_MAX_COMMENTS", "25")),
            max_lines_per_file=int(_env("REVIEW_MAX_LINES_PER_FILE", "1000")),
            context_lines=int(_env("REVIEW_CONTEXT_LINES", "0")),
            review_prompt_template=_env("REVIEW_PROMPT_TEMPLATE", "prompts/review.txt"),
            ramsay_authors=_env("REVIEW_RAMSAY_AUTHORS", ""),
            mention_trigger=_env("REVIEW_MENTION_TRIGGER", "noergler"),
            mention_prompt_template=_env("REVIEW_MENTION_PROMPT_TEMPLATE", "prompts/mention.txt"),
            optimize_diff_tokens=_env("REVIEW_OPTIMIZE_DIFF_TOKENS", "true").lower() in ("true", "1", "yes"),
        ),
        server=ServerConfig(
            host=_env("SERVER_HOST", "0.0.0.0"),
            port=int(_env("SERVER_PORT", "8080")),
        ),
    )
