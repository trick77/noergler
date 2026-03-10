import os

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


class BitbucketConfig(BaseModel):
    base_url: str
    token: str
    webhook_secret: str


class CopilotConfig(BaseModel):
    model: str = "openai/gpt-4.1"
    github_token: str
    api_url: str = "https://models.github.ai/inference/chat/completions"
    max_tokens_per_chunk: int = 80000


class ReviewConfig(BaseModel):
    allowed_authors: list[str] = []
    max_comments: int = 25
    max_lines_per_file: int = 1000
    review_prompt_template: str = "prompts/review.txt"
    review_tone: str = "default"

    @field_validator("allowed_authors", mode="before")
    @classmethod
    def parse_allowed_authors(cls, v):
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


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url=_env("BITBUCKET_URL"),
            token=_env("BITBUCKET_TOKEN"),
            webhook_secret=_env("BITBUCKET_WEBHOOK_SECRET"),
        ),
        copilot=CopilotConfig(
            model=_env("COPILOT_MODEL", "openai/gpt-4.1"),
            github_token=_env("GITHUB_TOKEN"),
            api_url=_env("COPILOT_API_URL", "https://models.github.ai/inference/chat/completions"),
            max_tokens_per_chunk=int(_env("COPILOT_MAX_TOKENS", "80000")),
        ),
        review=ReviewConfig(
            allowed_authors=_env("REVIEW_ALLOWED_AUTHORS", ""),
            max_comments=int(_env("REVIEW_MAX_COMMENTS", "25")),
            max_lines_per_file=int(_env("REVIEW_MAX_LINES_PER_FILE", "1000")),
            review_prompt_template=_env("REVIEW_PROMPT_TEMPLATE", "prompts/review.txt"),
            review_tone=_env("REVIEW_TONE", "default"),
        ),
        server=ServerConfig(
            host=_env("SERVER_HOST", "0.0.0.0"),
            port=int(_env("SERVER_PORT", "8080")),
        ),
    )
