import logging

from app.config import AppConfig, BitbucketConfig, CopilotConfig, ReviewConfig, ServerConfig, log_config


def _make_config():
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url="https://bitbucket.example.com",
            token="secret-bb-token",
            webhook_secret="secret-webhook",
        ),
        copilot=CopilotConfig(
            model="openai/gpt-5.2",
            github_token="ghp_secret123",
            api_url="https://models.github.ai/inference/chat/completions",
            max_tokens_per_chunk=80000,
        ),
        review=ReviewConfig(
            auto_review_authors=["alice", "bob"],
            max_comments=10,
            max_lines_per_file=500,
            review_prompt_template="prompts/review.txt",
            ramsay_authors=["alice"],
        ),
        server=ServerConfig(host="0.0.0.0", port=9090),
    )


def test_log_config_masks_secrets(caplog):
    config = _make_config()
    with caplog.at_level(logging.INFO):
        log_config(config, logging.getLogger("test_config"))

    text = caplog.text

    # Secret fields must be masked
    assert "secret-bb-token" not in text
    assert "secret-webhook" not in text
    assert "ghp_secret123" not in text
    assert text.count("***") == 3

    # Non-secret fields must appear as-is
    assert "https://bitbucket.example.com" in text
    assert "openai/gpt-5.2" in text
    assert "80000" in text
    assert "['alice', 'bob']" in text
    assert "10" in text
    assert "500" in text
    assert "alice" in text
    assert "9090" in text

    # Section headers must appear
    assert "[config.bitbucket]" in text
    assert "[config.copilot]" in text
    assert "[config.review]" in text
    assert "[config.server]" in text
