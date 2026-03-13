import logging

from app.config import AppConfig, BitbucketConfig, CopilotConfig, ReviewConfig, ServerConfig, load_config, log_config


def _make_config():
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url="https://bitbucket.example.com",
            token="secret-bb-token",
            webhook_secret="secret-webhook",
        ),
        copilot=CopilotConfig(
            model="openai/gpt-4",
            github_token="ghp_secret123",
            api_url="https://models.github.ai/inference/chat/completions",
            max_tokens_per_chunk=80000,
        ),
        review=ReviewConfig(
            auto_review_authors=["alice", "bob"],
            max_comments=10,
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
    assert "openai/gpt-4" in text
    assert "80000" in text
    assert "['alice', 'bob']" in text
    assert "10" in text
    assert "alice" in text
    assert "9090" in text

    # Section headers must appear
    assert "[config.bitbucket]" in text
    assert "[config.copilot]" in text
    assert "[config.review]" in text
    assert "[config.server]" in text


def test_diff_context_defaults():
    rc = ReviewConfig()
    assert rc.diff_extra_lines_before == 3
    assert rc.diff_extra_lines_after == 1
    assert rc.diff_max_extra_lines_dynamic_context == 8
    assert rc.diff_allow_dynamic_context is True


def test_diff_context_from_env(monkeypatch):
    env = {
        "BITBUCKET_URL": "https://bb.example.com",
        "BITBUCKET_TOKEN": "tok",
        "BITBUCKET_WEBHOOK_SECRET": "sec",
        "GITHUB_TOKEN": "ghp_tok",
        "REVIEW_DIFF_EXTRA_LINES_BEFORE": "5",
        "REVIEW_DIFF_EXTRA_LINES_AFTER": "2",
        "REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT": "12",
        "REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT": "false",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config = load_config()
    assert config.review.diff_extra_lines_before == 5
    assert config.review.diff_extra_lines_after == 2
    assert config.review.diff_max_extra_lines_dynamic_context == 12
    assert config.review.diff_allow_dynamic_context is False
