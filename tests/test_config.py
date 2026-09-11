import logging
from pathlib import Path

import pytest

from app.config import (
    AppConfig,
    BitbucketConfig,
    DatabaseConfig,
    JiraConfig,
    LLMConfig,
    ProjectScope,
    ReviewConfig,
    RiptideConfig,
    ServerConfig,
    TeamConfig,
    TeamsFileError,
    load_config,
    load_instance_config,
    load_teams,
    log_config,
    team_env_prefix,
)

CATALOG_URL = "https://catalog.test/model_prices_and_context_window.json"

MINIMAL_TEAMS = """
teams:
  - slug: platform
    webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET
    projects:
      - key: PLAT
    inference:
      api_key_env: TEAM_PLATFORM_OPENAI_API_KEY
"""


def _instance_env(teams_path: str) -> dict[str, str]:
    return {
        "BITBUCKET_URL": "https://bb.example.com",
        "BITBUCKET_TOKEN": "tok",
        "BITBUCKET_USERNAME": "bot",
        "OPENAI_BASE_URL": "https://llm.example.com/v1",
        "MODEL_CATALOG_URL": CATALOG_URL,
        "JIRA_URL": "https://jira.example.com",
        "JIRA_TOKEN": "jira-tok",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "TEAMS_CONFIG": teams_path,
        "TEAM_PLATFORM_WEBHOOK_SECRET": "plat-secret",
        "TEAM_PLATFORM_OPENAI_API_KEY": "plat-key",
    }


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    """Instance env + a one-team file. Returns a setter for extra vars and
    the path of the teams file so a test can overwrite it."""
    teams_file = tmp_path / "teams.yaml"
    teams_file.write_text(MINIMAL_TEAMS)
    for k, v in _instance_env(str(teams_file)).items():
        monkeypatch.setenv(k, v)

    class _Env:
        path = teams_file

        @staticmethod
        def set(**kwargs: str) -> None:
            for k, v in kwargs.items():
                monkeypatch.setenv(k, v)

        @staticmethod
        def unset(*names: str) -> None:
            for n in names:
                monkeypatch.delenv(n, raising=False)

        @staticmethod
        def teams(text: str) -> None:
            teams_file.write_text(text)

    return _Env()


def _make_config():
    return AppConfig(
        bitbucket=BitbucketConfig(
            base_url="https://bitbucket.example.com",
            token="secret-bb-token",
            username="bot-user",
        ),
        llm=LLMConfig(
            model="gpt-5.3-codex",
            api_key="",
            api_url="https://llm.example.com/v1",
            catalog_url=CATALOG_URL,
        ),
        review=ReviewConfig(
            auto_review_authors=["alice", "bob"],
            max_comments=10,
            review_prompt_template="prompts/review.txt",
        ),
        server=ServerConfig(host="0.0.0.0", port=9090),
        jira=JiraConfig(url="https://jira.example.com", token="secret-jira-token"),
        database=DatabaseConfig(url="postgresql://noergler:secret@localhost/noergler"),
        teams_config_path="teams.yaml",
        teams={
            "platform": TeamConfig(
                slug="platform",
                name="Platform",
                webhook_secret="secret-webhook",
                projects=[ProjectScope(key="PLAT")],
                llm=LLMConfig(
                    model="gpt-5.3-codex", api_key="secret-api-key",
                    api_url="https://llm.example.com/v1", catalog_url=CATALOG_URL,
                ),
                review=ReviewConfig(),
                jira=JiraConfig(url="https://jira.example.com", token="secret-jira-token"),
                riptide=RiptideConfig(url="https://riptide.example.com", token="secret-riptide-token"),
            ),
        },
        disabled={"payments": "TEAM_PAYMENTS_OPENAI_API_KEY is not set"},
    )


def test_log_config_masks_secrets(caplog):
    config = _make_config()
    with caplog.at_level(logging.INFO):
        log_config(config, logging.getLogger("test_config"))

    text = caplog.text

    for secret in (
        "secret-bb-token", "secret-webhook", "secret-api-key",
        "secret-jira-token", "secret-riptide-token",
    ):
        assert secret not in text, secret
    # instance: bb token, (empty) llm api key, jira token, database url;
    # team: webhook secret, llm api key, jira token, riptide token
    assert text.count("***") == 8

    assert "https://bitbucket.example.com" in text
    assert "gpt-5.3-codex" in text
    assert "['alice', 'bob']" in text
    assert "9090" in text

    for header in (
        "[config.bitbucket]", "[config.llm]", "[config.review]", "[config.jira]",
        "[config.server]", "[config.database]", "[config.teams.platform]",
        "[config.teams.platform.llm]", "[config.teams.platform.riptide]",
    ):
        assert header in text, header
    assert "['PLAT']" in text
    assert "[config.teams.payments] DISABLED: TEAM_PAYMENTS_OPENAI_API_KEY is not set" in text


def test_log_config_marks_riptide_off_for_a_team_without_it(caplog):
    config = _make_config()
    config.teams["platform"].riptide = None
    with caplog.at_level(logging.INFO):
        log_config(config, logging.getLogger("test_config"))
    assert "[config.teams.platform.riptide] disabled" in caplog.text


# --- Instance env -----------------------------------------------------------


def test_diff_context_from_env(env):
    env.set(
        REVIEW_DIFF_EXTRA_LINES_BEFORE="5",
        REVIEW_DIFF_EXTRA_LINES_AFTER="2",
        REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT="12",
        REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT="false",
    )
    config = load_config()
    assert config.review.diff_extra_lines_before == 5
    assert config.review.diff_extra_lines_after == 2
    assert config.review.diff_max_extra_lines_dynamic_context == 12
    assert config.review.diff_allow_dynamic_context is False
    # instance review defaults flow into the team
    assert config.teams["platform"].review.diff_extra_lines_before == 5


def test_ticket_compliance_check_from_env(env):
    env.set(REVIEW_TICKET_COMPLIANCE_CHECK="false")
    assert load_config().review.ticket_compliance_check is False


def test_ticket_compliance_check_default_from_env(env):
    assert load_config().review.ticket_compliance_check is True


def test_opt_out_branch_keyword_default():
    assert ReviewConfig().opt_out_branch_keyword == "noergloff"


def test_catalog_url_missing_is_fatal(env):
    # MODEL_CATALOG_URL is required with no default: intg and prod point at
    # different catalogs, so defaulting to either would silently price and size
    # a deployment against the wrong one. Startup must fail loudly instead.
    env.unset("MODEL_CATALOG_URL")
    with pytest.raises(ValueError, match="MODEL_CATALOG_URL"):
        load_config()


def test_instance_has_no_api_key_and_no_webhook_secret(env):
    # Both moved to the team block; the old env vars are ignored.
    env.set(OPENAI_API_KEY="stale", BITBUCKET_WEBHOOK_SECRET="stale")
    config = load_config()
    assert config.llm.api_key == ""
    assert not hasattr(config.bitbucket, "webhook_secret")
    assert config.teams["platform"].llm.api_key == "plat-key"
    assert config.teams["platform"].webhook_secret == "plat-secret"


def test_reasoning_effort_default_high(env):
    env.unset("OPENAI_REASONING_EFFORT")
    config = load_config()
    assert config.llm.reasoning_effort == "high"
    assert config.teams["platform"].llm.reasoning_effort == "high"


def test_reasoning_effort_valid_from_env(env):
    env.set(OPENAI_REASONING_EFFORT="HIGH")
    assert load_config().llm.reasoning_effort == "high"


def test_reasoning_effort_empty_string_rejected(env):
    # noergler requires a reasoning-capable model: an explicitly empty value is
    # rejected rather than silently defaulting.
    env.set(OPENAI_REASONING_EFFORT="")
    with pytest.raises(Exception):
        load_config()


def test_reasoning_effort_invalid_rejected(env):
    env.set(OPENAI_REASONING_EFFORT="extreme")
    with pytest.raises(Exception):
        load_config()


def test_opt_out_branch_keyword_from_env(env):
    env.set(REVIEW_OPT_OUT_BRANCH_KEYWORD="skipme")
    assert load_config().review.opt_out_branch_keyword == "skipme"


def test_teams_config_path_defaults_to_teams_yaml(env):
    env.unset("TEAMS_CONFIG")
    assert load_instance_config().teams_config_path == "teams.yaml"


# --- teams.yaml: file-level faults abort ------------------------------------


def _load(env) -> AppConfig:
    return load_config()


def test_missing_teams_file_aborts(env):
    env.set(TEAMS_CONFIG=str(env.path.parent / "nope.yaml"))
    with pytest.raises(TeamsFileError, match="not found"):
        _load(env)


def test_unparseable_teams_file_aborts(env):
    env.teams("teams: [unclosed")
    with pytest.raises(TeamsFileError, match="not valid YAML"):
        _load(env)


def test_teams_file_without_teams_key_aborts(env):
    env.teams("foo: bar")
    with pytest.raises(TeamsFileError, match="top-level `teams:`"):
        _load(env)


def test_zero_teams_aborts(env):
    env.teams("teams: []")
    with pytest.raises(TeamsFileError, match="non-empty"):
        _load(env)


def test_duplicate_slug_aborts(env):
    env.teams(MINIMAL_TEAMS + MINIMAL_TEAMS.replace("teams:\n", ""))
    with pytest.raises(TeamsFileError, match="duplicate slug"):
        _load(env)


# --- teams.yaml: team-level faults disable that team only -------------------

TWO_TEAMS = MINIMAL_TEAMS + """
  - slug: payments
    name: "Payments"
    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET
    projects:
      - key: PAY
        repos: [billing, ledger]
    inference:
      api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY
      model: gpt-5.5
      reasoning_effort: medium
      context_window: 1200000
    review:
      auto_review_authors: [alice]
      max_pr_cost_usd: 8.5
    jira:
      acceptance_criteria_prefixes: [AC, DoD]
    riptide:
      url: https://riptide-payments.example.com/
      token_env: TEAM_PAYMENTS_RIPTIDE_TOKEN
"""


def _payments_env(env) -> None:
    env.set(
        TEAM_PAYMENTS_WEBHOOK_SECRET="pay-secret",
        TEAM_PAYMENTS_OPENAI_API_KEY="pay-key",
        TEAM_PAYMENTS_RIPTIDE_TOKEN="pay-riptide",
    )


def test_full_team_block_resolves_with_overrides_on_top_of_instance_defaults(env):
    env.teams(TWO_TEAMS)
    _payments_env(env)
    env.set(OPENAI_MODEL="gpt-5.4", REVIEW_MAX_COMMENTS="7")
    config = _load(env)
    assert config.disabled == {}
    assert sorted(config.teams) == ["payments", "platform"]

    plat = config.teams["platform"]
    assert plat.name == "platform"  # defaults to the slug
    assert plat.llm.model == "gpt-5.4"
    assert plat.llm.api_key == "plat-key"
    assert plat.llm.api_url == "https://llm.example.com/v1"
    assert plat.llm.catalog_url == CATALOG_URL
    assert plat.review.max_comments == 7
    assert plat.review.auto_review_authors == []
    assert plat.jira.acceptance_criteria_prefixes == JiraConfig(url="", token="").acceptance_criteria_prefixes
    assert plat.riptide is None

    pay = config.teams["payments"]
    assert pay.name == "Payments"
    assert pay.webhook_secret == "pay-secret"
    assert pay.projects == [ProjectScope(key="PAY", repos=["billing", "ledger"])]
    assert pay.llm.model == "gpt-5.5"
    assert pay.llm.reasoning_effort == "medium"
    assert pay.llm.context_window == 1_200_000
    assert pay.llm.api_key == "pay-key"
    # instance-only values are inherited, never overridden
    assert pay.llm.api_url == "https://llm.example.com/v1"
    assert pay.llm.catalog_url == CATALOG_URL
    assert pay.review.auto_review_authors == ["alice"]
    assert pay.review.max_pr_cost_usd == 8.5
    assert pay.review.max_comments == 7
    assert pay.jira.url == "https://jira.example.com"
    assert pay.jira.token == "jira-tok"
    assert pay.jira.acceptance_criteria_prefixes == ["AC", "DoD"]
    assert pay.riptide == RiptideConfig(url="https://riptide-payments.example.com", token="pay-riptide")


def test_team_for_resolves_ownership_by_project_and_repo(env):
    env.teams(TWO_TEAMS)
    _payments_env(env)
    config = _load(env)
    assert config.team_for("PLAT", "anything") is config.teams["platform"]
    assert config.team_for("PAY", "billing") is config.teams["payments"]
    assert config.team_for("PAY", "other") is None
    assert config.team_for("NOPE", "x") is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        # secrets
        (lambda t: t.replace("TEAM_PAYMENTS_OPENAI_API_KEY", "TEAM_PAYMENTS_MISSING"),
         "inference.api_key_env: environment variable TEAM_PAYMENTS_MISSING is not set"),
        (lambda t: t.replace("TEAM_PAYMENTS_WEBHOOK_SECRET", "TEAM_PAYMENTS_MISSING"),
         "webhook_secret_env: environment variable TEAM_PAYMENTS_MISSING is not set"),
        (lambda t: t.replace("TEAM_PAYMENTS_RIPTIDE_TOKEN", "TEAM_PAYMENTS_MISSING"),
         "riptide.token_env: environment variable TEAM_PAYMENTS_MISSING is not set"),
        # required fields
        (lambda t: t.replace("    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET\n", ""),
         "webhook_secret_env: Field required"),
        (lambda t: t.replace("      api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY\n", ""),
         "inference.api_key_env: Field required"),
        (lambda t: t.replace("      - key: PAY\n        repos: [billing, ledger]\n", "      []\n"),
         "projects: Value error, projects must list at least one"),
        (lambda t: t.replace("repos: [billing, ledger]", "repos: []"),
         "projects.0.repos: Value error, repos must list at least one slug"),
        # instance-only knobs in a team block
        (lambda t: t.replace("      model: gpt-5.5\n", "      model: gpt-5.5\n      base_url: https://x\n"),
         "inference.base_url: Extra inputs are not permitted"),
        (lambda t: t.replace("      model: gpt-5.5\n", "      model: gpt-5.5\n      catalog_url: https://x\n"),
         "inference.catalog_url: Extra inputs are not permitted"),
        (lambda t: t.replace("      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      review_prompt_template: x\n"),
         "review.review_prompt_template: Extra inputs are not permitted"),
        (lambda t: t.replace("      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      mention_prompt_template: x\n"),
         "review.mention_prompt_template: Extra inputs are not permitted"),
        (lambda t: t.replace("      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      max_comment: 3\n"),
         "review.max_comment: Extra inputs are not permitted"),
        # value validation reuses the instance validators
        (lambda t: t.replace("reasoning_effort: medium", "reasoning_effort: extreme"),
         "inference: reasoning_effort: Value error, reasoning_effort must be one of"),
        # riptide is both-or-neither
        (lambda t: t.replace("      token_env: TEAM_PAYMENTS_RIPTIDE_TOKEN\n", ""),
         "riptide.token_env: Field required"),
        (lambda t: t.replace("      url: https://riptide-payments.example.com/\n", ""),
         "riptide.url: Field required"),
        (lambda t: t.replace("url: https://riptide-payments.example.com/", "url: ''"),
         "riptide.url must be non-empty"),
        # slug
        (lambda t: t.replace("slug: payments", "slug: Payments"),
         "slug: Value error, slug 'Payments' must match"),
    ],
)
def test_team_level_fault_disables_only_that_team(env, caplog, mutation, reason):
    env.teams(mutation(TWO_TEAMS))
    _payments_env(env)
    with caplog.at_level(logging.ERROR):
        config = _load(env)
    assert sorted(config.teams) == ["platform"]
    (slug, got), = config.disabled.items()
    assert slug.lower() == "payments"  # keyed by the raw slug even when invalid
    assert got.startswith(reason), got
    assert f"team_disabled team={slug} reason=" in caplog.text


def test_empty_secret_value_disables_the_team(env):
    env.teams(TWO_TEAMS)
    _payments_env(env)
    env.set(TEAM_PAYMENTS_OPENAI_API_KEY="   ")
    config = _load(env)
    assert config.disabled == {
        "payments": "inference.api_key_env: environment variable TEAM_PAYMENTS_OPENAI_API_KEY is empty",
    }


def test_shared_project_disables_every_claimant(env):
    env.teams(TWO_TEAMS.replace("key: PAY", "key: PLAT"))
    _payments_env(env)
    config = _load(env)
    assert config.teams == {}
    assert config.disabled == {
        "platform": "project PLAT is also claimed by team payments",
        "payments": "project PLAT is also claimed by team platform",
    }


def test_disjoint_repo_lists_on_one_project_do_not_conflict(env):
    text = TWO_TEAMS.replace("      - key: PLAT\n", "      - key: PAY\n        repos: [core]\n")
    env.teams(text)
    _payments_env(env)
    config = _load(env)
    assert config.disabled == {}
    assert config.team_for("PAY", "core") is config.teams["platform"]
    assert config.team_for("PAY", "billing") is config.teams["payments"]


def test_a_disabled_team_does_not_take_part_in_ownership_conflicts(env):
    # payments is disabled for a missing key; its PLAT claim must not drag
    # platform down with it.
    text = TWO_TEAMS.replace("key: PAY", "key: PLAT")
    env.teams(text)
    env.set(TEAM_PAYMENTS_WEBHOOK_SECRET="pay-secret", TEAM_PAYMENTS_RIPTIDE_TOKEN="x")
    config = _load(env)
    assert list(config.teams) == ["platform"]
    assert list(config.disabled) == ["payments"]


def test_load_teams_is_usable_standalone(env):
    instance = load_instance_config()
    enabled, disabled = load_teams(instance.teams_config_path, instance)
    assert list(enabled) == ["platform"]
    assert disabled == {}


def test_team_env_prefix_uppercases_and_replaces_dashes():
    assert team_env_prefix("data-platform") == "TEAM_DATA_PLATFORM_"
