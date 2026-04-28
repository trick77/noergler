"""Structural tests for alembic migrations. No live DB — files parsed as text."""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_VERSIONS = Path("alembic/versions")
_M002 = _VERSIONS / "002_metrics_and_lifecycle.py"
_M003 = _VERSIONS / "003_pr_cost.py"


def test_migration_chain_is_linear_and_reaches_003():
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    revs = [(r.revision, r.down_revision) for r in script.walk_revisions()]
    chain = [r for r, _ in revs]
    assert chain == ["003", "002", "001"]
    assert dict(revs) == {"003": "002", "002": "001", "001": None}


def test_003_adds_cost_columns_with_numeric_type():
    text = _M003.read_text()
    assert "ADD COLUMN total_cost_usd NUMERIC(10,6)" in text
    assert "ADD COLUMN final_cost_usd NUMERIC(10,6)" in text
    # downgrade reverses both
    assert "DROP COLUMN IF EXISTS total_cost_usd" in text
    assert "DROP COLUMN IF EXISTS final_cost_usd" in text


def test_002_creates_four_metric_views():
    text = _M002.read_text()
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"CREATE VIEW {view}" in text, f"missing CREATE VIEW {view}"


def test_002_downgrade_drops_every_view_it_creates():
    text = _M002.read_text()
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"DROP VIEW IF EXISTS {view}" in text, f"missing DROP for {view}"


def test_002_excludes_deleted_prs_from_all_non_lead_time_views():
    text = _M002.read_text()
    # reviewer_precision and activity exclude deleted PRs via `pr.deleted_at IS NULL`.
    # Cost-by-model reads only review_statistics (no pr_reviews join) by design.
    assert text.count("pr.deleted_at IS NULL") >= 3


def test_002_lead_time_requires_merged_and_opened_at():
    text = _M002.read_text()
    assert "merged_at IS NOT NULL" in text
    assert "opened_at IS NOT NULL" in text


def test_002_precision_buckets_disagrees_by_finding_posted_week():
    """Both CTEs must bucket by review_findings.created_at — otherwise a
    finding posted in week N and disagreed in week N+1 splits across rows
    and a quiet week shows precision_score < 0."""
    text = _M002.read_text()
    assert "rf.bitbucket_comment_id = fe.bitbucket_comment_id" in text
    assert text.count("DATE_TRUNC('week', rf.created_at)") >= 4
    assert "DATE_TRUNC('week', fe.created_at)" not in text


def test_002_precision_score_cast_to_float8():
    """asyncpg returns NUMERIC as Decimal; cast in SQL avoids per-FastAPI-version drift."""
    text = _M002.read_text()
    assert "::float8" in text


def test_002_creates_lead_time_supporting_index():
    text = _M002.read_text()
    assert "CREATE INDEX idx_pr_reviews_lead_time" in text
    assert "DROP INDEX IF EXISTS idx_pr_reviews_lead_time" in text


def test_002_adds_lifecycle_columns():
    text = _M002.read_text()
    for col in ("opened_at", "merged_at", "deleted_at"):
        assert f"ADD COLUMN {col} TIMESTAMPTZ" in text


def test_002_renames_warning_count_column():
    text = _M002.read_text()
    assert "RENAME COLUMN warning_count TO important_count" in text
    # downgrade reverses it
    assert "RENAME COLUMN important_count TO warning_count" in text


def test_002_backfills_warning_severity_to_important():
    """Historical review_findings rows must be normalized to the new severity name."""
    text = _M002.read_text()
    assert "UPDATE review_findings SET severity = 'important' WHERE severity = 'warning'" in text
    # downgrade restores them
    assert "UPDATE review_findings SET severity = 'warning' WHERE severity = 'important'" in text
