"""Structural tests for alembic migrations. No live DB — files parsed as text."""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_VERSIONS = Path("alembic/versions")
_M003 = _VERSIONS / "003_pr_cost.py"
_M004 = _VERSIONS / "004_rename_severity.py"
_M005 = _VERSIONS / "005_drop_metrics_layer.py"


def test_migration_chain_is_linear_and_reaches_005():
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    revs = [(r.revision, r.down_revision) for r in script.walk_revisions()]
    chain = [r for r, _ in revs]
    assert chain == ["005", "004", "003", "002", "001"]
    assert dict(revs) == {
        "005": "004",
        "004": "003",
        "003": "002",
        "002": "001",
        "001": None,
    }


def test_003_adds_cost_columns_with_numeric_type():
    text = _M003.read_text()
    assert "ADD COLUMN total_cost_usd NUMERIC(10,6)" in text
    assert "ADD COLUMN final_cost_usd NUMERIC(10,6)" in text
    # downgrade reverses both
    assert "DROP COLUMN IF EXISTS total_cost_usd" in text
    assert "DROP COLUMN IF EXISTS final_cost_usd" in text


def test_004_renames_severity_columns():
    text = _M004.read_text()
    assert "RENAME COLUMN critical_count TO issue_count" in text
    assert "RENAME COLUMN important_count TO suggestion_count" in text
    # downgrade reverses both
    assert "RENAME COLUMN suggestion_count TO important_count" in text
    assert "RENAME COLUMN issue_count TO critical_count" in text


def test_004_backfills_severity_values():
    """Existing 'critical'/'important' rows must be normalized to the new vocabulary."""
    text = _M004.read_text()
    assert "UPDATE review_findings SET severity = 'issue' WHERE severity = 'critical'" in text
    assert "UPDATE review_findings SET severity = 'suggestion' WHERE severity = 'important'" in text
    assert "UPDATE feedback_events SET severity = 'issue' WHERE severity = 'critical'" in text
    assert "UPDATE feedback_events SET severity = 'suggestion' WHERE severity = 'important'" in text
    # downgrade restores all four
    assert "UPDATE review_findings SET severity = 'critical' WHERE severity = 'issue'" in text
    assert "UPDATE review_findings SET severity = 'important' WHERE severity = 'suggestion'" in text
    assert "UPDATE feedback_events SET severity = 'critical' WHERE severity = 'issue'" in text
    assert "UPDATE feedback_events SET severity = 'important' WHERE severity = 'suggestion'" in text


def test_005_drops_every_metric_view():
    text = _M005.read_text()
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"DROP VIEW IF EXISTS {view}" in text, f"missing DROP for {view}"


def test_005_drops_review_statistics_table_and_lead_time_index():
    text = _M005.read_text()
    assert "DROP TABLE IF EXISTS review_statistics" in text
    assert "DROP INDEX IF EXISTS idx_pr_reviews_lead_time" in text


def test_005_downgrade_recreates_views_and_table():
    """Downgrade restores the metrics layer so cluster operators can roll back."""
    text = _M005.read_text()
    assert "CREATE TABLE review_statistics" in text
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"CREATE VIEW {view}" in text, f"downgrade missing CREATE VIEW {view}"
