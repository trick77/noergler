"""Structural tests for alembic migrations. No live DB — files parsed as text."""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_VERSIONS = Path("alembic/versions")


def test_migration_chain_is_linear_and_reaches_004():
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    revs = [(r.revision, r.down_revision) for r in script.walk_revisions()]
    # Newest first from walk_revisions
    chain = [r for r, _ in revs]
    assert chain == ["004", "003", "002", "001"]
    # Down-chain links match
    assert dict(revs) == {"004": "003", "003": "002", "002": "001", "001": None}


def test_004_creates_four_metric_views():
    text = (_VERSIONS / "004_metric_views.py").read_text()
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"CREATE VIEW {view}" in text, f"missing CREATE VIEW {view}"


def test_004_downgrade_drops_every_view_it_creates():
    text = (_VERSIONS / "004_metric_views.py").read_text()
    for view in (
        "v_reviewer_precision",
        "v_activity_weekly",
        "v_cost_by_model",
        "v_lead_time",
    ):
        assert f"DROP VIEW IF EXISTS {view}" in text, f"missing DROP for {view}"


def test_004_excludes_deleted_prs_from_all_non_lead_time_views():
    text = (_VERSIONS / "004_metric_views.py").read_text()
    # reviewer_precision, activity, and cost-indirect all exclude deleted PRs
    # via the `pr.deleted_at IS NULL` filter. Cost-by-model reads only
    # review_statistics (no pr_reviews join), so it's excluded from this check.
    assert text.count("pr.deleted_at IS NULL") >= 3


def test_004_lead_time_requires_merged_and_opened_at():
    text = (_VERSIONS / "004_metric_views.py").read_text()
    # v_lead_time filters rows that have both timestamps
    assert "merged_at IS NOT NULL" in text
    assert "opened_at IS NOT NULL" in text


def test_003_adds_lifecycle_columns():
    text = (_VERSIONS / "003_pr_lifecycle_timestamps.py").read_text()
    for col in ("opened_at", "merged_at", "deleted_at"):
        assert f"ADD COLUMN {col} TIMESTAMPTZ" in text
