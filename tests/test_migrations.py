"""Structural tests for alembic migrations. No live DB — files parsed as text."""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_VERSIONS = Path("alembic/versions")
_M001 = _VERSIONS / "001_initial_schema.py"


def test_migration_chain_is_a_single_squashed_revision():
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    revs = [(r.revision, r.down_revision) for r in script.walk_revisions()]
    assert revs == [("001", None)]


def test_001_creates_the_two_tables_and_drops_them_in_downgrade():
    text = _M001.read_text()
    assert "CREATE TABLE pr_reviews" in text
    assert "CREATE TABLE review_findings" in text
    assert "DROP TABLE IF EXISTS review_findings" in text
    assert "DROP TABLE IF EXISTS pr_reviews" in text
    # The metrics layer, feedback events and the pricing cache stayed dropped.
    for gone in ("review_statistics", "feedback_events", "model_pricing"):
        assert gone not in text


def test_001_pr_reviews_carries_the_team_column_not_null_and_indexed():
    text = _M001.read_text()
    assert "team_slug TEXT NOT NULL" in text
    assert "CREATE INDEX idx_pr_reviews_team ON pr_reviews (team_slug)" in text


def test_001_pr_reviews_carries_every_column_the_repository_writes():
    text = _M001.read_text()
    for col in (
        # identity + summary
        "project_key", "repo_slug", "pr_id", "last_reviewed_commit",
        "summary_comment_id", "summary_comment_version", "author", "pr_title",
        # lifecycle
        "opened_at", "merged_at", "declined_at", "deleted_at", "ignored_at",
        # cost
        "total_cost_usd NUMERIC(10,6)", "final_cost_usd NUMERIC(10,6)",
        # riptide rollup
        "total_prompt_tokens", "total_completion_tokens", "total_elapsed_ms",
        "total_findings_count", "total_runs", "models_used TEXT[]",
        "first_review_at", "final_source_commit_sha", "final_merge_commit_sha",
        "final_lines_added", "final_lines_removed", "final_files_changed",
        "riptide_emitted_at",
    ):
        assert col in text, col
    assert "UNIQUE (project_key, repo_slug, pr_id)" in text
    assert "CREATE INDEX idx_pr_reviews_lifecycle ON pr_reviews (merged_at, deleted_at)" in text


def test_001_review_findings_keeps_the_dedup_index():
    text = _M001.read_text()
    assert "REFERENCES pr_reviews(id) ON DELETE CASCADE" in text
    assert "ON review_findings (pr_review_id, file_path, line_number, severity)" in text
