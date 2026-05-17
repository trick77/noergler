"""Aggregate per-PR review stats for the riptide pr_completed rollup

Revision ID: 007
Revises: 006
Create Date: 2026-05-17

Riptide moved from a per-run 'completed' event to a per-PR 'pr_completed'
rollup emitted once at PR close (merged / declined / deleted). This migration
adds the aggregate columns on pr_reviews that noergler accumulates over the
PR's lifetime and reads out when emitting the rollup.

- total_prompt_tokens / total_completion_tokens / total_elapsed_ms /
  total_findings_count / total_runs — running sums across review runs.
- models_used — distinct model labels seen across runs.
- first_review_at — timestamp of the first review run (lower bound on
  "AI review touched this PR").
- declined_at — sibling to merged_at / deleted_at; previously not tracked.
- final_source_commit_sha — refreshed on every review run (latest reviewed
  HEAD). At close time it therefore reflects the last source-branch commit
  noergler actually saw, which is the right key for joining to bitbucket
  events.
- final_merge_commit_sha — only set at close, when the pr:merged payload
  carries a merge commit; NULL for declined / deleted outcomes.
- final_lines_added / final_lines_removed / final_files_changed — best-effort
  refreshed from the cumulative PR diff at close so the rollup carries an
  unambiguous diff size (not the last incremental review's diff). If the
  diff is unavailable (e.g. PR already gone for deleted), the per-run
  trail values are kept.
- riptide_emitted_at — idempotency marker; the emit path skips if set.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_prompt_tokens BIGINT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_completion_tokens BIGINT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_elapsed_ms BIGINT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_findings_count INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_runs INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN models_used TEXT[] NOT NULL DEFAULT '{}'")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN first_review_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN declined_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_source_commit_sha TEXT")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_merge_commit_sha TEXT")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_lines_added INTEGER")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_lines_removed INTEGER")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_files_changed INTEGER")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN riptide_emitted_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS riptide_emitted_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_files_changed")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_lines_removed")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_lines_added")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_merge_commit_sha")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_source_commit_sha")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS declined_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS first_review_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS models_used")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_runs")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_findings_count")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_elapsed_ms")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_completion_tokens")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_prompt_tokens")
