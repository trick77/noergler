package store

import (
	"context"
	"os"
	"strings"
	"testing"
)

// A run's tasks are its own figure, never among its findings (user,
// 2026-11-17): every page that reads the run gets both.
func TestRunTasksPostedReadBack(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	runID, err := s.InsertRun(ctx, Run{PullRequestID: prID, Kind: RunAuto, ToCommit: "a", ModelLabel: "m",
		FindingsPosted: 3, TasksPosted: 2})
	if err != nil {
		t.Fatal(err)
	}
	addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", RunID: &runID})

	pg, ok, err := s.PRPage(ctx, prToken(t, s, prID))
	if err != nil || !ok || len(pg.Runs) != 1 || pg.Runs[0].Run == nil {
		t.Fatalf("PRPage = %+v, %v, %v", pg, ok, err)
	}
	if r := pg.Runs[0].Run; r.FindingsPosted != 3 || r.TasksPosted != 2 {
		t.Errorf("PR page run = %d findings, %d tasks", r.FindingsPosted, r.TasksPosted)
	}
	d, ok, err := s.PRRun(ctx, prToken(t, s, prID), 1)
	if err != nil || !ok || d.Run == nil || d.Run.FindingsPosted != 3 || d.Run.TasksPosted != 2 {
		t.Errorf("run page = %+v, %v, %v", d.Run, ok, err)
	}
	rows, err := s.RecentAttempts(ctx, "", "", 10)
	if err != nil || len(rows) != 1 || rows[0].Findings == nil || *rows[0].Findings != 3 ||
		rows[0].Tasks == nil || *rows[0].Tasks != 2 {
		t.Errorf("feed = %+v, %v", rows, err)
	}
	st, err := s.PRStatus(ctx, key, "platform")
	if err != nil || st.LastRun == nil || st.LastRun.Findings != 3 || st.LastRun.Tasks != 2 {
		t.Errorf("status = %+v, %v", st.LastRun, err)
	}
}

// 0037's backfill moves a run's tasks out of findings_posted, by its
// findings rows: an open task and one whose rule left AGENTS.md since; a
// task Bitbucket refused went up as a comment and stays a finding.
func TestTasksPostedBackfill(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	runID, err := s.InsertRun(ctx, Run{PullRequestID: prID, Kind: RunAuto, ToCommit: "a", ModelLabel: "m", FindingsPosted: 4})
	if err != nil {
		t.Fatal(err)
	}
	other, err := s.InsertRun(ctx, Run{PullRequestID: prID, Kind: RunAuto, ToCommit: "b", ModelLabel: "m", FindingsPosted: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i, outcome := range []*string{str("task"), str("rule_removed"), str("downgraded_task_failed"), nil} {
		if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "a.go", LineNumber: i + 1,
			Severity: "issue", CommentText: "c", BitbucketCommentID: num(700 + i), BlockingOutcome: outcome}); err != nil {
			t.Fatal(err)
		}
	}

	raw, err := os.ReadFile("migrations/0037_runs_tasks_posted.sql")
	if err != nil {
		t.Fatal(err)
	}
	sql := string(raw)
	if _, err := s.pool.Exec(ctx, sql[strings.Index(sql, "UPDATE review_runs"):]); err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		id                  int64
		findings, wantTasks int
	}{{runID, 2, 2}, {other, 1, 0}} {
		var f, tk int
		if err := s.pool.QueryRow(ctx, `SELECT findings_posted, tasks_posted FROM review_runs WHERE id = $1`, c.id).
			Scan(&f, &tk); err != nil {
			t.Fatal(err)
		}
		if f != c.findings || tk != c.wantTasks {
			t.Errorf("run %d = %d findings, %d tasks, want %d and %d", c.id, f, tk, c.findings, c.wantTasks)
		}
	}
}
