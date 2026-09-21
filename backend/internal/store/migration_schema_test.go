package store

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// The schema tests. Migrate() being idempotent and safe under concurrent
// callers was already covered; what the migration actually CREATED was not,
// so an edit to 0001_initial.sql broke no test.
//
// These assert the LIVE schema through information_schema rather than
// matching the migration SQL as text: a text match passes on SQL that never
// ran, and cannot see a column whose type or nullability drifted.
//
// The table names changed with the schema: pr_reviews became pull_requests,
// review_findings became findings, and review_runs is new (schema is runs,
// not accumulators).

// columnSpec is one column the code depends on: its type and whether the
// writer may omit it.
type columnSpec struct {
	name     string
	dataType string
	notNull  bool
}

func columns(t *testing.T, s *Store, table string) map[string]columnSpec {
	t.Helper()
	rows, err := s.pool.Query(context.Background(), `
		SELECT column_name, data_type, is_nullable
		FROM information_schema.columns
		WHERE table_schema = current_schema() AND table_name = $1`, table)
	if err != nil {
		t.Fatalf("read columns of %s: %v", table, err)
	}
	defer rows.Close()
	out := map[string]columnSpec{}
	for rows.Next() {
		var name, dataType, isNullable string
		if err := rows.Scan(&name, &dataType, &isNullable); err != nil {
			t.Fatal(err)
		}
		out[name] = columnSpec{name, dataType, isNullable == "NO"}
	}
	if rows.Err() != nil {
		t.Fatal(rows.Err())
	}
	if len(out) == 0 {
		t.Fatalf("table %s does not exist", table)
	}
	return out
}

// indexColumns is the index's columns in INDEX order, which is what decides
// whether a lookup can use it. Reading indexdef and testing for each name
// instead would accept any permutation.
func indexColumns(t *testing.T, s *Store, name string) []string {
	t.Helper()
	var cols []string
	err := s.pool.QueryRow(context.Background(), `
		SELECT array(
			SELECT a.attname
			FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
			JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
			ORDER BY k.ord
		)
		FROM pg_index i
		JOIN pg_class c ON c.oid = i.indexrelid
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = current_schema() AND c.relname = $1`, name).Scan(&cols)
	if err != nil {
		t.Fatalf("index %s not found: %v", name, err)
	}
	return cols
}

func indexDef(t *testing.T, s *Store, name string) string {
	t.Helper()
	var def string
	err := s.pool.QueryRow(context.Background(),
		`SELECT indexdef FROM pg_indexes
		 WHERE schemaname = current_schema() AND indexname = $1`, name).Scan(&def)
	if err != nil {
		t.Fatalf("index %s not found: %v", name, err)
	}
	return def
}

func TestSchema_TablesExist(t *testing.T) {
	s := testStore(t)
	// columns() fatals on a missing table, which would stop at the first one,
	// so the existence check is its own query: a migration that dropped three
	// tables should say so once.
	var present []string
	err := s.pool.QueryRow(context.Background(), `
		SELECT array(
			SELECT table_name FROM information_schema.tables
			WHERE table_schema = current_schema()
			ORDER BY table_name
		)`).Scan(&present)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{
		"pull_requests", "review_runs", "findings",
		"team_claims", "team_settings", "schema_migrations",
	} {
		if !slices.Contains(present, table) {
			t.Errorf("table %s is missing; schema has %v", table, present)
		}
	}
}

// Every column the repository writes, with the type and nullability the
// writer assumes. A cost column that lost its NULLability would silently
// break "cost fails open"; a nano-USD column narrowed to INTEGER would
// overflow.
func TestSchema_PullRequestsCarriesEveryColumnTheStoreWrites(t *testing.T) {
	s := testStore(t)
	cols := columns(t, s, "pull_requests")
	want := []columnSpec{
		{"id", "bigint", true},
		{"project_key", "text", true},
		{"repo_slug", "text", true},
		{"pr_id", "integer", true},
		// The slug the webhook route authenticated, never a payload value.
		{"team_slug", "text", true},
		{"author", "text", false},
		{"title", "text", false},
		{"last_reviewed_commit", "text", false},
		{"summary_comment_id", "integer", false},
		{"summary_comment_version", "integer", false},
		{"opened_at", "timestamp with time zone", false},
		{"merged_at", "timestamp with time zone", false},
		{"declined_at", "timestamp with time zone", false},
		{"deleted_at", "timestamp with time zone", false},
		{"ignored_at", "timestamp with time zone", false},
		// NULL = unpriced. Cost fails open, so this must stay nullable.
		{"final_cost_nano_usd", "bigint", false},
		{"final_source_commit", "text", false},
		{"final_merge_commit", "text", false},
		{"final_lines_added", "integer", false},
		{"final_lines_removed", "integer", false},
		{"final_files_changed", "integer", false},
		{"riptide_emitted_at", "timestamp with time zone", false},
		{"created_at", "timestamp with time zone", true},
		{"updated_at", "timestamp with time zone", true},
	}
	for _, w := range want {
		got, ok := cols[w.name]
		if !ok {
			t.Errorf("pull_requests.%s is missing", w.name)
			continue
		}
		if got.dataType != w.dataType {
			t.Errorf("pull_requests.%s type = %s, want %s", w.name, got.dataType, w.dataType)
		}
		if got.notNull != w.notNull {
			t.Errorf("pull_requests.%s NOT NULL = %v, want %v", w.name, got.notNull, w.notNull)
		}
	}
}

// review_runs holds one row per run, with totals aggregated over it rather
// than accumulated in a column. A run counts even when its price does not, so
// cost_nano_usd is the one nullable column here.
func TestSchema_ReviewRunsIsRowsNotAccumulators(t *testing.T) {
	s := testStore(t)
	cols := columns(t, s, "review_runs")
	for _, name := range []string{
		"pull_request_id", "kind", "incremental", "to_commit", "model_label",
		"prompt_tokens", "cached_tokens", "completion_tokens", "elapsed_ms",
		"findings_posted", "lines_added", "lines_removed", "files_changed",
	} {
		got, ok := cols[name]
		if !ok {
			t.Errorf("review_runs.%s is missing", name)
			continue
		}
		if !got.notNull {
			t.Errorf("review_runs.%s is nullable; of the columns listed here "+
				"only cost may be NULL (from_commit is nullable by design: a "+
				"non-incremental run has no from commit)", name)
		}
	}
	if got, ok := cols["cost_nano_usd"]; !ok {
		t.Error("review_runs.cost_nano_usd is missing")
	} else {
		if got.notNull {
			t.Error("review_runs.cost_nano_usd is NOT NULL; an unpriced run must still be recorded")
		}
		// Nano-USD needs the full 64 bits: INTEGER overflows above ~2.15 USD.
		if got.dataType != "bigint" {
			t.Errorf("review_runs.cost_nano_usd type = %s, want bigint: "+
				"nano-USD overflows INTEGER at ~2.15 USD", got.dataType)
		}
	}

	// kind separates an auto run from a mention run everywhere downstream,
	// including the per-PR cost cap, which skips only later AUTO runs. The
	// CHECK is the only thing stopping a third value from being stored.
	var check string
	err := s.pool.QueryRow(context.Background(), `
		SELECT coalesce(string_agg(pg_get_constraintdef(c.oid), ' '), '')
		FROM pg_constraint c
		JOIN pg_class t ON t.oid = c.conrelid
		JOIN pg_namespace n ON n.oid = t.relnamespace
		WHERE n.nspname = current_schema() AND t.relname = 'review_runs'
		  AND c.contype = 'c'`).Scan(&check)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"'auto'", "'mention'"} {
		if !strings.Contains(check, want) {
			t.Errorf("review_runs has no CHECK admitting %s: %q", want, check)
		}
	}
}

// The columns InsertFinding writes and ExistingFindings reads back. Only
// comment_text is required: a model may return a finding with no headline,
// no suggestion and no confidence, and it must still be storable.
func TestSchema_FindingsCarriesEveryColumnTheStoreWrites(t *testing.T) {
	s := testStore(t)
	cols := columns(t, s, "findings")
	want := []columnSpec{
		{"pull_request_id", "bigint", true},
		{"review_run_id", "bigint", true},
		{"file_path", "text", true},
		{"line_number", "integer", true},
		{"severity", "text", true},
		{"comment_text", "text", true},
		{"confidence", "integer", false},
		{"headline", "text", false},
		{"suggestion", "text", false},
		{"bitbucket_comment_id", "integer", false},
	}
	for _, w := range want {
		got, ok := cols[w.name]
		if !ok {
			t.Errorf("findings.%s is missing", w.name)
			continue
		}
		if got.dataType != w.dataType {
			t.Errorf("findings.%s type = %s, want %s", w.name, got.dataType, w.dataType)
		}
		if got.notNull != w.notNull {
			t.Errorf("findings.%s NOT NULL = %v, want %v", w.name, got.notNull, w.notNull)
		}
	}
}

// UpsertPullRequest says ON CONFLICT (project_key, repo_slug, pr_id), which
// Postgres resolves against the exact column SET of a unique index. Matching
// indexdef on substrings would accept a superset - adding team_slug keeps
// every substring present while the upsert fails at runtime with "no unique
// or exclusion constraint matching the ON CONFLICT specification" - so this
// compares the column set itself. Order does not matter to the inference.
func TestSchema_PullRequestsIsUniquePerProjectRepoPR(t *testing.T) {
	s := testStore(t)
	rows, err := s.pool.Query(context.Background(), `
		SELECT array(
			SELECT a.attname
			FROM unnest(i.indkey) AS k(attnum)
			JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
			ORDER BY a.attname
		)
		FROM pg_index i
		JOIN pg_class c ON c.oid = i.indrelid
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = current_schema() AND c.relname = 'pull_requests'
		  AND i.indisunique AND i.indpred IS NULL`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	wantCols := []string{"project_key", "repo_slug", "pr_id"}
	slices.Sort(wantCols)
	want := strings.Join(wantCols, ",")
	var found bool
	var seen []string
	for rows.Next() {
		var cols []string
		if err := rows.Scan(&cols); err != nil {
			t.Fatal(err)
		}
		seen = append(seen, strings.Join(cols, ","))
		if strings.Join(cols, ",") == want {
			found = true
		}
	}
	if rows.Err() != nil {
		t.Fatal(rows.Err())
	}
	if !found {
		t.Errorf("no unique index on exactly (project_key, repo_slug, pr_id); "+
			"got %v. UpsertPullRequest's ON CONFLICT resolves against the "+
			"exact column set, so a superset breaks it at runtime", seen)
	}
}

func TestSchema_PullRequestsTeamAndLifecycleAreIndexed(t *testing.T) {
	s := testStore(t)
	if def := indexDef(t, s, "idx_pull_requests_team"); !strings.Contains(def, "team_slug") {
		t.Errorf("idx_pull_requests_team does not index team_slug: %s", def)
	}
	def := indexDef(t, s, "idx_pull_requests_lifecycle")
	if !strings.Contains(def, "merged_at") || !strings.Contains(def, "deleted_at") {
		t.Errorf("idx_pull_requests_lifecycle = %s, want (merged_at, deleted_at)", def)
	}
}

// Order matters here, so this compares the column list rather than testing
// for each name: ExistingFindings joins on f.pull_request_id, so the index
// is only usable for that lookup while pull_request_id LEADS. Reordered to
// (severity, line_number, file_path, pull_request_id) every name is still
// present and every review falls back to a seq scan over findings.
func TestSchema_FindingsKeepsTheDedupIndex(t *testing.T) {
	s := testStore(t)
	want := []string{"pull_request_id", "file_path", "line_number", "severity"}
	if got := indexColumns(t, s, "idx_findings_dedup"); !slices.Equal(got, want) {
		t.Errorf("idx_findings_dedup = %v, want %v (in this order: "+
			"ExistingFindings joins on pull_request_id, which must lead)", got, want)
	}
}

// One owner per repo, one per whole project. Both indexes are PARTIAL: a
// plain unique index cannot express the NULL repo_slug case, so a
// whole-project claim and a repo claim on the same project would collide.
func TestSchema_TeamClaimsArePartialUniquePerRepoAndProject(t *testing.T) {
	s := testStore(t)
	repo := indexDef(t, s, "uq_team_claims_repo")
	if !strings.Contains(repo, "UNIQUE") || !strings.Contains(repo, "repo_slug IS NOT NULL") {
		t.Errorf("uq_team_claims_repo = %s, want a partial unique index WHERE repo_slug IS NOT NULL", repo)
	}
	project := indexDef(t, s, "uq_team_claims_project")
	if !strings.Contains(project, "UNIQUE") || !strings.Contains(project, "repo_slug IS NULL") {
		t.Errorf("uq_team_claims_project = %s, want a partial unique index WHERE repo_slug IS NULL", project)
	}
}

// The default a team inherits before it ever writes settings. An infra repo
// that silently started being reviewed is the failure this pins.
func TestSchema_TeamSettingsDefaultsExcludeInfraRepos(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if _, err := s.pool.Exec(ctx,
		`INSERT INTO team_settings (team_slug) VALUES ('platform')`); err != nil {
		t.Fatal(err)
	}
	var excludes, autoAuthors, ignoreAuthors []string
	err := s.pool.QueryRow(ctx, `
		SELECT exclude_repos, auto_review_authors, ignore_authors
		FROM team_settings WHERE team_slug = 'platform'`).
		Scan(&excludes, &autoAuthors, &ignoreAuthors)
	if err != nil {
		t.Fatal(err)
	}
	if len(excludes) != 1 || excludes[0] != "*-infra" {
		t.Errorf("exclude_repos default = %v, want [*-infra]", excludes)
	}
	if len(autoAuthors) != 0 || len(ignoreAuthors) != 0 {
		t.Errorf("author lists default = %v / %v, want empty", autoAuthors, ignoreAuthors)
	}
}

// A run and its findings must go when the PR does, or a purged project
// leaves orphans behind.
func TestSchema_RunsAndFindingsCascadeFromThePullRequest(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "abc123")
	var runID int64
	if err := s.pool.QueryRow(ctx, `
		INSERT INTO review_runs (pull_request_id, kind, incremental, to_commit,
			model_label, prompt_tokens, cached_tokens, completion_tokens,
			elapsed_ms, findings_posted, lines_added, lines_removed, files_changed)
		VALUES ($1, 'auto', false, 'abc123', 'gpt-5.5', 1, 0, 1, 1, 0, 0, 0, 0)
		RETURNING id`, prID).Scan(&runID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.pool.Exec(ctx, `
		INSERT INTO findings (pull_request_id, review_run_id, file_path,
			line_number, severity, comment_text)
		VALUES ($1, $2, 'a.go', 1, 'issue', 'x')`, prID, runID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.pool.Exec(ctx, `DELETE FROM pull_requests WHERE id = $1`, prID); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"review_runs", "findings"} {
		var n int
		if err := s.pool.QueryRow(ctx,
			`SELECT count(*) FROM `+table+` WHERE pull_request_id = $1`, prID).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n != 0 {
			t.Errorf("%s still has %d row(s) after the pull request was deleted", table, n)
		}
	}
}
