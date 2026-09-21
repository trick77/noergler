package onboarding

import "testing"

// The tables are golden: compared byte for byte, so any drift in the
// fixed-width layout fails here.
func TestRenderStatusGolden(t *testing.T) {
	rows := []StatusRow{
		{Target{Project: "PROJ"}, true, true, "ok", []string{"PROJ/x"}, nil},
		{Target{Project: "PROJ", Repo: "my-repo"}, true, false, "missing", nil, nil},
		{Target{Project: "OTHER"}, false, false,
			"not owned by this team in teams.yaml; ask the noergler admin", nil, nil},
		{Target{Project: "LONGPROJECTKEY"}, true, true,
			"stale: url: None -> 'https://n.test/webhook/p'; active: False -> True",
			[]string{"A/b", "A/c"}, []string{"A/d -> https://old.test/webhook"}},
	}
	if got, want := RenderStatus(rows), golden(t, "status.golden"); got != want {
		t.Errorf("RenderStatus =\n%s\n---want---\n%s", got, want)
	}
}

func TestRenderResultsGolden(t *testing.T) {
	results := []TargetResult{
		{Target{Project: "PROJ"}, "ok", "webhook created (id=42), noergler granted PROJECT_WRITE", nil},
		{Target{Project: "PROJ", Repo: "my-repo"}, "failed", "upsert webhook HTTP 401: nope", nil},
		{Target{Project: "LONGPROJECTKEY", Repo: "r"}, "skipped", "no 'noergler' webhook found", nil},
	}
	if got, want := RenderResults(results), golden(t, "results.golden"); got != want {
		t.Errorf("RenderResults =\n%s\n---want---\n%s", got, want)
	}
}

// An empty table falls back to a target-column width of 10.
func TestRenderEmptyTablesUseWidthTen(t *testing.T) {
	cases := []struct{ got, want string }{
		{RenderStatus(nil), "target      owned  bot   webhook\n--------------------------------"},
		{RenderResults(nil), "target      status   detail\n---------------------------"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("empty table =\n%q\nwant\n%q", c.got, c.want)
		}
	}
}

func TestStatusHealthy(t *testing.T) {
	ok := StatusRow{Target{Project: "P"}, true, true, "ok", nil, nil}
	cases := []struct {
		name string
		rows []StatusRow
		want bool
	}{
		{"empty is vacuously healthy", nil, true},
		{"all ok", []StatusRow{ok}, true},
		// A foreign hook belongs to another instance and does not make this
		// one unhealthy.
		{"foreign is not counted", []StatusRow{{Target{Project: "P"}, true, true, "ok", nil, []string{"P/r -> https://old"}}}, true},
		{"not owned", []StatusRow{{Target{Project: "P"}, false, true, "ok", nil, nil}}, false},
		{"bot cannot read", []StatusRow{{Target{Project: "P"}, true, false, "ok", nil, nil}}, false},
		{"stale hook", []StatusRow{{Target{Project: "P"}, true, true, "stale: x", nil, nil}}, false},
		{"stray hooks", []StatusRow{{Target{Project: "P"}, true, true, "ok", []string{"P/r"}, nil}}, false},
		{"one bad row spoils it", []StatusRow{ok, {Target{Project: "Q"}, true, true, "missing", nil, nil}}, false},
	}
	for _, c := range cases {
		if got := StatusHealthy(c.rows); got != c.want {
			t.Errorf("%s: StatusHealthy = %t, want %t", c.name, got, c.want)
		}
	}
}

func TestResultsHealthy(t *testing.T) {
	cases := []struct {
		name string
		rows []TargetResult
		want bool
	}{
		{"empty is vacuously healthy", nil, true},
		{"ok and skipped", []TargetResult{{Status: "ok"}, {Status: "skipped"}}, true},
		{"one failure", []TargetResult{{Status: "ok"}, {Status: "failed"}}, false},
	}
	for _, c := range cases {
		if got := ResultsHealthy(c.rows); got != c.want {
			t.Errorf("%s: ResultsHealthy = %t, want %t", c.name, got, c.want)
		}
	}
}

// ljust pads but never truncates, so a long label overhangs its column.
func TestLjustNeverTruncates(t *testing.T) {
	if got := ljust("abc", 5); got != "abc  " {
		t.Errorf("ljust pad = %q", got)
	}
	if got := ljust("abcdefgh", 3); got != "abcdefgh" {
		t.Errorf("ljust truncated: %q", got)
	}
}
