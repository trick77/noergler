package store

import (
	"strings"
	"testing"
)

// almanach_notes is one row per PR with the merge stamps and the forwarder's
// mark beside the note; everything but the key and the title may be NULL,
// because a finalize can land before any round wrote a note (a stub).
func TestSchema_AlmanachNotesCarriesStubsAndStamps(t *testing.T) {
	s := testStore(t)
	cols := columns(t, s, "almanach_notes")
	want := []columnSpec{
		{"project_key", "text", true},
		{"repo_slug", "text", true},
		{"pr_id", "integer", true},
		{"team_slug", "text", true},
		{"pr_title", "text", true},
		{"run_id", "bigint", false},
		{"note", "jsonb", false},
		{"chars", "integer", false},
		{"change_kind", "text", false},
		{"jira_issue_type", "text", false},
		{"finalized_at", "timestamp with time zone", false},
		{"merged_at", "timestamp with time zone", false},
		{"merge_commit", "text", false},
		{"source_commit", "text", false},
		{"issues_posted", "integer", false},
		{"suggestions_posted", "integer", false},
		{"almanach_emitted_at", "timestamp with time zone", false},
	}
	for _, w := range want {
		got, ok := cols[w.name]
		if !ok {
			t.Errorf("almanach_notes.%s is missing", w.name)
			continue
		}
		if got.dataType != w.dataType || got.notNull != w.notNull {
			t.Errorf("almanach_notes.%s = %s NOT NULL %v, want %s NOT NULL %v", w.name, got.dataType, got.notNull, w.dataType, w.notNull)
		}
	}
	if got := indexColumns(t, s, "almanach_notes_pending_idx"); len(got) != 2 || got[0] != "team_slug" || got[1] != "finalized_at" {
		t.Errorf("pending index columns = %v, want team_slug, finalized_at", got)
	}
	if def := indexDef(t, s, "almanach_notes_pending_idx"); !strings.Contains(def, "WHERE") {
		t.Errorf("pending index is not partial: %s", def)
	}
	// The cost lane keeps the full 64 bits and stays nullable, like llm_calls'.
	calls := columns(t, s, "almanach_calls")
	if got := calls["cost_nano_usd"]; got.dataType != "bigint" || got.notNull {
		t.Errorf("almanach_calls.cost_nano_usd = %+v, want nullable bigint", got)
	}
	if got := calls["note_status"]; !got.notNull {
		t.Error("almanach_calls.note_status is nullable; every call row says what became of the note")
	}
}
