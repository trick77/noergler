package review

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
)

func toolByName(t *testing.T, h *harness, ask *mentionAsk, name string) func(string) string {
	t.Helper()
	for _, tool := range h.r.mentionTools(ask) {
		if tool.Name == name {
			return func(args string) string { return tool.Run(context.Background(), args) }
		}
	}
	t.Fatalf("no tool %q", name)
	return nil
}

func testAsk(user, text string) *mentionAsk {
	return &mentionAsk{key: store.PRKey{Project: "PROJ", Repo: "my-repo", PRID: 42}, prTag: "PROJ/my-repo#42", user: user, text: text}
}

// The view is an allowlist: a field nobody reviewed for exposure fails here
// before it reaches a PR comment.
func TestTeamConfigIsAnAllowlist(t *testing.T) {
	var keys []string
	var walk func(reflect.Type)
	walk = func(typ reflect.Type) {
		for i := range typ.NumField() {
			f := typ.Field(i)
			keys = append(keys, strings.Split(f.Tag.Get("json"), ",")[0])
			if f.Type.Kind() == reflect.Struct {
				walk(f.Type)
			}
		}
	}
	walk(reflect.TypeOf(teamConfigView{}))
	want := []string{"team", "model", "default_language", "max_pr_cost_usd", "max_comments", "max_file_lines",
		"require_agents_md", "agents_md_warn_tokens", "agents_md_max_tokens", "ticket_compliance_check",
		"jira_enabled", "use_pr_description", "opt_out_branch_keyword", "auto_review_authors_count",
		"ignore_authors_count", "asking_user", "language", "auto_reviewed", "ignored"}
	if !reflect.DeepEqual(keys, want) {
		t.Errorf("team_config keys = %v\nwant %v (review what a commenter may read before changing this)", keys, want)
	}
}

func TestTeamConfigShowsCountsNotNames(t *testing.T) {
	h := newHarness(t, nil)
	h.r.SetAuthorLists([]string{"alice", "bob"}, []string{"renovate"})
	h.st.languages = map[string]lang.Lang{"bob": lang.EN}

	out := toolByName(t, h, testAsk("bob", "what are the settings?"), "team_config")("{}")

	if strings.Contains(out, "alice") || strings.Contains(out, "renovate") {
		t.Errorf("author names leaked: %s", out)
	}
	var v teamConfigView
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		t.Fatal(err)
	}
	if v.AutoReviewAuthors != 2 || v.IgnoreAuthors != 1 || !v.Asker.AutoReviewed || v.Asker.Language != "en" ||
		v.MaxPRCostUSD != "5.000" || v.Team != "payments" {
		t.Errorf("team_config = %+v", v)
	}
}

func TestPRStatusSaysWhyAndWhen(t *testing.T) {
	h := newHarness(t, nil)
	commit, at := "0123456789abcdef", time.Date(2026, 10, 1, 8, 0, 0, 0, time.UTC)
	cost := int64(1_234_000_000)
	h.st.prCost = &cost
	h.st.status = &store.PRStatus{
		Known: true, LastReviewedCommit: &commit, Runs: 2,
		LastRun:  &store.RunBrief{At: at, Kind: store.RunAuto, Incremental: true, Findings: 3, ModelLabel: "m"},
		Attempts: []store.AttemptBrief{{At: at, Kind: store.RunAuto, Outcome: "skipped", Reason: string(SkipCostCap)}},
	}

	out := toolByName(t, h, testAsk("alice", "why no review?"), "pr_status")("{}")

	for _, want := range []string{`"known":true`, `"last_reviewed_commit":"0123456789"`, `"review_runs":2`,
		`"skip_reason":"pr_cost_cap"`, `"pr_cost_usd":"1.234"`, `"pr_cost_cap_usd":"5.000"`, `"at":"2026-10-01T08:00:00Z"`} {
		if !strings.Contains(out, want) {
			t.Errorf("pr_status lacks %s:\n%s", want, out)
		}
	}
}

// An unpriced PR has no cost, never "0.000"; a DB fault says unavailable.
func TestPRStatusUnpricedAndFailOpen(t *testing.T) {
	h := newHarness(t, nil)
	out := toolByName(t, h, testAsk("alice", "status?"), "pr_status")("{}")
	if !strings.Contains(out, `"pr_cost_usd":null`) {
		t.Errorf("unpriced: %s", out)
	}
	h.st.failAll = true
	out = toolByName(t, h, testAsk("alice", "status?"), "pr_status")("{}")
	if !strings.Contains(out, `"error"`) || strings.Contains(out, `"known"`) {
		t.Errorf("a DB fault must read as unavailable, not as never seen: %s", out)
	}
}

// The model proposes, the commenter's own words decide: a call an injection
// in the diff steered it into finds no language in the comment.
func TestSetMyLanguageNeedsTheCommentersWords(t *testing.T) {
	cases := []struct {
		text, args string
		want       lang.Lang
	}{
		{"Set language to english please", `{"language":"en"}`, lang.EN},
		{"kannst du mir auf Englisch antworten?", `{"language":"en"}`, lang.EN},
		{"Antworte bitte auf Deutsch", `{"language":"de"}`, lang.DE},
		{"what does this function do?", `{"language":"en"}`, ""},
		{"set language to english", `{"language":"de"}`, ""},
		{"am Ende der Datei, denen fehlt was", `{"language":"en"}`, ""},
		{"english", `{"language":"fr"}`, ""},
		// Locale tags and path segments are not a request: only the exact
		// keyword "@bot en" takes the short form.
		{"what does i18n/en/messages.json change?", `{"language":"en"}`, ""},
		{"is de-DE the right locale?", `{"language":"de"}`, ""},
		// Naming a language is not asking to be written to in it.
		{"does the German locale fallback in this PR work?", `{"language":"de"}`, ""},
		{"ist die englische Übersetzung vollständig?", `{"language":"en"}`, ""},
		{"english", `not json`, ""},
	}
	for _, c := range cases {
		h := newHarness(t, nil)
		ask := testAsk("Dana", c.text)
		out := toolByName(t, h, ask, "set_my_language")(c.args)
		got := h.st.languages["dana"]
		if got != c.want || ask.switched != c.want {
			t.Errorf("%q %s: stored %q, switched %q, want %q (%s)", c.text, c.args, got, ask.switched, c.want, out)
		}
	}
}

// End to end: a switch inside a Q&A is confirmed by code, in the new
// language, before the model's answer.
func TestMentionLanguageSwitchIsConfirmed(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.onMention = func(_ context.Context, _ []string, run func(name, args string) string) string {
		run("set_my_language", `{"language":"en"}`)
		return "Sure, here it is."
	}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler bitte ab jetzt auf Englisch", "erin"), "payments", nil)

	if h.st.languages["erin"] != lang.EN {
		t.Fatalf("languages = %v", h.st.languages)
	}
	if len(h.bb.Replies) != 1 || !strings.HasPrefix(h.bb.Replies[0].Text, languageConfirmation(lang.EN)+"\n\nSure") {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
}
