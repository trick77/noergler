package render

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/trick77/noergler/internal/lang"
)

// skipGolden is the expected output of the skip-summary builders. These
// strings are posted verbatim on real PRs, so they are pinned whole rather
// than asserted by substring.
type skipGolden struct {
	OptOut                 string              `json:"opt_out"`
	AgentsMissing          string              `json:"agents_missing"`
	DiffTooLarge           string              `json:"diff_too_large"`
	AgentsTooLargeNoLink   string              `json:"agents_too_large_nolink"`
	AgentsTooLargeMarkdown string              `json:"agents_too_large_markdown"`
	AgentsTooLargeBareURL  string              `json:"agents_too_large_bareurl"`
	AgentsTooLargeParen    string              `json:"agents_too_large_paren"`
	ParseCustomLink        map[string][]string `json:"parse_custom_link"`
}

func loadSkipGolden(t *testing.T) skipGolden {
	t.Helper()
	blob, err := os.ReadFile("testdata/skip_golden.json")
	if err != nil {
		t.Fatalf("read skip golden: %v", err)
	}
	var g skipGolden
	if err := json.Unmarshal(blob, &g); err != nil {
		t.Fatalf("decode skip golden: %v", err)
	}
	return g
}

func TestSkipSummariesIsPinned(t *testing.T) {
	g := loadSkipGolden(t)

	if got := OptOutBranchSummary(lang.EN, "noergloff", "feature/noergloff-thing"); got != g.OptOut {
		t.Errorf("OptOutBranchSummary\n got:  %q\n want: %q", got, g.OptOut)
	}
	if got := AgentsMDMissingSummary(lang.EN, "AGENTS.md"); got != g.AgentsMissing {
		t.Errorf("AgentsMDMissingSummary\n got:  %q\n want: %q", got, g.AgentsMissing)
	}
	if got := DiffTooLargeSummary(lang.EN, 10*1024*1024); got != g.DiffTooLarge {
		t.Errorf("DiffTooLargeSummary\n got:  %q\n want: %q", got, g.DiffTooLarge)
	}
}

// The PR author cannot change teams.yaml; the summary only says how to add
// the file, never how to switch the requirement off.
func TestAgentsMDMissingSummaryHasNoOptOut(t *testing.T) {
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		got := AgentsMDMissingSummary(l, "AGENTS.md")
		for _, banned := range []string{"require_agents_md", "REVIEW_REQUIRE_AGENTS_MD", "teams.yaml", "Opt-out"} {
			if strings.Contains(got, banned) {
				t.Errorf("%s summary mentions %q:\n%s", l, banned, got)
			}
		}
		if !strings.Contains(got, "AGENTS.md") {
			t.Errorf("%s summary lost its body:\n%s", l, got)
		}
	}
}

func TestAgentsMDTooLargeSummaryIsPinned(t *testing.T) {
	g := loadSkipGolden(t)
	cases := []struct {
		name   string
		custom string
		want   string
	}{
		{"no custom link", "", g.AgentsTooLargeNoLink},
		{"markdown custom link", "[Our guide](https://intra/guide)", g.AgentsTooLargeMarkdown},
		{"bare url custom link", "https://intra/guide", g.AgentsTooLargeBareURL},
		{"url with a parenthesis", "[Wiki](https://en.wikipedia.org/wiki/Foo_(bar))", g.AgentsTooLargeParen},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := AgentsMDTooLargeSummary(lang.EN, "AGENTS.md", 12345, 7000, c.custom); got != c.want {
				t.Errorf("\n got:  %q\n want: %q", got, c.want)
			}
		})
	}
}

// A team that chose another name reads that name everywhere the summary
// talks about its file; "AGENTS.md" is left only in the further-reading
// titles, which are articles, not the team's file.
func TestSkipSummariesNameTheTeamsFile(t *testing.T) {
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		for what, got := range map[string]string{
			"missing":   AgentsMDMissingSummary(l, "REVIEW.md"),
			"too large": AgentsMDTooLargeSummary(l, "REVIEW.md", 12345, 7000, ""),
		} {
			body, _, _ := strings.Cut(got, "**Further reading**")
			body, _, _ = strings.Cut(body, "**Weiterführende Links**")
			if !strings.Contains(body, "`REVIEW.md`") {
				t.Errorf("%s %s summary does not name REVIEW.md:\n%s", l, what, got)
			}
			if strings.Contains(body, "AGENTS.md") {
				t.Errorf("%s %s summary still names AGENTS.md:\n%s", l, what, got)
			}
		}
	}
}

// The skip summaries wrap like the review summary (user, 2026-11-23), in both
// languages; the title and the further-reading links stay whole.
func TestSkipSummariesAreWrapped(t *testing.T) {
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		for what, got := range map[string]string{
			"opt-out":        OptOutBranchSummary(l, "noergloff", "feature/noergloff-thing"),
			"missing":        AgentsMDMissingSummary(l, "AGENTS.md"),
			"diff too large": DiffTooLargeSummary(l, 10*1024*1024),
			"too large":      AgentsMDTooLargeSummary(l, "AGENTS.md", 12345, 7000, "[A guide whose title has several words](https://intra/a/rather/long/path/to/the/guide/that/runs/on)"),
		} {
			lines := strings.Split(got, "\n")
			if len(lines) < 5 {
				t.Errorf("%s %s summary has %d lines", l, what, len(lines))
			}
			for _, line := range lines {
				if strings.HasPrefix(line, "###") || strings.HasPrefix(line, "- [") {
					continue
				}
				if n := utf8.RuneCountInString(line); n > WrapWidth {
					t.Errorf("%s %s summary: line of %d columns: %q", l, what, n, line)
				}
				if strings.Contains(line, emDash) {
					t.Errorf("%s %s summary: an em dash in the body, a list item when it starts a line: %q", l, what, line)
				}
			}
			// Links whole on their line, titles included.
			var links []string
			switch what {
			case "missing", "too large":
				links = append(links, l.Pick("[ETH-Studie](", "[ETH\u00a0study](")+ETHStudyURL+")")
			}
			if what == "too large" {
				for _, link := range agentsMDFurtherReading {
					links = append(links, "- ["+link.Title+"]("+link.URL+")\n")
				}
				links = append(links, "- [A guide whose title has several words](https://intra/a/rather/long/path/to/the/guide/that/runs/on)\n")
			}
			for _, link := range links {
				if !strings.Contains(got, link) {
					t.Errorf("%s %s summary: link not whole: %q in\n%s", l, what, link, got)
				}
			}
		}
	}
}

// TestSkipGoldenUpdate rewrites testdata/skip_golden.json's summaries from
// the current builders when SKIP_GOLDEN_UPDATE=1; otherwise it does nothing.
func TestSkipGoldenUpdate(t *testing.T) {
	if os.Getenv("SKIP_GOLDEN_UPDATE") != "1" {
		t.Skip("set SKIP_GOLDEN_UPDATE=1 to rewrite the golden")
	}
	g := loadSkipGolden(t)
	g.OptOut = OptOutBranchSummary(lang.EN, "noergloff", "feature/noergloff-thing")
	g.AgentsMissing = AgentsMDMissingSummary(lang.EN, "AGENTS.md")
	g.DiffTooLarge = DiffTooLargeSummary(lang.EN, 10*1024*1024)
	g.AgentsTooLargeNoLink = AgentsMDTooLargeSummary(lang.EN, "AGENTS.md", 12345, 7000, "")
	g.AgentsTooLargeMarkdown = AgentsMDTooLargeSummary(lang.EN, "AGENTS.md", 12345, 7000, "[Our guide](https://intra/guide)")
	g.AgentsTooLargeBareURL = AgentsMDTooLargeSummary(lang.EN, "AGENTS.md", 12345, 7000, "https://intra/guide")
	g.AgentsTooLargeParen = AgentsMDTooLargeSummary(lang.EN, "AGENTS.md", 12345, 7000, "[Wiki](https://en.wikipedia.org/wiki/Foo_(bar))")
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", " ")
	if err := enc.Encode(g); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("testdata/skip_golden.json", buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestParseCustomLink(t *testing.T) {
	g := loadSkipGolden(t)
	for in, want := range g.ParseCustomLink {
		got, ok := ParseCustomLink(in)
		if want == nil {
			if ok {
				t.Errorf("ParseCustomLink(%q) = %+v, want no link", in, got)
			}
			continue
		}
		if !ok {
			t.Errorf("ParseCustomLink(%q) returned no link, want %v", in, want)
			continue
		}
		if got.Title != want[0] || got.URL != want[1] {
			t.Errorf("ParseCustomLink(%q) = (%q, %q), want (%q, %q)", in, got.Title, got.URL, want[0], want[1])
		}
	}
}
