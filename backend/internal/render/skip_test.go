package render

import (
	"encoding/json"
	"os"
	"testing"
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

	if got := OptOutBranchSummary("noergloff", "feature/noergloff-thing"); got != g.OptOut {
		t.Errorf("OptOutBranchSummary\n got:  %q\n want: %q", got, g.OptOut)
	}
	if got := AgentsMDMissingSummary(); got != g.AgentsMissing {
		t.Errorf("AgentsMDMissingSummary\n got:  %q\n want: %q", got, g.AgentsMissing)
	}
	if got := DiffTooLargeSummary(10 * 1024 * 1024); got != g.DiffTooLarge {
		t.Errorf("DiffTooLargeSummary\n got:  %q\n want: %q", got, g.DiffTooLarge)
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
			if got := AgentsMDTooLargeSummary(12345, 7000, c.custom); got != c.want {
				t.Errorf("\n got:  %q\n want: %q", got, c.want)
			}
		})
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
