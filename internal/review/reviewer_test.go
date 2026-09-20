package review

import (
	"testing"

	"github.com/trick77/noergler-go/internal/config"
)

// Probed against the venv in both directions. A naive `\b([A-Z]{2,10}-\d{1,7})\b`
// port matches the first three (Python does not: its \b is Unicode, so a
// letter next to the key means no boundary) and misses the Arabic-Indic case
// (Python's \d is Unicode).
func TestExtractTicketIDMatchesPython(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"ABC-123", "ABC-123"},
		{"üABC-123", ""},
		{"ABC-123ü", ""},
		{"ÄABC-123", ""},
		{"ABC-١٢٣", "ABC-١٢٣"},
		{"xABC-123", ""},
		{"ABC-123x", ""},
		{"(ABC-123)", "ABC-123"},
		{"AB-1234567", "AB-1234567"},
		{"AB-12345678", ""},
		{"feature/ABC-123-do-thing", "ABC-123"},
		{"ABC-123: fix the thing", "ABC-123"},
		{"no ticket here", ""},
		{"a-1", ""},
		{"TOOLONGPREFIX-1", ""},
	}
	for _, c := range cases {
		if got := extractTicketID(c.in, ""); got != c.want {
			t.Errorf("extractTicketID(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// The branch name wins; the title is the fallback.
func TestExtractTicketIDPrefersBranch(t *testing.T) {
	if got := extractTicketID("feature/ABC-1-thing", "DEF-2 title"); got != "ABC-1" {
		t.Errorf("= %q, want ABC-1 (branch wins)", got)
	}
	if got := extractTicketID("feature/no-key", "DEF-2 title"); got != "DEF-2" {
		t.Errorf("= %q, want DEF-2 (title fallback)", got)
	}
}

// The ignore list wins over the allow list; an empty allow list means every
// author except the ignored ones.
func TestIsAutoReviewAuthor(t *testing.T) {
	cases := []struct {
		name   string
		allow  []string
		ignore []string
		author string
		want   bool
	}{
		{"empty allow list allows anyone", nil, nil, "alice", true},
		{"author in allow list", []string{"alice"}, nil, "alice", true},
		{"author not in allow list", []string{"alice"}, nil, "bob", false},
		{"ignore wins over allow", []string{"alice"}, []string{"alice"}, "alice", false},
		{"ignored with empty allow list", nil, []string{"ci-bot"}, "ci-bot", false},
		{"not ignored with empty allow list", nil, []string{"ci-bot"}, "alice", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := &Reviewer{cfg: config.Review{AutoReviewAuthors: c.allow, IgnoreAuthors: c.ignore}}
			if got := r.IsAutoReviewAuthor(c.author); got != c.want {
				t.Errorf("IsAutoReviewAuthor(%q) = %v, want %v", c.author, got, c.want)
			}
		})
	}
}

func TestShortSHA(t *testing.T) {
	if got := shortSHA("abcdef0123456789", 8); got != "abcdef01" {
		t.Errorf("= %q", got)
	}
	if got := shortSHA("abc", 8); got != "abc" {
		t.Errorf("short input changed: %q", got)
	}
	if got := shortSHA("", 8); got != "" {
		t.Errorf("empty input = %q", got)
	}
}
