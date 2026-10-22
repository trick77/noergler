package render

import (
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

func threads(n int) []HandoffThread {
	out := make([]HandoffThread, n)
	for i := range out {
		out[i] = HandoffThread{File: "a.go", Line: i + 1, URL: fmt.Sprintf("https://bb/c%d", i+1)}
	}
	return out
}

func TestHandoffBannerCapsTheLinksAndCountsTheRest(t *testing.T) {
	got := HandoffBanner(lang.EN, threads(7))
	want := HandoffBannerSentinel + "\n🔎 **Ready for review** — 7 noergler threads still open: " +
		"[`a.go:1`](https://bb/c1), [`a.go:2`](https://bb/c2), [`a.go:3`](https://bb/c3), " +
		"[`a.go:4`](https://bb/c4), [`a.go:5`](https://bb/c5) and 2 more."
	if got != want {
		t.Errorf("banner =\n%s\nwant\n%s", got, want)
	}
	de := HandoffBanner(lang.DE, threads(6))
	if !strings.HasSuffix(de, " und 1 weitere.") || !strings.Contains(de, "noch 6 offene noergler-Threads: ") {
		t.Errorf("German banner = %s", de)
	}
	// A whole-file comment has no line.
	if got := HandoffBanner(lang.EN, []HandoffThread{{File: "a.go", URL: "u"}}); !strings.Contains(got, "[`a.go`](u).") {
		t.Errorf("file-level banner = %s", got)
	}
}

func TestStripHandoffBannerBySentinelAndByItsVisibleRow(t *testing.T) {
	body := "## Summary\n\nbody"
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		banner := HandoffBanner(l, threads(1))
		if got := StripHandoffBanner(banner + "\n\n" + body); got != body {
			t.Errorf("%s with sentinel: %q", l, got)
		}
		// A renderer that ate the sentinel.
		row := strings.TrimPrefix(banner, HandoffBannerSentinel+"\n")
		if got := StripHandoffBanner(row + "\n\n" + body); got != body {
			t.Errorf("%s without sentinel: %q", l, got)
		}
	}
	if got := StripHandoffBanner(body); got != body {
		t.Errorf("no banner: %q", got)
	}
}

// Every banner keeps its place and none stacks, however often each is
// written: cost and stale on top (their strippers look only there), the
// handoff below them.
func TestPlaceHandoffBannerNeverStacksAndStaysBelowTheOthers(t *testing.T) {
	body := "## Summary\n\nbody"
	cost := CostLimitBanner(lang.EN, 6, 5, false)
	stale := StaleBanner("⚠️ x — findings below reflect an earlier commit.")
	hand := HandoffBanner(lang.EN, threads(2))

	got := PlaceHandoffBanner(cost+"\n\n"+stale+"\n\n"+body, hand)
	want := cost + "\n\n" + stale + "\n\n" + hand + "\n\n" + body
	if got != want {
		t.Fatalf("placed =\n%s\nwant\n%s", got, want)
	}
	newer := HandoffBanner(lang.EN, threads(1))
	if again := PlaceHandoffBanner(got, newer); again != cost+"\n\n"+stale+"\n\n"+newer+"\n\n"+body {
		t.Errorf("replaced =\n%s", again)
	}
	if gone := PlaceHandoffBanner(got, ""); gone != cost+"\n\n"+stale+"\n\n"+body {
		t.Errorf("removed =\n%s", gone)
	}
	if same := PlaceHandoffBanner(body, ""); same != body {
		t.Errorf("no banner, none to remove: %q", same)
	}
	// The next cost write still finds its banner on top.
	if StripCostBanner(got) == got {
		t.Error("cost banner no longer on top")
	}
}
