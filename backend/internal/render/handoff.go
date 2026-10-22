package render

import (
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/lang"
)

// The handoff banner: written on the summary when a draft PR goes ready
// while noergler's finding threads are still open, so the human reviewers
// it is handed to see that the bot's round is not closed (user,
// 2026-10-22: a draft is the bot's turn, ready is the humans').
const (
	HandoffBannerSentinel = "<!-- noergler-handoff-banner -->"
	handoffVisibleEN      = "🔎 **Ready for review**"
	handoffVisibleDE      = "🔎 **Bereit für Reviewer**"
	// HandoffMaxLinks is how many open threads the banner links; the rest
	// are counted.
	HandoffMaxLinks = 5
)

// HandoffThread is one open finding thread the banner links to.
type HandoffThread struct {
	File string
	Line int
	URL  string
}

// HandoffBanner is the sentinel-marked banner for open threads. One line:
// the strippers end a banner at its first blank line.
func HandoffBanner(l lang.Lang, open []HandoffThread) string {
	n := len(open)
	links := make([]string, 0, min(n, HandoffMaxLinks))
	for _, t := range open[:min(n, HandoffMaxLinks)] {
		where := t.File
		if t.Line > 0 {
			where = fmt.Sprintf("%s:%d", t.File, t.Line)
		}
		links = append(links, fmt.Sprintf("[`%s`](%s)", where, t.URL))
	}
	list := strings.Join(links, ", ")
	more := n - len(links)
	var msg string
	if l == lang.EN {
		threads := "threads"
		if n == 1 {
			threads = "thread"
		}
		msg = fmt.Sprintf("%s — %d noergler %s still open: %s", handoffVisibleEN, n, threads, list)
		if more > 0 {
			msg += fmt.Sprintf(" and %d more", more)
		}
	} else {
		threads := "offene noergler-Threads"
		if n == 1 {
			threads = "offener noergler-Thread"
		}
		msg = fmt.Sprintf("%s — noch %d %s: %s", handoffVisibleDE, n, threads, list)
		if more > 0 {
			msg += fmt.Sprintf(" und %d weitere", more)
		}
	}
	return HandoffBannerSentinel + "\n" + msg + "."
}

// StripHandoffBanner removes a leading handoff banner block. Idempotent.
func StripHandoffBanner(body string) string {
	return stripBanner(body, HandoffBannerSentinel, func(line string) bool {
		return strings.HasPrefix(line, handoffVisibleEN) || strings.HasPrefix(line, handoffVisibleDE)
	})
}

// PlaceHandoffBanner rewrites a summary body with banner in place of any
// handoff banner it carries; an empty banner only removes the old one.
//
// The cost and stale strippers look at the top of the body only, so the
// handoff banner always goes below them: on top it would hide theirs from
// their next run, and those would stack. Banners on top are peeled off in
// order, kept, and put back above it.
func PlaceHandoffBanner(body, banner string) string {
	var head []string
	for {
		if s := StripHandoffBanner(body); s != body {
			body = s
			continue
		}
		if s := StripCostBanner(body); s != body {
			head = append(head, strings.TrimRight(body[:len(body)-len(s)], "\n"))
			body = s
			continue
		}
		if s := StripStaleBanner(body); s != body {
			head = append(head, strings.TrimRight(body[:len(body)-len(s)], "\n"))
			body = s
			continue
		}
		break
	}
	if banner != "" {
		head = append(head, banner)
	}
	return strings.Join(append(head, body), "\n\n")
}
