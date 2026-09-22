package onboarding

import "strings"

// labelWidth is the width of the target column: the longest label, or 10 for
// an empty table.
func labelWidth[T any](items []T, label func(T) string) int {
	if len(items) == 0 {
		return 10
	}
	w := 0
	for _, it := range items {
		if n := len([]rune(label(it))); n > w {
			w = n
		}
	}
	return w
}

// ljust pads s on the right to n runes. It never truncates, so a long value
// overhangs its column.
func ljust(s string, n int) string {
	if pad := n - len([]rune(s)); pad > 0 {
		return s + strings.Repeat(" ", pad)
	}
	return s
}

// RenderStatus is the fixed-width status table. No trailing newline. The rule
// under the header is as wide as the header, so a long webhook verdict hangs
// past it.
func RenderStatus(rows []StatusRow) string {
	width := labelWidth(rows, func(r StatusRow) string { return r.Target.Label() })
	// "write", not "bot": the column answers whether the bot can POST a
	// comment, which is what a review needs. It used to say "bot" while
	// only proving the bot could read.
	header := ljust("target", width) + "  owned  write  webhook"
	lines := []string{header, strings.Repeat("-", len([]rune(header)))}
	for _, r := range rows {
		owned := "no"
		if r.Owned {
			owned = "yes"
		}
		bot := "-"
		if r.Owned {
			bot = "no"
			if r.BotCanWrite {
				bot = "yes"
			}
		}
		line := ljust(r.Target.Label(), width) + "  " + ljust(owned, 5) + "  " + ljust(bot, 5) + "  " + r.Webhook
		if len(r.Stray) > 0 {
			line += "  stray repo hooks: " + strings.Join(r.Stray, ", ")
		}
		if len(r.Foreign) > 0 {
			line += "  foreign hooks: " + strings.Join(r.Foreign, ", ")
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

// RenderResults is the fixed-width result table. No trailing newline.
func RenderResults(results []TargetResult) string {
	width := labelWidth(results, func(r TargetResult) string { return r.Target.Label() })
	header := ljust("target", width) + "  status   detail"
	lines := []string{header, strings.Repeat("-", len([]rune(header)))}
	for _, r := range results {
		lines = append(lines, ljust(r.Target.Label(), width)+"  "+ljust(r.Status, 7)+"  "+r.Detail)
	}
	return strings.Join(lines, "\n")
}

// StatusHealthy reports whether every row is owned, WRITABLE by the bot, has a working webhook,
// and has no stray repo hooks. Foreign hooks are deliberately not part of it.
//
// Writable, not readable: the bot posts review comments. An instance whose
// bot lost write access is not healthy, and used to say it was.
func StatusHealthy(rows []StatusRow) bool {
	for _, r := range rows {
		if !r.Owned || !r.BotCanWrite || r.Webhook != "ok" || len(r.Stray) > 0 {
			return false
		}
	}
	return true
}

// ResultsHealthy reports whether no row failed. A skip is not a failure.
func ResultsHealthy(results []TargetResult) bool {
	for _, r := range results {
		if r.Status == "failed" {
			return false
		}
	}
	return true
}
