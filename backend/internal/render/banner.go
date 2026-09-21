package render

import (
	"fmt"
	"strings"
)

// Sentinel markers for the two banners we prepend to an existing summary
// comment. Re-runs detect these and replace the previous banner instead of
// stacking new ones.
const (
	StaleBannerSentinel = "<!-- noergler-stale-banner -->"
	CostBannerSentinel  = "<!-- noergler-cost-banner -->"
)

// Visible markers, used as a defensive fallback in case Bitbucket's markdown
// renderer strips the HTML comment sentinel before returning the body.
// Without it, banners would stack on repeated failures.
//
// The cost banner has one wording, so a prefix identifies it. The stale
// banner has three (timeout, unparseable, too large), so matching only the
// timeout prefix would let a renderer that ate the sentinel stack the other
// two. staleBannerMarker is the phrase all six variants share, checked as a
// substring of the banner row rather than a prefix
// (TestStaleBannerFallbackCoversAllThreeNoticeKinds).
const (
	staleBannerVisiblePrefix = "⚠️"
	staleBannerMarker        = "findings below reflect"
	costBannerVisiblePrefix  = "⚠️ **Cost limit exceeded**"
)

// StripStaleBanner removes a leading staleness banner block. Idempotent.
func StripStaleBanner(body string) string {
	return stripBanner(body, StaleBannerSentinel, isStaleBannerRow)
}

// StripCostBanner removes a leading cost-limit banner block. Idempotent.
func StripCostBanner(body string) string {
	return stripBanner(body, CostBannerSentinel, func(line string) bool {
		return strings.HasPrefix(line, costBannerVisiblePrefix)
	})
}

// isStaleBannerRow reports whether a line looks like one of our three stale
// banners: the warning sign and the phrase every variant carries. Both are
// required, so an unrelated warning the model wrote is left alone.
func isStaleBannerRow(line string) bool {
	return strings.HasPrefix(line, staleBannerVisiblePrefix) &&
		strings.Contains(line, staleBannerMarker)
}

// stripBanner implements both strippers, which are the same algorithm.
//
// Detection priority is sentinel first, then the visible prefix on the first
// non-blank line. On a hit, everything up to and including the first blank
// line that follows the banner row is dropped.
//
// One edge is surprising and is reproduced deliberately: when a
// sentinel-less banner is preceded by blank lines, detection finds the prefix
// on the first non-blank line, but the strip scan starts at line 0, ends
// immediately, and then only eats the leading blanks. The banner row itself
// survives. Unreachable with our own banners, which never lead with a blank
// line; pinned so a tidy-up cannot quietly change it.
func stripBanner(body, sentinel string, isBannerRow func(string) bool) string {
	hasSentinel := strings.HasPrefix(body, sentinel)
	lines := strings.Split(body, "\n")
	if !hasSentinel {
		firstContent := ""
		for _, line := range lines {
			if strings.TrimSpace(line) != "" {
				firstContent = line
				break
			}
		}
		if !isBannerRow(firstContent) {
			return body
		}
	}

	i := 0
	if hasSentinel {
		i = 1
	}
	for i < len(lines) && strings.TrimSpace(lines[i]) != "" {
		i++
	}
	for i < len(lines) && strings.TrimSpace(lines[i]) == "" {
		i++
	}
	return strings.Join(lines[i:], "\n")
}

// StaleBanner builds the sentinel-marked banner block prepended to an
// existing summary when a re-review did not complete.
func StaleBanner(line string) string {
	return StaleBannerSentinel + "\n" + line
}

// CostLimitBanner builds the sentinel-marked over-limit banner block.
//
// blocked reports whether an automatic review was skipped entirely (the
// pushed commit was not reviewed); false is a completed run that overshot the
// cap, or a manual mention review while already over.
func CostLimitBanner(cumulativeUSD, limitUSD float64, botUsername string, blocked bool) string {
	msg := fmt.Sprintf(
		"%s — PR total **$%.2f**, over the **$%.2f** limit. Automatic reviews are paused. "+
			"`@%s` to review manually, or ask an admin to raise `REVIEW_MAX_PR_COST_USD`.",
		costBannerVisiblePrefix, cumulativeUSD, limitUSD, botUsername)
	if blocked {
		msg += " This push was **not** reviewed automatically."
	}
	return CostBannerSentinel + "\n" + msg
}
