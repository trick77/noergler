package render

import "testing"

// Expected values generated from the Python (_strip_stale_banner,
// _strip_cost_banner, _cost_limit_banner) with the venv.
func TestStripStaleBanner(t *testing.T) {
	const body = "### Overview\nBody."
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "no banner returns unchanged",
			in:   "Just a normal summary.\n\nWith two paragraphs.",
			want: "Just a normal summary.\n\nWith two paragraphs.",
		},
		{
			name: "sentinel form is stripped",
			in:   StaleBannerSentinel + "\n⚠️ No response from the model within 10 minutes on commit `abc`.\n\n" + body,
			want: body,
		},
		{
			name: "visible prefix alone is stripped when the renderer ate the sentinel",
			in:   "⚠️ No response from the model within 10 minutes on commit `abc` — findings below reflect an earlier commit.\n\n" + body,
			want: body,
		},
		{
			name: "an unrelated warning emoji is left alone",
			in:   "⚠️ **Something else entirely** happened here.\n\n" + body,
			want: "⚠️ **Something else entirely** happened here.\n\n" + body,
		},
		{
			name: "several blank lines after the banner are all consumed",
			in:   StaleBannerSentinel + "\n⚠️ No response from the model within 10 minutes — findings below reflect an earlier commit.\n\n\n" + body,
			want: body,
		},
		{
			name: "empty body",
			in:   "",
			want: "",
		},
		{
			// Leading blank lines: the "skip to the first blank" loop ends
			// immediately at line 0, then the "skip blanks" loop eats both,
			// so the banner row itself survives and only the blanks go.
			// Unreachable with our own banners, which never lead with a
			// blank line; pinned because it is surprising.
			name: "leading blank lines are consumed but the banner row survives",
			in:   "\n\n⚠️ No response from the model within 10 minutes — findings below reflect an earlier commit.\n\nBody.",
			want: "⚠️ No response from the model within 10 minutes — findings below reflect an earlier commit.\n\nBody.",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := StripStaleBanner(c.in); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestStripCostBanner(t *testing.T) {
	const body = "### Overview\nBody."
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no banner", "Just a normal summary.\n\nWith two paragraphs.", "Just a normal summary.\n\nWith two paragraphs."},
		{"sentinel form", CostBannerSentinel + "\n⚠️ **Cost limit exceeded** — PR total **$1.00**, over the **$0.50** limit.\n\n" + body, body},
		{"visible prefix only", "⚠️ **Cost limit exceeded** — PR total **$1.00**.\n\n" + body, body},
		{"unrelated warning", "⚠️ Some other warning.\n\nBody.", "⚠️ Some other warning.\n\nBody."},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := StripCostBanner(c.in); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// Repeated failures must replace the banner, never stack them. This is the
// property test_repeated_timeout_does_not_stack_banners pins in Python.
func TestStrippersAreIdempotentAndDoNotStack(t *testing.T) {
	body := "### Overview\nBody."
	first := StaleBanner("⚠️ No response from the model within 10 minutes on commit `aaa`.") + "\n\n" + body
	// A second failure strips the old banner before prepending the new one.
	second := StaleBanner("⚠️ No response from the model within 10 minutes on commit `bbb`.") + "\n\n" + StripStaleBanner(first)
	if got := StripStaleBanner(second); got != body {
		t.Errorf("banners stacked: stripping once left %q", got)
	}
	if got := StripStaleBanner(StripStaleBanner(first)); got != body {
		t.Errorf("not idempotent: %q", got)
	}
}

// All three notice kinds share the stale banner, so the visible-prefix
// fallback has to catch all three. Python matched only the timeout wording,
// which was fine while timeouts were the only user; routing unparseable and
// too-large through the same banner would otherwise stack them whenever
// Bitbucket's renderer ate the sentinel, which is the only case the fallback
// exists for.
func TestStaleBannerFallbackCoversAllThreeNoticeKinds(t *testing.T) {
	const body = "### Overview\nThe original body."
	banners := map[string]string{
		"timeout":     "⚠️ No response from the model within 5 minutes on commit `abc` — findings below reflect an earlier commit.",
		"unparseable": "⚠️ The model returned an unprocessable response on commit `abc` — findings below reflect an earlier commit.",
		"too large":   "⚠️ Commit `abc` is too large to review within the model's context window — findings below reflect the earlier commit `def`.",
	}
	for name, line := range banners {
		t.Run(name, func(t *testing.T) {
			// The sentinel is gone, as a renderer that strips HTML comments
			// would leave it.
			withBanner := line + "\n\n" + body
			if got := StripStaleBanner(withBanner); got != body {
				t.Errorf("banner not stripped, so a second failure would stack it:\n got: %q", got)
			}
		})
	}
}

// The marker alone is not enough: an unrelated warning the model wrote must
// survive, or a real finding could be eaten.
func TestStaleBannerFallbackIgnoresUnrelatedWarnings(t *testing.T) {
	for _, body := range []string{
		"⚠️ **Something else entirely** happened here.\n\nBody.",
		"The findings below reflect an earlier commit, says the model.\n\nBody.",
		"⚠️ A warning with no marker at all.\n\nBody.",
	} {
		if got := StripStaleBanner(body); got != body {
			t.Errorf("stripped an unrelated warning\n in:  %q\n got: %q", body, got)
		}
	}
}

func TestCostLimitBanner(t *testing.T) {
	const want = CostBannerSentinel + "\n⚠️ **Cost limit exceeded** — PR total **$1.23**, over the **$0.50** limit. " +
		"Automatic reviews are paused. `@noergler` to review manually, or ask an admin to raise `REVIEW_MAX_PR_COST_USD`."

	if got := CostLimitBanner(1.2345, 0.5, "noergler", false); got != want {
		t.Errorf("completed-run banner\n got:  %q\n want: %q", got, want)
	}
	blocked := want + " This push was **not** reviewed automatically."
	if got := CostLimitBanner(1.2345, 0.5, "noergler", true); got != blocked {
		t.Errorf("blocked banner\n got:  %q\n want: %q", got, blocked)
	}
}

// The banner a cost notice writes must be strippable by the cost stripper, or
// repeated blocked pushes stack.
func TestCostBannerRoundTrips(t *testing.T) {
	body := "### Overview\nBody."
	withBanner := CostLimitBanner(1.0, 0.5, "noergler", true) + "\n\n" + body
	if got := StripCostBanner(withBanner); got != body {
		t.Errorf("round trip failed: %q", got)
	}
}
