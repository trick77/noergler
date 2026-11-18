package inference

import (
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// complianceFixture is a config change that drops an attribute, the shape of
// a requirement met elsewhere: the diff shows the attribute removed and one
// line added, the full file one more line the diff does not show.
func complianceFixture() diff.AnchorIndex {
	return diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path: "app/settings.config",
		Diff: "@@ -1,3 +1,3 @@\n" +
			" <settings>\n" +
			`-  <store builder="EnvBuilder" />` + "\n" +
			`+  <store mode="protected" />` + "\n" +
			" </settings>\n",
		Content:        "<settings>\n  <store mode=\"protected\" />\n</settings>\n<!-- keep last -->\n",
		ContentFetched: true,
	}})
}

func TestConfirmCompliance(t *testing.T) {
	idx := complianceFixture()
	met := func(file, quote string) ComplianceRequirement {
		return ComplianceRequirement{Requirement: "store is protected", Met: true, File: file, Quote: quote}
	}
	for _, tc := range []struct {
		name string
		req  ComplianceRequirement
		want ComplianceDrop
	}{
		{"added line proves it", met("app/settings.config", `<store mode="protected" />`), ""},
		{"diff marker copied along", met("app/settings.config", `+  <store mode="protected" />`), ""},
		{"side prefix on the path", met("b/app/settings.config", `<store mode="protected" />`), ""},
		{"full-file line outside the diff", met("app/settings.config", "<!-- keep last -->"), ""},
		{"removed line proves nothing", met("app/settings.config", `<store builder="EnvBuilder" />`), ComplianceQuoteRemoved},
		{"invented line", met("app/settings.config", `<store mode="encrypted" />`), ComplianceQuoteNotFound},
		{"file the prompt did not show", met("deploy/Deploy-Web.ps1", `<store mode="protected" />`), ComplianceUnknownFile},
		{"no quote", met("app/settings.config", "  "), ComplianceNoQuote},
		{"no file", met("", `<store mode="protected" />`), ComplianceNoQuote},
		// A bare marker strips to the blank line every fetched file ends
		// in; a brace or a closing tag matches some line of most files.
		{"lone diff marker", met("app/settings.config", "+"), ComplianceQuoteTooShort},
		{"brace", met("app/settings.config", "}"), ComplianceQuoteTooShort},
		{"closing tag", met("app/settings.config", "</store>"), ComplianceQuoteTooShort},
		// The case that misled: a requirement met outside the PR, called
		// unmet, quoting the changed line of the file the PR did touch.
		{"unmet is never kept", ComplianceRequirement{Requirement: "encrypt the config on deploy", Met: false,
			File: "app/settings.config", Quote: `<store builder="EnvBuilder" />`}, ComplianceNotMet},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kept, checks := ConfirmCompliance([]ComplianceRequirement{tc.req}, idx, nil)
			if len(checks) != 1 || checks[0].Drop != tc.want {
				t.Fatalf("checks = %+v, want drop %q", checks, tc.want)
			}
			if (len(kept) == 1) != (tc.want == "") {
				t.Errorf("kept = %+v, want kept only when nothing drops it", kept)
			}
		})
	}
}

// A removed line that still stands elsewhere in the file as it is now is
// code at the PR head: the file decides, not the hunk.
func TestConfirmComplianceRemovedButStillInFile(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path:           "deploy/Deploy-Web.ps1",
		Diff:           "@@ -1,2 +1,1 @@\n-Lock-Store settings\n Write-Host done\n",
		Content:        "Write-Host done\nLock-Store settings\n",
		ContentFetched: true,
	}})
	kept, _ := ConfirmCompliance([]ComplianceRequirement{{Requirement: "lock the store", Met: true,
		File: "deploy/Deploy-Web.ps1", Quote: "Lock-Store settings"}}, idx, nil)
	if len(kept) != 1 {
		t.Errorf("kept = %+v, want the requirement kept", kept)
	}
}

// On an incremental review the summary is still the PR's: a line the
// cumulative diff shows for a file the latest push never touched, added or
// unchanged, proves its requirement. A line the PR removed proves nothing,
// and neither does one the cumulative diff never showed.
func TestConfirmComplianceAcrossPushes(t *testing.T) {
	// The latest push changed only the config.
	idx := complianceFixture()
	history := diff.BuildAnchorIndex(diff.FilesOf("diff --git a/deploy/Deploy-Web.ps1 b/deploy/Deploy-Web.ps1\n" +
		"--- a/deploy/Deploy-Web.ps1\n+++ b/deploy/Deploy-Web.ps1\n" +
		"@@ -1,3 +1,3 @@\n Write-Host starting\n-Unlock-Store settings\n+Lock-Store settings\n Write-Host finished\n"))

	for _, tc := range []struct {
		name, file, quote string
		want              ComplianceDrop
	}{
		{"line an earlier push added", "deploy/Deploy-Web.ps1", "Lock-Store settings", ""},
		{"unchanged line the cumulative diff shows", "deploy/Deploy-Web.ps1", "Write-Host starting", ""},
		{"side prefix on the path", "b/deploy/Deploy-Web.ps1", "Lock-Store settings", ""},
		{"line the PR removed", "deploy/Deploy-Web.ps1", "Unlock-Store settings", ComplianceUnknownFile},
		{"line never shown", "deploy/Deploy-Web.ps1", "Remove-Item cache", ComplianceUnknownFile},
		{"line in another file", "deploy/Deploy-Worker.ps1", "Lock-Store settings", ComplianceUnknownFile},
		{"the latest push still counts first", "app/settings.config", `<store mode="protected" />`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := ComplianceRequirement{Requirement: "lock the store", Met: true, File: tc.file, Quote: tc.quote}
			kept, checks := ConfirmCompliance([]ComplianceRequirement{req}, idx, history)
			if checks[0].Drop != tc.want {
				t.Errorf("drop = %q, want %q", checks[0].Drop, tc.want)
			}
			if tc.want == "" && kept[0].File != StripSidePrefix(tc.file) {
				t.Errorf("kept file = %q, want the shown path", kept[0].File)
			}
		})
	}
	// Unmet stays unmet, whatever the PR shows.
	_, checks := ConfirmCompliance([]ComplianceRequirement{{Requirement: "lock the store", Met: false,
		File: "deploy/Deploy-Web.ps1", Quote: "Lock-Store settings"}}, idx, history)
	if checks[0].Drop != ComplianceNotMet {
		t.Errorf("drop = %q, want %q", checks[0].Drop, ComplianceNotMet)
	}
}

// Order is the model's, and an empty input is an empty list, never nil: nil
// is the summary's "extraction failed".
func TestConfirmComplianceKeepsOrderAndNeverNil(t *testing.T) {
	idx := complianceFixture()
	kept, _ := ConfirmCompliance(nil, idx, nil)
	if kept == nil || len(kept) != 0 {
		t.Errorf("kept = %#v, want an empty non-nil list", kept)
	}
	reqs := []ComplianceRequirement{
		{Requirement: "B", Met: true, File: "app/settings.config", Quote: `<store mode="protected" />`},
		{Requirement: "X", Met: false},
		{Requirement: "A", Met: true, File: "app/settings.config", Quote: "<!-- keep last -->"},
	}
	kept, checks := ConfirmCompliance(reqs, idx, nil)
	if len(kept) != 2 || kept[0].Requirement != "B" || kept[1].Requirement != "A" {
		t.Errorf("kept = %+v, want B then A", kept)
	}
	if len(checks) != 3 {
		t.Errorf("checks = %+v, want one per requirement", checks)
	}
}
