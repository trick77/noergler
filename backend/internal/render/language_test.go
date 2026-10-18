package render

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
)

func TestSwissSSLeavesCodeAlone(t *testing.T) {
	cases := map[string]string{
		"Das Maß ist groß.":               "Das Mass ist gross.",
		"GROẞ":                            "GROSS",
		"`straße` bleibt, Straße nicht":   "`straße` bleibt, Strasse nicht",
		"```go\nx := \"ß\"\n```\nfließt":  "```go\nx := \"ß\"\n```\nfliesst",
		"offen ` bis Zeilenende ß\nund ß": "offen ` bis Zeilenende ß\nund ss",
		"keine Änderung":                  "keine Änderung",
		"~~~\ns := \"Straße\"\n~~~\nGroß": "~~~\ns := \"Straße\"\n~~~\nGross",
		"``a`ß`` und ß":                   "``a`ß`` und ss",
		"Text:\n\n    x := \"ß\"\nEnde ß": "Text:\n\n    x := \"ß\"\nEnde ss",
	}
	for in, want := range cases {
		if got := SwissSS(in); got != want {
			t.Errorf("SwissSS(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGermanSummary(t *testing.T) {
	in := base()
	in.Lang = lang.DE
	h := "Fehlende Prüfung"
	in.Findings = []inference.ReviewFinding{{Severity: "issue", Comment: "Das Maß fehlt.", Headline: &h}}
	in.Summary.Overview = "Der Code ist groß."
	in.FilesCountsSet, in.FilesReviewed, in.TotalFiles = true, 1, 1
	in.Ticket = &jira.Ticket{Key: "ABC-1", URL: "u", Title: "T"}
	in.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Maße werden geprüft", Met: true}}

	got := Summary(in)
	for _, want := range []string{
		"### Überblick\nDer Code ist gross.", "### Probleme und Vorschläge", "1. Fehlende Prüfung",
		"### Empfehlung\n**Freigeben ✅**", "_1 Datei geprüft_",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("summary lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "ß") {
		t.Errorf("German summary keeps a ß:\n%s", got)
	}
	// The caller's findings are stored as they came.
	if in.Findings[0].Comment != "Das Maß fehlt." {
		t.Error("Summary changed the caller's finding")
	}
}

func TestGermanInlineComment(t *testing.T) {
	s := "x := maß"
	got := InlineComment(inference.ReviewFinding{Severity: "suggestion", Comment: "Das Maß.", Suggestion: &s}, lang.DE)
	want := "**Vorschlag:** Das Mass.\n\n**Vorgeschlagene Änderung:**\n```suggestion\nx := maß\n```"
	if got != want {
		t.Errorf("InlineComment =\n%s\nwant\n%s", got, want)
	}
}

// A PR author who switches language leaves a banner in the old one; the
// next run must still replace it, sentinel or not.
func TestBannersStripInBothLanguages(t *testing.T) {
	body := "summary body"
	for _, l := range []lang.Lang{lang.DE, lang.EN} {
		banner := CostLimitBanner(l, 6, 5, "noergler", true)
		noSentinel := strings.TrimPrefix(banner, CostBannerSentinel+"\n")
		if got := StripCostBanner(noSentinel + "\n\n" + body); got != body {
			t.Errorf("%s cost banner not stripped without its sentinel: %q", l, got)
		}
	}
	de := "⚠️ Commit `abc` ist zu gross — " + StaleBannerMarkerDE + " einen früheren Commit."
	if got := StripStaleBanner(de + "\n\n" + body); got != body {
		t.Errorf("German stale banner not stripped: %q", got)
	}
}

func TestGermanSecurityKeywords(t *testing.T) {
	for _, c := range []string{"Hier droht eine SQL-Injektion.", "Die Zugangsdaten landen im Log.", "Eine Sicherheitslücke",
		"Injektionen über den Query-Parameter", "fehlende Autorisierungsprüfung"} {
		if !IsSecurityFinding(c) {
			t.Errorf("IsSecurityFinding(%q) = false", c)
		}
	}
	if IsSecurityFinding("Überinjektion") {
		t.Error("a keyword inside a word must not match")
	}
	// Bereinigung is any cleanup in German, not sanitization.
	if IsSecurityFinding("Eine Bereinigung der ungenutzten Imports wäre sinnvoll.") {
		t.Error("a cleanup suggestion counted as a security issue")
	}
}
