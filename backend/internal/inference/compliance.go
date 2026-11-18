package inference

import (
	"strings"
	"unicode"

	"github.com/trick77/noergler/internal/diff"
)

// ComplianceDrop says why ConfirmCompliance left a requirement out. The
// values are log and run page vocabulary; do not reword them.
type ComplianceDrop string

const (
	// ComplianceNotMet means the model listed the requirement as not met. A
	// PR is a delta, so "not met" is never proven: the requirement can be
	// met outside it (another file, an earlier PR, another repo, the
	// pipeline).
	ComplianceNotMet ComplianceDrop = "not_met"
	// ComplianceNoQuote means no file or no line was quoted.
	ComplianceNoQuote ComplianceDrop = "no_quote"
	// ComplianceQuoteTooShort means the quote has too few letters and
	// digits to tie to anything: a bare brace, a diff marker, a blank line.
	ComplianceQuoteTooShort ComplianceDrop = "quote_too_short"
	// ComplianceUnknownFile means the quoted file is not one the prompt
	// showed.
	ComplianceUnknownFile ComplianceDrop = "unknown_file"
	// ComplianceQuoteRemoved means the quoted line is one the PR removed,
	// which shows nothing implemented.
	ComplianceQuoteRemoved ComplianceDrop = "quote_removed"
	// ComplianceQuoteNotFound means the quoted line is nowhere in the file.
	ComplianceQuoteNotFound ComplianceDrop = "quote_not_found"
)

// minQuoteAlnum is the fewest letters and digits a quote must carry. A
// blank line, `}` or a lone `+` matches some line of nearly every file
// (the blank one at its end at least), so it proves any requirement. A
// generic line above the floor (`return nil`) still passes: the rule checks
// that the line is there, not that it implements the requirement.
const minQuoteAlnum = 6

// ComplianceCheck is ConfirmCompliance's verdict on one requirement: kept
// when Drop is empty.
type ComplianceCheck struct {
	Requirement ComplianceRequirement
	Drop        ComplianceDrop
}

// ConfirmCompliance keeps the requirements whose quoted line proves them
// implemented, in input order, and says for every one what became of it.
//
// Only a met requirement can be proven, and only by a line of the named file
// as the prompt showed it: an added or unchanged diff line, or a line of the
// full content. A removed line proves nothing, and a not-met requirement is
// never kept: an absence is not provable from a delta, and that verdict was
// the one that misled. Blunt on purpose, like ValidateFindings: hashes and a
// fixed rule, never a second model. What it cannot check is that the line
// really implements the requirement; it checks that the line is there.
//
// history is the cumulative PR diff the prompt showed, by file, on an
// incremental review (nil on a full one, whose diff is the whole PR). The
// summary is the PR's on every push, so an added or context line an earlier
// push shows there proves its requirement on a push that never touched that
// file. A line outside every hunk of such a file is not shown on that push
// and proves nothing then (known limit).
func ConfirmCompliance(reqs []ComplianceRequirement, idx, history diff.AnchorIndex) (kept []ComplianceRequirement, checks []ComplianceCheck) {
	kept = []ComplianceRequirement{}
	for _, r := range reqs {
		drop := precheck(r)
		if drop == "" {
			drop = confirmIn(&r, idx)
			if drop != "" && history != nil {
				if alt := r; confirmIn(&alt, history) == "" {
					r, drop = alt, ""
				}
			}
		}
		checks = append(checks, ComplianceCheck{Requirement: r, Drop: drop})
		if drop == "" {
			kept = append(kept, r)
		}
	}
	return kept, checks
}

// precheck is what no index can rescue: unmet, no quote, a quote too thin.
func precheck(r ComplianceRequirement) ComplianceDrop {
	switch {
	case !r.Met:
		return ComplianceNotMet
	case strings.TrimSpace(r.File) == "" || strings.TrimSpace(r.Quote) == "":
		return ComplianceNoQuote
	case alnum(r.Quote) < minQuoteAlnum:
		return ComplianceQuoteTooShort
	}
	return ""
}

// confirmIn checks the quote against one index, normalizing the file to the
// path the index knows it by.
func confirmIn(r *ComplianceRequirement, idx diff.AnchorIndex) ComplianceDrop {
	file := ShownPath(r.File, idx)
	fa, ok := idx[file]
	if !ok {
		return ComplianceUnknownFile
	}
	r.File = file
	switch h, where := matchEvidence(r.Quote, fa); where {
	case onNewSide, inContentOnly:
		return ""
	case onRemovedSide:
		// The same line can be removed in one place and still stand in
		// another: the file as it is now decides.
		if fa.InContent(h) {
			return ""
		}
		return ComplianceQuoteRemoved
	default:
		return ComplianceQuoteNotFound
	}
}

func alnum(s string) int {
	n := 0
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			n++
		}
	}
	return n
}
