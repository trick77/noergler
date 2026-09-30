package inference

import (
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// DropReason says why ValidateFindings refused a finding. The values are log
// and eval-report vocabulary; do not reword them.
type DropReason string

const (
	// DropUnknownFile means the finding names a file the prompt did not show.
	DropUnknownFile DropReason = "unknown_file"
	// DropNoEvidence means no non-blank evidence line was quoted.
	DropNoEvidence DropReason = "no_evidence"
	// DropEvidenceNotFound means a quoted line is in neither the file's diff
	// nor its full content. The model quoted code that is not there.
	DropEvidenceNotFound DropReason = "evidence_not_found"
	// DropAnchorMismatch means the evidence exists, but not near `line`, and
	// the first evidence line matches several shown lines, so there is no
	// single line to move the finding to.
	DropAnchorMismatch DropReason = "anchor_mismatch"
	// DropNoopSuggestion means the suggested fix is the code already there,
	// or only reorders `key: value` entries. The finding argues a change that
	// changes nothing.
	DropNoopSuggestion DropReason = "noop_suggestion"
)

// DroppedFinding is a finding ValidateFindings refused, with the reason.
type DroppedFinding struct {
	Finding ReviewFinding
	Reason  DropReason
}

// anchorSlack is how far `line` may sit from a quoted evidence line and still
// count as anchored on it. A finding about a block is anchored on its opening
// line about as often as on the line that breaks; both are honest.
const anchorSlack = 2

// ValidateFindings checks every finding against the lines the prompt showed.
//
// Blunt on purpose, like eval scoring: file, line hashes and a fixed rule
// set, never a second model. Kept findings come back in input order; a
// re-anchored one carries its corrected line.
//
//   - the file must be one the prompt showed;
//   - every non-blank evidence line must be a shown line of that file;
//   - an evidence line must sit within anchorSlack of `line`, else the
//     finding moves to the first evidence line when that line is unique in
//     the file, else it is dropped;
//   - a suggestion that is the code already at `line`, or only reorders
//     `key: value` entries there, is dropped.
func ValidateFindings(findings []ReviewFinding, idx diff.AnchorIndex) Validation {
	var v Validation
	for _, f := range findings {
		reason, line := validateOne(f, idx)
		if reason != "" {
			v.Dropped = append(v.Dropped, DroppedFinding{Finding: f, Reason: reason})
			continue
		}
		if line != f.Line {
			v.Reanchored = append(v.Reanchored, Reanchor{File: f.File, From: f.Line, To: line})
			f.Line = line
		}
		v.Kept = append(v.Kept, f)
	}
	return v
}

// AdjustVerdict lowers the model's verdict to what the kept findings support,
// when validation dropped any. The prompt ties the decision to the findings
// (request_changes needs an `issue`, approve_with_followups a `suggestion`);
// a verdict argued from a dropped finding would otherwise say "request
// changes" beside zero posted comments. Only ever lowers: a drop cannot make
// a review stricter. The rationale is replaced, since the model's names the
// finding that is gone.
func AdjustVerdict(s ReviewSummary, kept []ReviewFinding, dropped int) ReviewSummary {
	if dropped == 0 {
		return s
	}
	supported := "approve"
	for _, f := range kept {
		if f.Severity == "issue" {
			supported = "request_changes"
			break
		}
		supported = "approve_with_followups"
	}
	if verdictRank(supported) >= verdictRank(s.VerdictDecision) {
		return s
	}
	s.VerdictDecision = supported
	s.VerdictRationale = fmt.Sprintf("Verdict lowered: %d finding(s) whose evidence the diff did not bear out were withheld.", dropped)
	return s
}

// verdictRank orders the decisions by strictness; an unknown one ranks as
// approve, the parser's default.
func verdictRank(d string) int {
	return max(0, slices.Index(VerdictDecisions, d))
}

// Validation is ValidateFindings' verdict on one review's findings.
type Validation struct {
	Kept    []ReviewFinding
	Dropped []DroppedFinding
	// Reanchored lists the kept findings whose line was moved to their
	// evidence. Each one is a line the model got wrong.
	Reanchored []Reanchor
}

// Reanchor is one finding moved from the line the model cited to the line
// its evidence is on.
type Reanchor struct {
	File     string
	From, To int
}

func validateOne(f ReviewFinding, idx diff.AnchorIndex) (DropReason, int) {
	fa, ok := idx[f.File]
	if !ok {
		// Bitbucket strips an a/ or b/ prefix when anchoring; so does this.
		fa, ok = idx[stripSidePrefix(f.File)]
	}
	if !ok {
		return DropUnknownFile, 0
	}

	// anchors are the evidence lines the diff shows on its new side. A
	// quoted removed line, or a full-file line outside the diff, is valid
	// evidence, but nothing can be posted on it.
	var anchors []uint64
	quoted := 0
	for _, e := range f.Evidence {
		if strings.TrimSpace(e) == "" {
			continue
		}
		h, onNewSide, found := matchEvidence(e, fa)
		if !found {
			return DropEvidenceNotFound, 0
		}
		quoted++
		if onNewSide {
			anchors = append(anchors, h)
		}
	}
	if quoted == 0 {
		return DropNoEvidence, 0
	}

	line := f.Line
	switch {
	case len(anchors) == 0:
		// Only removed lines quoted: the anchor cannot be checked against
		// them, so it must at least be a line the diff showed.
		if _, shown := fa.Shown[line]; !shown {
			return DropAnchorMismatch, 0
		}
	default:
		_, shown := fa.Shown[line]
		if shown && nearAny(line, anchors, fa.Shown) {
			break
		}
		// Near its evidence but on a line the diff did not show (past a
		// hunk's end, or past EOF): Bitbucket refuses both anchors there,
		// so the finding moves to the closest evidence line.
		if n, ok := nearestAnchor(line, anchors, fa.Shown); ok {
			line = n
			break
		}
		at := linesWithHash(anchors[0], fa.Shown)
		if len(at) != 1 {
			return DropAnchorMismatch, 0
		}
		line = at[0]
	}

	if f.Suggestion != nil && isNoopSuggestion(*f.Suggestion, line, fa.Shown, anchors) {
		return DropNoopSuggestion, 0
	}
	return "", line
}

// gutterRE is the line-number gutter NumberDiff writes, plus the diff marker
// after it, in case the model copied the whole rendered line. The removed-line
// form has a blank gutter, so it is the marker alone. ASCII digits on
// purpose: the gutter is ours, never the file's text.
var gutterRE = regexp.MustCompile(`^\s*[0-9]+ [ +]`)

// matchEvidence finds a quoted line among the shown, removed or full-file
// ones, and says whether it is on the diff's new side. The quote as given is
// tried first, then with a copied gutter or a leading diff marker taken off:
// code can itself start with `+`, `-` or a digit, so a stripped form is only
// a fallback.
func matchEvidence(quote string, fa diff.FileAnchors) (hash uint64, onNewSide, found bool) {
	candidates := []string{quote}
	if loc := gutterRE.FindStringIndex(quote); loc != nil {
		candidates = append(candidates, quote[loc[1]:])
	}
	if t := strings.TrimLeft(quote, " \t"); strings.HasPrefix(t, "+") || strings.HasPrefix(t, "-") {
		candidates = append(candidates, t[1:])
	}
	// Candidate first, then where it is found: the quote as written wins
	// anywhere before a stripped form is tried anywhere. Stripping first
	// would let a removed YAML line `- run: x` match the unrelated shown
	// line ` run: x`.
	for _, c := range candidates {
		h := diff.HashLine(c)
		if len(linesWithHash(h, fa.Shown)) > 0 {
			return h, true, true
		}
		if fa.Removed[h] || fa.InContent(h) {
			return h, false, true
		}
	}
	return 0, false, false
}

// nearestAnchor is the shown line within anchorSlack of line that carries an
// evidence hash, closest first, the lower line on a tie.
func nearestAnchor(line int, hashes []uint64, shown map[int]diff.ShownLine) (int, bool) {
	for d := 0; d <= anchorSlack; d++ {
		for _, n := range []int{line - d, line + d} {
			if s, ok := shown[n]; ok && slices.Contains(hashes, s.Hash) {
				return n, true
			}
		}
	}
	return 0, false
}

func nearAny(line int, hashes []uint64, shown map[int]diff.ShownLine) bool {
	for n := line - anchorSlack; n <= line+anchorSlack; n++ {
		s, ok := shown[n]
		if !ok {
			continue
		}
		for _, h := range hashes {
			if s.Hash == h {
				return true
			}
		}
	}
	return false
}

func linesWithHash(h uint64, shown map[int]diff.ShownLine) []int {
	var at []int
	for n, s := range shown {
		if s.Hash == h {
			at = append(at, n)
		}
	}
	return at
}

// keyValueRE is one comma-terminated `key: value` entry of an object, map,
// dict or struct literal, whose order does not change what the code does.
// The trailing comma is what makes it an entry: a Python annotated
// assignment, a dataclass field or a CSS declaration also reads `key: value`,
// and in each of those order matters. A value is required and `:=` is
// excluded. `[\pL\pN_.$-]`, not `\w`: RE2's `\w` is ASCII. Group 1 is the key.
var keyValueRE = regexp.MustCompile(`^['"]?([\pL\pN_.$-]+)['"]?[ \t]*:[ \t]*[^=\s].*,$`)

// isNoopSuggestion reports whether applying the suggestion at line would
// leave the code as it is: the same lines in the same order, or the same
// lines where every one that moved is a literal entry with its own key.
// Reordering statements can be a real fix, so a moved statement keeps the
// finding.
//
// Only lines the diff showed can be compared. If any line the suggestion
// would replace was not shown, the suggestion is not judged. Nor is one that
// stops short of an evidence line right after the lines it replaces: the
// finding quoted that line, so the suggestion deletes it.
func isNoopSuggestion(suggestion string, line int, shown map[int]diff.ShownLine, evidence []uint64) bool {
	var sug []string
	for _, l := range strings.Split(suggestion, "\n") {
		if strings.TrimSpace(l) != "" {
			sug = append(sug, l)
		}
	}
	if len(sug) == 0 {
		return false
	}
	// The existing lines the suggestion would replace, skipping blank ones
	// the way the suggestion's blank lines were skipped. Compared by Exact,
	// not Hash: a suggestion that only re-indents a line changes the code in
	// Python or YAML, so it is never a no-op. A model that re-indents a
	// genuine no-op keeps its finding; that is the cheaper mistake.
	blank := diff.HashLine("")
	var have []uint64
	n := line
	for ; len(have) < len(sug); n++ {
		s, ok := shown[n]
		if !ok {
			return false
		}
		if s.Hash != blank {
			have = append(have, s.Exact)
		}
	}
	if next, ok := shown[n]; ok && slices.Contains(evidence, next.Hash) {
		return false
	}

	counts := map[uint64]int{}
	for _, h := range have {
		counts[h]++
	}
	movedKeys := map[string]bool{}
	for i, l := range sug {
		h := diff.HashExact(l)
		if counts[h] == 0 {
			return false
		}
		counts[h]--
		if h == have[i] {
			continue
		}
		m := keyValueRE.FindStringSubmatch(strings.TrimSpace(l))
		if m == nil || movedKeys[m[1]] {
			return false
		}
		movedKeys[m[1]] = true
	}
	return true
}

func stripSidePrefix(p string) string {
	if strings.HasPrefix(p, "a/") || strings.HasPrefix(p, "b/") {
		return p[2:]
	}
	return p
}
