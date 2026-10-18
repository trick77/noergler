package inference

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// FixedIDPrefix marks a previously-posted finding's id in the prompt: F12 is
// store row 12.
const FixedIDPrefix = "F"

// FixedClaim is one `fixed` entry from the model: an earlier finding it says
// the focused diff fixes, with the diff lines that do it.
type FixedClaim struct {
	ID       string
	Evidence []string
}

// parseFixedClaim reads one `fixed` item. id must be a string; evidence is
// read like a finding's, and an absent one is left for ValidateFixed to drop
// with its reason.
func parseFixedClaim(raw json.RawMessage) (FixedClaim, bool) {
	var probe struct {
		ID       *string          `json:"id"`
		Evidence *json.RawMessage `json:"evidence"`
	}
	if json.Unmarshal(raw, &probe) != nil || probe.ID == nil {
		return FixedClaim{}, false
	}
	out := FixedClaim{ID: strings.TrimSpace(*probe.ID)}
	if probe.Evidence != nil {
		out.Evidence = coerceEvidence(*probe.Evidence)
	}
	return out, true
}

// FixedDropReason says why ValidateFixed refused a claim. Log vocabulary; do
// not reword.
type FixedDropReason string

const (
	// FixedUnknownID means the id names no finding the prompt listed.
	FixedUnknownID FixedDropReason = "unknown_id"
	// FixedDuplicate means the id was claimed fixed twice; the first stands.
	FixedDuplicate FixedDropReason = "duplicate"
	// FixedFileNotInDiff means the finding's file is not one this run's
	// focused diff changed: nothing shown could have fixed it.
	FixedFileNotInDiff FixedDropReason = "file_not_in_diff"
	// FixedNoEvidence means no non-blank evidence line was quoted.
	FixedNoEvidence FixedDropReason = "no_evidence"
	// FixedEvidenceNotFound means a quoted line is nowhere in that file.
	FixedEvidenceNotFound FixedDropReason = "evidence_not_found"
	// FixedEvidenceUnchanged means every quoted line is context or full-file:
	// none was added or removed, so the diff did not touch what it quotes.
	FixedEvidenceUnchanged FixedDropReason = "evidence_unchanged"
)

// FixedFinding is a claim ValidateFixed accepted.
type FixedFinding struct {
	Finding PostedFinding
}

// DroppedFixed is a claim ValidateFixed refused, with the reason.
type DroppedFixed struct {
	Claim  FixedClaim
	Reason FixedDropReason
}

// FixedValidation is ValidateFixed's verdict on one review's claims.
type FixedValidation struct {
	Kept    []FixedFinding
	Dropped []DroppedFixed
}

// ValidateFixed checks every `fixed` claim against the findings the prompt
// listed and the lines its diffs showed.
//
// Blunt on purpose, like ValidateFindings: ids, file names and line hashes,
// never a second model. A wrong "fixed" resolves a thread on a live bug, so
// the bar is a changed line, not a shown one:
//
//   - the id must be one the prompt listed, and claimed once;
//   - the finding's file must be one the focused diff changed;
//   - every non-blank evidence line must be in that file (a note or an
//     elision is skipped, as for findings), and at least one of them an
//     added or removed line.
func ValidateFixed(claims []FixedClaim, sent []PostedFinding, idx diff.AnchorIndex) FixedValidation {
	byID := make(map[int64]PostedFinding, len(sent))
	for _, f := range sent {
		if f.ID > 0 {
			byID[f.ID] = f
		}
	}
	seen := map[int64]bool{}
	var v FixedValidation
	for _, c := range claims {
		f, reason := validateFixedOne(c, byID, seen, idx)
		if reason != "" {
			v.Dropped = append(v.Dropped, DroppedFixed{Claim: c, Reason: reason})
			continue
		}
		seen[f.ID] = true
		v.Kept = append(v.Kept, FixedFinding{Finding: f})
	}
	return v
}

func validateFixedOne(c FixedClaim, byID map[int64]PostedFinding, seen map[int64]bool, idx diff.AnchorIndex) (PostedFinding, FixedDropReason) {
	id, err := strconv.ParseInt(strings.TrimPrefix(c.ID, FixedIDPrefix), 10, 64)
	if err != nil {
		return PostedFinding{}, FixedUnknownID
	}
	f, ok := byID[id]
	if !ok {
		return PostedFinding{}, FixedUnknownID
	}
	if seen[id] {
		return PostedFinding{}, FixedDuplicate
	}
	path := ShownPath(f.FilePath, idx)
	fa, ok := idx[path]
	if !ok {
		return PostedFinding{}, FixedFileNotInDiff
	}
	quoted, changed := 0, false
	for _, e := range c.Evidence {
		if strings.TrimSpace(e) == "" {
			continue
		}
		h, where := matchEvidence(e, fa)
		switch {
		case where == notFound && isAnnotation(e):
			continue
		case where == notFound:
			return PostedFinding{}, FixedEvidenceNotFound
		case where == onRemovedSide:
			changed = true
		case where == onNewSide && addedLine(h, fa.Shown):
			changed = true
		}
		quoted++
	}
	if quoted == 0 {
		return PostedFinding{}, FixedNoEvidence
	}
	if !changed {
		return PostedFinding{}, FixedEvidenceUnchanged
	}
	return f, ""
}

// addedLine reports whether a shown line with hash h is an added one, not
// context.
func addedLine(h uint64, shown map[int]diff.ShownLine) bool {
	for _, s := range shown {
		if s.Hash == h && s.Added {
			return true
		}
	}
	return false
}
