package inference

import (
	"encoding/json"
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// PreExistingBug is a real bug the model noticed in code the PR did not
// change. It is never posted to the PR (decided): the PR's author did not
// write it. The run page lists it with its explanation and a link to the
// line, so somebody can pick it up.
type PreExistingBug struct {
	File        string
	Line        int
	Headline    string
	Explanation string
	Evidence    []string
}

// Limits on what a pre-existing bug may carry onto the page: a few, short.
// The text is the model's, shown on an unauthenticated dashboard.
const (
	maxPreExisting         = 5
	maxPreExistingHeadline = 200
	maxPreExistingProse    = 1000
)

// parsePreExisting reads one `pre_existing` item, reporting whether it is
// usable at all: a file, a line, a headline and an explanation.
func parsePreExisting(item json.RawMessage) (PreExistingBug, bool) {
	var probe struct {
		File        *string          `json:"file"`
		Line        *json.RawMessage `json:"line"`
		Headline    *string          `json:"headline"`
		Explanation *string          `json:"explanation"`
		Evidence    *json.RawMessage `json:"evidence"`
	}
	if err := json.Unmarshal(item, &probe); err != nil {
		return PreExistingBug{}, false
	}
	if probe.File == nil || probe.Line == nil || probe.Headline == nil || probe.Explanation == nil {
		return PreExistingBug{}, false
	}
	line, ok := coerceInt(*probe.Line)
	if !ok {
		return PreExistingBug{}, false
	}
	b := PreExistingBug{
		File:        *probe.File,
		Line:        line,
		Headline:    truncateRunes(strings.TrimSpace(*probe.Headline), maxPreExistingHeadline),
		Explanation: truncateRunes(strings.TrimSpace(*probe.Explanation), maxPreExistingProse),
	}
	if b.Headline == "" || b.Explanation == "" {
		return PreExistingBug{}, false
	}
	if probe.Evidence != nil {
		b.Evidence = coerceEvidence(*probe.Evidence)
	}
	return b, true
}

// DroppedPreExisting is a pre-existing claim validation refused, and why.
type DroppedPreExisting struct {
	Bug    PreExistingBug
	Reason string
}

// ValidatePreExisting keeps the claims whose evidence is real: in the file
// the prompt showed (its diff, its context or its full content), and none of
// it on a line the PR added, which would make it the PR's own bug and a
// finding, not this. There is no in-diff rule: unchanged code is the point.
//
// The line is moved to the evidence when the evidence is a shown line and
// the cited one is not near it. Evidence found only in the full content has
// no line number in the index (hashes only), so the cited line stands: the
// page links to it, and the evidence at least proves the code exists.
//
// Blunt rules, like ValidateFindings, never a judge model. At most
// maxPreExisting are kept, in the model's order.
func ValidatePreExisting(bugs []PreExistingBug, idx diff.AnchorIndex) (kept []PreExistingBug, dropped []DroppedPreExisting) {
	for _, b := range bugs {
		b.File = ShownPath(b.File, idx)
		reason, line := validatePreExisting(b, idx)
		if reason == "" && len(kept) >= maxPreExisting {
			reason = "over_limit"
		}
		if reason != "" {
			dropped = append(dropped, DroppedPreExisting{Bug: b, Reason: reason})
			continue
		}
		b.Line = line
		kept = append(kept, b)
	}
	return kept, dropped
}

func validatePreExisting(b PreExistingBug, idx diff.AnchorIndex) (string, int) {
	fa, ok := idx[b.File]
	if !ok {
		return string(DropUnknownFile), 0
	}
	var shown []uint64
	quoted := 0
	for _, e := range b.Evidence {
		if strings.TrimSpace(e) == "" || isAnnotation(e) {
			continue
		}
		h, where := matchEvidence(e, fa)
		switch where {
		case notFound:
			return string(DropEvidenceNotFound), 0
		case onNewSide:
			for _, n := range linesWithHash(h, fa.Shown) {
				if fa.Shown[n].Added {
					return "evidence_on_added_line", 0
				}
			}
			shown = append(shown, h)
		case onRemovedSide:
			// The PR deleted it: whatever was wrong there is gone.
			return "evidence_removed_by_pr", 0
		}
		quoted++
	}
	if quoted == 0 {
		return string(DropNoEvidence), 0
	}
	line := b.Line
	if len(shown) > 0 && !nearAny(line, shown, fa.Shown) {
		if at := linesWithHash(shown[0], fa.Shown); len(at) == 1 {
			line = at[0]
		}
	}
	return "", line
}
