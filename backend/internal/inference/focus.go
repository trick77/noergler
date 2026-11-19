package inference

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// MaxFocus caps review_focus: a short list is the point, a long one is
// the diff again.
const MaxFocus = 3

// FocusItem is one place the model says a person should read first: a
// behavioural change, never a finding. File and Line are checked against
// the diff the prompt showed (ValidateFocus); Why is one sentence.
type FocusItem struct {
	File string
	Line int
	Why  string
}

// parseFocus reads one review_focus entry. A missing file, a fractional or
// absent line, or a blank reason drops the entry, never the review.
func parseFocus(raw json.RawMessage) (FocusItem, bool) {
	var probe struct {
		File *string          `json:"file"`
		Line *json.RawMessage `json:"line"`
		Why  *string          `json:"why"`
	}
	if json.Unmarshal(raw, &probe) != nil || probe.File == nil || probe.Line == nil || probe.Why == nil {
		return FocusItem{}, false
	}
	line, ok := coerceInt(*probe.Line)
	why := strings.Join(strings.Fields(*probe.Why), " ")
	if !ok || strings.TrimSpace(*probe.File) == "" || why == "" {
		return FocusItem{}, false
	}
	return FocusItem{File: strings.TrimSpace(*probe.File), Line: line, Why: why}, true
}

// FocusDropReason says why ValidateFocus refused an item. Log vocabulary.
type FocusDropReason string

// The reasons ValidateFocus drops an item: a file the prompt did not carry,
// a line its diff does not show, a place listed twice, one past MaxFocus.
const (
	FocusUnknownFile FocusDropReason = "unknown_file"
	FocusNotInDiff   FocusDropReason = "not_in_diff"
	FocusRepeat      FocusDropReason = "repeat"
	FocusOverCap     FocusDropReason = "over_cap"
)

// DroppedFocus is an item ValidateFocus refused, with the reason.
type DroppedFocus struct {
	Item   FocusItem
	Reason FocusDropReason
}

// ValidateFocus keeps the items that point at a line a diff of the prompt
// shows, at most MaxFocus, the first of each place: the focused diff, or on
// an incremental review also the cumulative PR diff (history, nil on a full
// one), since the focus is the summary's and the summary is the PR's.
// Blunt like every check here: a path and a line number, never a judgement
// of the sentence.
func ValidateFocus(items []FocusItem, idx, history diff.AnchorIndex) (kept []FocusItem, dropped []DroppedFocus) {
	seen := map[string]bool{}
	for _, it := range items {
		if p := ShownPath(it.File, idx); idx[p].Shown != nil {
			it.File = p
		} else {
			it.File = ShownPath(it.File, history)
		}
		fa, inFocused := idx[it.File]
		ha, inHistory := history[it.File]
		switch {
		case !inFocused && !inHistory:
			dropped = append(dropped, DroppedFocus{it, FocusUnknownFile})
			continue
		case (!inFocused || !inDiff(fa, it.Line)) && (!inHistory || !inDiff(ha, it.Line)):
			dropped = append(dropped, DroppedFocus{it, FocusNotInDiff})
			continue
		}
		key := it.File + ":" + strconv.Itoa(it.Line)
		switch {
		case seen[key]:
			dropped = append(dropped, DroppedFocus{it, FocusRepeat})
		case len(kept) == MaxFocus:
			dropped = append(dropped, DroppedFocus{it, FocusOverCap})
		default:
			seen[key] = true
			kept = append(kept, it)
		}
	}
	return kept, dropped
}

// inDiff reports whether line is one the file's diff shows.
func inDiff(fa diff.FileAnchors, line int) bool {
	_, ok := fa.Shown[line]
	return ok
}
