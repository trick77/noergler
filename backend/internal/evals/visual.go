package evals

import (
	"embed"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/visual"
)

// The screenshots a case changes, rendered from real HTML in a browser:
// drawn rectangles would test the crop, not whether a model can read a UI.
//
//go:embed corpus/screenshots/*.png
var screenshotFS embed.FS

// ExpectedVisual is one screenshot line a reviewer must write. Scored as
// bluntly as a finding: the line for that path mentions one of the keywords.
type ExpectedVisual struct {
	ID       string   `json:"id"`
	Path     string   `json:"path"`
	Keywords []string `json:"keywords"`
	Why      string   `json:"why"`
}

// CaseScreenshot is one changed baseline. Before is nil for an added one.
type CaseScreenshot struct {
	Path          string
	Before, After []byte
}

// VisualMatch records whether one expected screenshot line was written.
type VisualMatch struct {
	Expected ExpectedVisual
	Found    bool
	// Change is the model's line for the path, empty when it wrote none.
	Change string `json:",omitempty"`
	Why    string `json:",omitempty"`
}

// corpusScreenshot is a case file's screenshot entry: files under
// corpus/screenshots/, before empty for an added baseline.
type corpusScreenshot struct {
	Path   string `json:"path"`
	Before string `json:"before"`
	After  string `json:"after"`
}

func loadScreenshots(entries []corpusScreenshot) ([]CaseScreenshot, error) {
	var out []CaseScreenshot
	for _, e := range entries {
		s := CaseScreenshot{Path: e.Path}
		var err error
		if e.Before != "" {
			if s.Before, err = screenshotFS.ReadFile("corpus/screenshots/" + e.Before); err != nil {
				return nil, err
			}
		}
		if s.After, err = screenshotFS.ReadFile("corpus/screenshots/" + e.After); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// visualImages cuts each pair as production does (visual.Compare) and ids it
// in order, as review.prepareVisual does. ids maps a sent id to its path.
func visualImages(shots []CaseScreenshot) (imgs []inference.VisualImage, ids map[string]string, tokens int, err error) {
	ids = map[string]string{}
	for _, s := range shots {
		res, err := visual.Compare(s.Before, s.After)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("screenshot %s: %w", s.Path, err)
		}
		if res.Identical {
			continue
		}
		id := fmt.Sprintf("V%d", len(imgs)+1)
		img := inference.VisualImage{ID: id, Path: s.Path, After: visual.DataURI(res.After), Cropped: res.Cropped}
		if res.Before != nil {
			img.Before = visual.DataURI(res.Before)
		}
		imgs = append(imgs, img)
		ids[id] = s.Path
		tokens += res.Tokens
	}
	return imgs, ids, tokens, nil
}

// matchVisual scores the model's lines against what the case expects, and
// returns the lines by path for the report.
func matchVisual(expected []ExpectedVisual, ids map[string]string, answers []inference.VisualChange) ([]VisualMatch, map[string]string) {
	byPath := map[string]string{}
	for _, a := range answers {
		if p, ok := ids[a.ID]; ok {
			if _, seen := byPath[p]; !seen {
				byPath[p] = a.Change
			}
		}
	}
	var out []VisualMatch
	for _, e := range expected {
		m := VisualMatch{Expected: e, Change: byPath[e.Path]}
		lower := strings.ToLower(m.Change)
		for _, k := range e.Keywords {
			if m.Change != "" && strings.Contains(lower, strings.ToLower(k)) {
				m.Found = true
				break
			}
		}
		switch {
		case m.Change == "":
			m.Why = "no visual_changes line for the screenshot"
		case !m.Found:
			m.Why = "line mentions none of " + strings.Join(e.Keywords, ", ")
		}
		out = append(out, m)
	}
	if len(byPath) == 0 {
		byPath = nil
	}
	return out, byPath
}
