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

// caseImages is a case's screenshots cut as production cuts them, once per
// load rather than once per draw: ids maps a sent id to its path, pixels is
// the images' token estimate (the labels are counted at assembly).
type caseImages struct {
	images []inference.VisualImage
	ids    map[string]string
	pixels int
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

// loadImages reads a case's screenshots and cuts each pair with
// visual.Compare and inference.NewVisualImage, as review.prepareVisual
// does. A pair that does not differ is a broken case, not a skip.
func loadImages(entries []corpusScreenshot) (caseImages, error) {
	out := caseImages{ids: map[string]string{}}
	noCount := func(string) int { return 0 }
	for _, e := range entries {
		var before []byte
		var err error
		if e.Before != "" {
			if before, err = screenshotFS.ReadFile("corpus/screenshots/" + e.Before); err != nil {
				return caseImages{}, err
			}
		}
		after, err := screenshotFS.ReadFile("corpus/screenshots/" + e.After)
		if err != nil {
			return caseImages{}, err
		}
		res, err := visual.Compare(before, after)
		if err != nil {
			return caseImages{}, fmt.Errorf("screenshot %s: %w", e.Path, err)
		}
		if res.Identical {
			return caseImages{}, fmt.Errorf("screenshot %s: before and after have the same pixels", e.Path)
		}
		img, _ := inference.NewVisualImage(len(out.images), e.Path, res, noCount)
		out.images = append(out.images, img)
		out.ids[img.ID] = e.Path
		out.pixels += res.Tokens
	}
	return out, nil
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
