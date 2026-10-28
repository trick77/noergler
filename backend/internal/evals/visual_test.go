package evals

import (
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// Every screenshot a case names is embedded and decodes, and the cases that
// carry one are the ones expecting a line.
func TestCorpusScreenshotsLoadAndCompare(t *testing.T) {
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	withShots := 0
	for _, c := range cases {
		if len(c.Screenshots) == 0 {
			if len(c.ExpectedVisual) > 0 {
				t.Errorf("%s expects a screenshot line and carries no screenshot", c.Name)
			}
			continue
		}
		withShots++
		imgs, ids, tokens, err := visualImages(c.Screenshots)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		if len(imgs) != len(c.Screenshots) || len(ids) != len(imgs) || tokens <= 0 {
			t.Errorf("%s: %d image(s) for %d screenshot(s), %d tokens: a corpus pair must differ", c.Name, len(imgs), len(c.Screenshots), tokens)
		}
	}
	if withShots == 0 {
		t.Fatal("no case carries a screenshot")
	}
}

func TestMatchVisual(t *testing.T) {
	expected := []ExpectedVisual{
		{ID: "stacked", Path: "a.png", Keywords: []string{"stack", "column"}},
		{ID: "why", Path: "a.png", Keywords: []string{"login.css"}},
		{ID: "other", Path: "b.png", Keywords: []string{"empty"}},
	}
	ids := map[string]string{"V1": "a.png", "V2": "b.png"}
	got, lines := matchVisual(expected, ids, []inference.VisualChange{
		{ID: "V1", Change: "The fields now Stack in a column."},
		{ID: "V1", Change: "a repeated id is ignored: login.css"},
		{ID: "V9", Change: "an id never sent"},
	})
	if !got[0].Found || got[1].Found || got[1].Why == "" || got[2].Found || got[2].Why != "no visual_changes line for the screenshot" {
		t.Errorf("matches = %+v", got)
	}
	if len(lines) != 1 || lines["a.png"] != "The fields now Stack in a column." {
		t.Errorf("lines = %v", lines)
	}
}
