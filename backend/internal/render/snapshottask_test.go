package render

import (
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

func TestSnapshotTask(t *testing.T) {
	line := SnapshotLine{Path: "pom.xml", Line: 12, Text: "    <version>1.1-SNAPSHOT</version>"}
	en := SnapshotTask(lang.EN, "master", []SnapshotLine{line})
	for _, want := range []string{"**SNAPSHOT version into `master`**", "- `pom.xml:12` `<version>1.1-SNAPSHOT</version>`",
		"resolve the task yourself"} {
		if !strings.Contains(en, want) {
			t.Errorf("EN lacks %q:\n%s", want, en)
		}
	}
	de := SnapshotTask(lang.DE, "main", []SnapshotLine{line})
	if !strings.Contains(de, "darf nicht auf `main`") || strings.Contains(de, "ß") {
		t.Errorf("DE:\n%s", de)
	}

	var many []SnapshotLine
	for i := range SnapshotTaskMaxListed + 3 {
		many = append(many, SnapshotLine{Path: "pom.xml", Line: i + 1, Text: fmt.Sprintf("v%d-SNAPSHOT", i)})
	}
	got := SnapshotTask(lang.EN, "master", many)
	if strings.Count(got, "\n- `pom.xml:") != SnapshotTaskMaxListed || !strings.Contains(got, "- and 3 more") {
		t.Errorf("capped list:\n%s", got)
	}
}
