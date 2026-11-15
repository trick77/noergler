package render

import (
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/lang"
)

// SnapshotTaskMaxListed caps the lines a SNAPSHOT task names; the rest are
// counted.
const SnapshotTaskMaxListed = 10

// SnapshotLine is one line a SNAPSHOT task names.
type SnapshotLine struct {
	Path string
	Line int
	Text string
}

// SnapshotTask is the Bitbucket task on a PR into the default branch that
// adds a SNAPSHOT build (user, 2026-11-15). A task, not a finding: the
// author resolves it on purpose, and where the repo's "incomplete tasks"
// merge check is on it blocks the merge until then.
func SnapshotTask(l lang.Lang, branch string, lines []SnapshotLine) string {
	var b strings.Builder
	b.WriteString(l.Pick("**SNAPSHOT-Version auf ", "**SNAPSHOT version into ") + inlineCode(branch) + "**\n\n")
	b.WriteString(l.Pick("Dieser PR fügt eine SNAPSHOT-Version hinzu, und die darf nicht auf ",
		"This PR adds a SNAPSHOT version, which must not reach ") + inlineCode(branch) + ":\n\n")
	for i, s := range lines {
		if i == SnapshotTaskMaxListed {
			fmt.Fprintf(&b, l.Pick("- und %d weitere\n", "- and %d more\n"), len(lines)-i)
			break
		}
		fmt.Fprintf(&b, "- %s %s\n", inlineCode(fmt.Sprintf("%s:%d", s.Path, s.Line)), inlineCode(strings.TrimSpace(s.Text)))
	}
	b.WriteString("\n" + l.Pick(
		"Stell auf ein Release um: Ist keine solche Zeile mehr da, erledige ich diesen Task selbst. "+
			"Soll der PR trotzdem so gemergt werden, erledige den Task von Hand.",
		"Switch to a released version: once no such line is left, I resolve this task myself. "+
			"To merge the PR as it is anyway, resolve the task yourself."))
	return b.String()
}
