package review

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// pomDiff is a PR diff of pom.xml that adds the given lines after <a>.
func pomDiff(added ...string) string {
	var b strings.Builder
	b.WriteString("diff --git src://pom.xml dst://pom.xml\n--- src://pom.xml\n+++ dst://pom.xml\n@@ -1,1 +1,1 @@\n <a>\n")
	for _, l := range added {
		b.WriteString("+" + l + "\n")
	}
	return b.String()
}

// snapshotHarness is a harness with both task switches on and a PR into
// master, the default branch, that changes pom.xml.
func snapshotHarness(t *testing.T, added ...string) *harness {
	t.Helper()
	h := newHarness(t, func(h *harness) {
		h.bb.changes = []string{"pom.xml"}
		h.bb.changeSet = bitbucket.ChangeSet{Changes: []bitbucket.Change{{Path: "pom.xml", Type: "MODIFY"}}}
		h.bb.prDiff = pomDiff(added...)
		h.bb.files["src1:pom.xml"] = "<a>\n" + strings.Join(added, "\n") + "\n"
	})
	h.r.createTasks = true
	h.r.teamTasks.Store(true)
	return h
}

func (h *harness) push(t *testing.T) {
	t.Helper()
	h.llm.review = okResultWith()
	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)
}

func (h *harness) taskState(id int) string { return h.bb.comments[id].State }

// taskEdit is the last edit of comment id, "" for none.
func (h *harness) taskEdit(id int) string {
	var out string
	for _, u := range h.bb.Updates {
		if u.Line == id {
			out = u.Text
		}
	}
	return out
}

const snap1, snap2 = "<version>1.1-SNAPSHOT</version>", "<version>2.0-SNAPSHOT</version>"

func TestSnapshotTaskPostedOnce(t *testing.T) {
	h := snapshotHarness(t, snap1)

	h.push(t)
	if len(h.bb.Tasks) != 1 || !strings.Contains(h.bb.Tasks[0].Text, "`pom.xml:2`") ||
		!strings.Contains(h.bb.Tasks[0].Text, "must not reach `master`") {
		t.Fatalf("tasks = %+v", h.bb.Tasks)
	}
	if st := h.st.snapshotTasks; len(st) != 1 || !st[0].Open || len(st[0].Keys) != 1 {
		t.Fatalf("stored = %+v", st)
	}
	id := h.st.snapshotTasks[0].ID

	h.push(t)
	if len(h.bb.Tasks) != 1 || h.taskEdit(id) != "" {
		t.Fatalf("second push: tasks = %d, edit = %q", len(h.bb.Tasks), h.taskEdit(id))
	}
}

// The author resolved the task: the same lines never get another; a new
// SNAPSHOT line gets a task of its own that names only it.
func TestSnapshotTaskResolvedByTheAuthor(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.push(t)
	h.bb.comments[h.st.snapshotTasks[0].ID].State = "RESOLVED"

	h.push(t)
	if st := h.st.snapshotTasks; len(h.bb.Tasks) != 1 || len(st) != 1 || st[0].Open || len(st[0].Keys) != 1 {
		t.Fatalf("tasks = %d, stored = %+v", len(h.bb.Tasks), st)
	}

	h.bb.prDiff = pomDiff(snap1, snap2)
	h.push(t)
	if len(h.bb.Tasks) != 2 || strings.Contains(h.bb.Tasks[1].Text, "1.1-SNAPSHOT") ||
		!strings.Contains(h.bb.Tasks[1].Text, "2.0-SNAPSHOT") {
		t.Fatalf("tasks = %+v", h.bb.Tasks)
	}
}

// A new SNAPSHOT line while a task is open gets a task of its own; the open
// one keeps naming only its lines.
func TestSnapshotTaskNewLineNewTask(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.push(t)
	first := h.st.snapshotTasks[0].ID

	h.bb.prDiff = pomDiff(snap1, snap2)
	h.push(t)
	if len(h.bb.Tasks) != 2 || strings.Contains(h.bb.Tasks[1].Text, "1.1-SNAPSHOT") || h.taskEdit(first) != "" {
		t.Fatalf("tasks = %+v, edit = %q", h.bb.Tasks, h.taskEdit(first))
	}
}

// One of a task's lines released: its text is edited to the rest; all of
// them gone: noergler resolves it; added again: a new task.
func TestSnapshotTaskFollowsItsLines(t *testing.T) {
	h := snapshotHarness(t, snap1, snap2)
	h.push(t)
	id := h.st.snapshotTasks[0].ID

	h.bb.prDiff = pomDiff(snap1, "<version>2.0</version>")
	h.push(t)
	if e := h.taskEdit(id); !strings.Contains(e, "1.1-SNAPSHOT") || strings.Contains(e, "2.0-SNAPSHOT") {
		t.Fatalf("edit = %q", e)
	}
	if st := h.st.snapshotTasks; len(st) != 1 || len(st[0].Keys) != 1 {
		t.Fatalf("stored = %+v", st)
	}

	h.bb.prDiff = pomDiff("<version>1.1</version>")
	h.push(t)
	if len(h.bb.ResolvedTasks) != 1 || h.bb.ResolvedTasks[0] != id || h.taskState(id) != "RESOLVED" || len(h.st.snapshotTasks) != 0 {
		t.Fatalf("resolved = %v, stored = %+v", h.bb.ResolvedTasks, h.st.snapshotTasks)
	}

	h.bb.prDiff = pomDiff(snap1)
	h.push(t)
	if len(h.bb.Tasks) != 2 {
		t.Fatalf("re-added: tasks = %d", len(h.bb.Tasks))
	}
}

// Two dependencies on one version line: the second, added after the author
// resolved the first's task, is new.
func TestSnapshotTaskSameVersionLineTwice(t *testing.T) {
	h := snapshotHarness(t, "<artifactId>a</artifactId>", snap2)
	h.push(t)
	h.bb.comments[h.st.snapshotTasks[0].ID].State = "RESOLVED"

	h.bb.prDiff = pomDiff("<artifactId>a</artifactId>", snap2, "<artifactId>b</artifactId>", snap2)
	h.push(t)
	if len(h.bb.Tasks) != 2 || !strings.Contains(h.bb.Tasks[1].Text, "`pom.xml:5`") {
		t.Fatalf("tasks = %+v", h.bb.Tasks)
	}
}

// A PR into another branch gets none; one retargeted there has its task
// resolved.
func TestSnapshotTaskOnlyIntoTheDefaultBranch(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.bb.defaultBranch = "refs/heads/main"
	h.push(t)
	if len(h.bb.Tasks) != 0 {
		t.Fatalf("tasks = %+v", h.bb.Tasks)
	}

	h = snapshotHarness(t, snap1)
	h.push(t)
	h.bb.defaultBranch = "refs/heads/main"
	h.r.defaultBranches = branchCache{}
	h.push(t)
	if len(h.bb.ResolvedTasks) != 1 {
		t.Fatalf("resolved = %v", h.bb.ResolvedTasks)
	}
}

// A retarget onto the default branch runs the task alone, no review.
func TestSnapshotTaskOnRetarget(t *testing.T) {
	h := snapshotHarness(t, snap1)
	p := prPayload(webhook.EventModified)
	p.PreviousTarget = &webhook.Ref{ID: "refs/heads/develop", DisplayID: "develop"}
	h.r.HandleRetargeted(context.Background(), p)
	if len(h.bb.Tasks) != 1 || len(h.llm.Reviews) != 0 {
		t.Fatalf("tasks = %d, reviews = %d", len(h.bb.Tasks), len(h.llm.Reviews))
	}
}

// Off is off: no task call at all, an open task left as it is.
func TestSnapshotTaskOffIsOff(t *testing.T) {
	for name, off := range map[string]func(*Reviewer){
		"team":     func(r *Reviewer) { r.teamTasks.Store(false) },
		"instance": func(r *Reviewer) { r.createTasks = false },
	} {
		h := snapshotHarness(t, snap1)
		h.push(t)
		id := h.st.snapshotTasks[0].ID
		off(h.r)
		h.bb.DefaultBranchCalls = 0
		h.bb.prDiff = pomDiff("<version>1.1</version>")
		h.push(t)
		p := prPayload(webhook.EventModified)
		p.PreviousTarget = &webhook.Ref{DisplayID: "develop"}
		h.r.HandleRetargeted(context.Background(), p)
		if h.bb.DefaultBranchCalls != 0 || len(h.bb.ResolvedTasks) != 0 || h.taskState(id) != "OPEN" {
			t.Errorf("%s: default branch calls = %d, resolved = %v", name, h.bb.DefaultBranchCalls, h.bb.ResolvedTasks)
		}
	}
}

// A read that fails is unknown, never "no SNAPSHOT": nothing changes.
func TestSnapshotTaskUnknownLeavesItAsItIs(t *testing.T) {
	for name, fail := range map[string]func(*harness){
		"default branch": func(h *harness) {
			h.r.defaultBranches = branchCache{}
			h.bb.defaultBranchErr = errors.New("boom")
		},
		"file diff":  func(h *harness) { h.bb.prDiffErr = errors.New("boom") },
		"open tasks": func(h *harness) { h.bb.openTasksErr = errors.New("boom") },
	} {
		h := snapshotHarness(t, snap1)
		h.push(t)
		sets := h.st.SnapshotTaskSets
		h.bb.prDiff = pomDiff()
		fail(h)
		h.r.HandleRetargeted(context.Background(), retargetPayload())
		if len(h.bb.ResolvedTasks) != 0 || h.st.SnapshotTaskSets != sets {
			t.Errorf("%s: resolved = %v, stored = %+v", name, h.bb.ResolvedTasks, h.st.snapshotTasks)
		}
	}
}

func retargetPayload() *webhook.Payload {
	p := prPayload(webhook.EventModified)
	p.PreviousTarget = &webhook.Ref{DisplayID: "develop"}
	return p
}

// A stopped PR gets no check (user, 2026-11-15); a PR the review skips
// for a missing AGENTS.md still does.
func TestSnapshotTaskStoppedVersusSkipped(t *testing.T) {
	h := snapshotHarness(t, snap1)
	now := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &now, IgnoreReason: store.IgnoreStopped}
	h.push(t)
	h.r.HandleRetargeted(context.Background(), retargetPayload())
	if h.bb.DefaultBranchCalls != 0 || len(h.bb.Tasks) != 0 {
		t.Fatalf("stopped: calls = %d, tasks = %d", h.bb.DefaultBranchCalls, len(h.bb.Tasks))
	}

	h = snapshotHarness(t, snap1)
	delete(h.bb.files, "src1:AGENTS.md")
	h.push(t)
	if len(h.bb.Tasks) != 1 {
		t.Fatalf("no AGENTS.md: tasks = %d", len(h.bb.Tasks))
	}
}

// A task a person resolved and then reopened is open again: noergler
// resolves it once its lines are gone.
func TestSnapshotTaskReopened(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.push(t)
	id := h.st.snapshotTasks[0].ID
	h.bb.comments[id].State = "RESOLVED"
	h.push(t)
	h.bb.comments[id].State = "OPEN"

	h.bb.prDiff = pomDiff("<version>1.1</version>")
	h.push(t)
	if len(h.bb.ResolvedTasks) != 1 || h.bb.ResolvedTasks[0] != id {
		t.Fatalf("resolved = %v", h.bb.ResolvedTasks)
	}
}

// The row lost a posted task's id: the open task with that text is taken
// over, never posted twice.
func TestSnapshotTaskLostIDTakenOver(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.push(t)
	id := h.st.snapshotTasks[0].ID
	h.st.snapshotTasks = nil

	h.push(t)
	if len(h.bb.Tasks) != 1 || len(h.st.snapshotTasks) != 1 || h.st.snapshotTasks[0].ID != id {
		t.Fatalf("tasks = %d, stored = %+v", len(h.bb.Tasks), h.st.snapshotTasks)
	}
}

// Past maxSnapshotBuildFiles the whole PR diff is read once.
func TestSnapshotTaskManyBuildFiles(t *testing.T) {
	h := snapshotHarness(t, snap1)
	var cs []bitbucket.Change
	for i := range maxSnapshotBuildFiles + 1 {
		cs = append(cs, bitbucket.Change{Path: strings.Repeat("m/", i) + "pom.xml", Type: "MODIFY"})
	}
	h.bb.changeSet = bitbucket.ChangeSet{Changes: cs}
	h.push(t)
	if len(h.bb.Tasks) != 1 || h.bb.FileDiffs != 0 {
		t.Fatalf("tasks = %d, file diffs = %d", len(h.bb.Tasks), h.bb.FileDiffs)
	}
}

// Build files unchanged since the last check (same blob ids): no diff read.
func TestSnapshotTaskRemembersHits(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.bb.changeSet = bitbucket.ChangeSet{Changes: []bitbucket.Change{{Path: "pom.xml", Type: "MODIFY", FromContentID: "a", ContentID: "b"}}}
	h.push(t)
	h.push(t)
	if h.bb.FileDiffs != 1 || len(h.bb.Tasks) != 1 {
		t.Fatalf("file diffs = %d, tasks = %d", h.bb.FileDiffs, len(h.bb.Tasks))
	}
	h.bb.changeSet.Changes[0].ContentID = "c"
	h.push(t)
	if h.bb.FileDiffs != 2 {
		t.Fatalf("changed blob: file diffs = %d", h.bb.FileDiffs)
	}
}

// A retarget of a PR no longer open does nothing.
func TestSnapshotTaskRetargetClosedPR(t *testing.T) {
	h := snapshotHarness(t, snap1)
	p := retargetPayload()
	p.PullRequest.State = "MERGED"
	h.r.HandleRetargeted(context.Background(), p)
	if h.bb.DefaultBranchCalls != 0 || len(h.bb.Tasks) != 0 {
		t.Fatalf("calls = %d, tasks = %d", h.bb.DefaultBranchCalls, len(h.bb.Tasks))
	}
}

// A PR that touches no build file reads no file diff and writes nothing.
func TestSnapshotTaskNoBuildFile(t *testing.T) {
	h := snapshotHarness(t, snap1)
	h.bb.changeSet = bitbucket.ChangeSet{Changes: []bitbucket.Change{{Path: "a.go", Type: "MODIFY"}}}
	h.push(t)
	if len(h.bb.Tasks) != 0 || h.bb.FileDiffs != 0 || h.st.SnapshotTaskSets != 0 {
		t.Fatalf("tasks = %d, file diffs = %d, writes = %d", len(h.bb.Tasks), h.bb.FileDiffs, h.st.SnapshotTaskSets)
	}
}
