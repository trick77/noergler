package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/inference"
)

// conflictStub declines every conflict, or answers T1 when answer is set:
// valid on every case, so a control answered that way is wrong.
type conflictStub struct {
	stubClient
	answer bool
}

func (s conflictStub) ResolveConflict(context.Context, inference.ConflictRequest) inference.ConflictResult {
	if !s.answer {
		return inference.ConflictResult{Resolution: inference.ConflictResolution{Feasible: false, Reason: "author's call"}}
	}
	return inference.ConflictResult{Resolution: inference.ConflictResolution{Feasible: true,
		Lines: map[int][]string{1: {"T1"}}}}
}

func TestRunConflicts(t *testing.T) {
	// A control with one conflict for the model, so T1 is a valid answer.
	const control = "inv-config-timeout-values"
	opt := baseOptions(t, conflictStub{})
	opt.conflicts, opt.samples, opt.parallel = true, 2, 2
	opt.jsonOut = filepath.Join(t.TempDir(), "run.json")
	missed, err := run(context.Background(), opt)
	if err != nil || missed {
		t.Fatalf("declining everything: missed %v, err %v", missed, err)
	}
	out := opt.stdout.(*strings.Builder).String()
	if !strings.Contains(out, "declined") || !strings.Contains(out, "wrote "+opt.jsonOut) {
		t.Errorf("report:\n%s", out)
	}
	if _, err := os.Stat(opt.jsonOut); err != nil {
		t.Fatal(err)
	}

	opt = baseOptions(t, conflictStub{answer: true})
	opt.conflicts, opt.only = true, control
	if missed, err := run(context.Background(), opt); !missed || err == nil {
		t.Errorf("an answered control: missed %v, err %v", missed, err)
	}

	draft := filepath.Join(t.TempDir(), "draft.txt")
	if err := os.WriteFile(draft, []byte("draft {conflicts}"), 0o600); err != nil {
		t.Fatal(err)
	}
	opt = baseOptions(t, conflictStub{})
	opt.conflicts, opt.only, opt.promptPath = true, control, draft
	if _, err := run(context.Background(), opt); err != nil {
		t.Errorf("a draft prompt: %v", err)
	}
	opt.promptPath = draft + ".missing"
	if _, err := run(context.Background(), opt); err == nil || !strings.Contains(err.Error(), "read prompt") {
		t.Errorf("a missing draft: %v", err)
	}

	for name, mut := range map[string]func(*options){
		"rescore":  func(o *options) { o.rescore = "x.json" },
		"samples":  func(o *options) { o.samples = -1 },
		"no draft": func(o *options) { o.promptPath = "/nonexistent" },
	} {
		dialled := false
		opt = baseOptions(t, conflictStub{})
		opt.conflicts = true
		opt.newClient = func(evals.Settings, string) (evals.Reviewer, error) { dialled = true; return conflictStub{}, nil }
		mut(&opt)
		if _, err := run(context.Background(), opt); err == nil || dialled {
			t.Errorf("%s: err %v, dialled %v", name, err, dialled)
		}
	}

	opt = baseOptions(t, stubClient{})
	opt.conflicts = true
	if _, err := run(context.Background(), opt); err == nil || !strings.Contains(err.Error(), "cannot resolve") {
		t.Errorf("a client without ResolveConflict: %v", err)
	}
	opt.only = "no-such-case"
	if _, err := run(context.Background(), opt); err == nil {
		t.Error("an unknown case ran")
	}
}
