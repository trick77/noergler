package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/evals"
)

// sampledReport runs a three-sample run against a stub that answers every
// seeded bug every time, and returns the report it wrote.
func sampledReport(t *testing.T, name string) string {
	t.Helper()
	client, _ := perfectFindings(t)
	opt := baseOptions(t, client)
	opt.samples = 3
	opt.jsonOut = filepath.Join(t.TempDir(), name)
	if missed, err := run(context.Background(), opt); err != nil || missed {
		t.Fatalf("missed=%v err=%v, want a clean sampled run", missed, err)
	}
	return opt.jsonOut
}

func TestRun_SampledWritesItsOwnReport(t *testing.T) {
	path := sampledReport(t, "s.json")
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got evals.Sampled
	if err := json.Unmarshal(blob, &got); err != nil {
		t.Fatal(err)
	}
	if got.Samples != 3 || got.Model != "some-model" || got.Effort != "some-level" {
		t.Errorf("samples %d, pair %s/%s", got.Samples, got.Model, got.Effort)
	}
	for _, v := range got.Views {
		if v.AllCaught != 1 || v.ControlsClean != 1 {
			t.Errorf("%s: all-caught %.2f, controls clean %.2f on a perfect reviewer", v.Name, v.AllCaught, v.ControlsClean)
		}
	}
}

// A sampled run is judged by the vote: a reviewer that never finds anything
// is exit 1, and the report is on disk before that verdict.
func TestRun_SampledMissIsExitOneAfterTheReportIsWritten(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	opt.samples = 3
	opt.jsonOut = filepath.Join(t.TempDir(), "s.json")
	missed, err := run(context.Background(), opt)
	if err == nil || !missed {
		t.Fatalf("missed=%v err=%v, want a miss", missed, err)
	}
	if _, statErr := os.Stat(opt.jsonOut); statErr != nil {
		t.Errorf("report not written before the verdict: %v", statErr)
	}
}

// Two samples form neither the series run nor a vote.
func TestRun_TwoSamplesIsRefused(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	opt.samples = 2
	missed, err := run(context.Background(), opt)
	if err == nil || missed || !strings.Contains(err.Error(), "-samples 2") {
		t.Fatalf("missed=%v err=%v", missed, err)
	}
}

// -mix needs no gateway: it must work with no credentials and no client.
func TestRun_MixScoresTwoReportsWithoutAGateway(t *testing.T) {
	primary, secondary := sampledReport(t, "p.json"), sampledReport(t, "s.json")
	out := &strings.Builder{}
	opt := options{
		mix:     primary + "," + secondary,
		jsonOut: filepath.Join(t.TempDir(), "mix.json"),
		getenv:  func(string) string { return "" },
		stdout:  out,
	}
	if missed, err := run(context.Background(), opt); err != nil || missed {
		t.Fatalf("missed=%v err=%v", missed, err)
	}
	if !strings.Contains(out.String(), evals.ViewMixVote) {
		t.Errorf("stdout carries no mixed view:\n%s", out.String())
	}
	blob, err := os.ReadFile(opt.jsonOut)
	if err != nil {
		t.Fatal(err)
	}
	var got evals.Mixed
	if err := json.Unmarshal(blob, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Views) != 3 || got.Seeded == 0 {
		t.Errorf("mixed report: %d view(s), %d seeded", len(got.Views), got.Seeded)
	}
}

func TestRun_MixNamesWhatIsWrong(t *testing.T) {
	good := sampledReport(t, "p.json")
	notJSON := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(notJSON, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	for arg, want := range map[string]string{
		"only-one.json":            "primary.json,secondary.json",
		good + ",/nonexistent/x":   "nonexistent",
		good + "," + notJSON:       "bad.json",
		notJSON + ",whatever.json": "bad.json",
	} {
		opt := options{mix: arg, getenv: func(string) string { return "" }, stdout: &strings.Builder{}}
		missed, err := run(context.Background(), opt)
		if err == nil || missed || !strings.Contains(err.Error(), want) {
			t.Errorf("-mix %s: missed=%v err=%v, want %q named", arg, missed, err, want)
		}
	}
}

// -rescore scores a finished report again from its own samples, with no
// gateway and no credentials.
func TestRun_RescoreReadsOnlyTheReport(t *testing.T) {
	src := sampledReport(t, "s.json")
	out := &strings.Builder{}
	opt := options{
		rescore: src,
		jsonOut: filepath.Join(t.TempDir(), "again.json"),
		getenv:  func(string) string { return "" },
		stdout:  out,
	}
	if missed, err := run(context.Background(), opt); err != nil || missed {
		t.Fatalf("missed=%v err=%v", missed, err)
	}
	if !strings.Contains(out.String(), evals.ViewTiered3) {
		t.Errorf("stdout carries no views:\n%s", out.String())
	}
	if _, err := os.Stat(opt.jsonOut); err != nil {
		t.Errorf("rescored report not written: %v", err)
	}

	notJSON := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(notJSON, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	stray := filepath.Join(t.TempDir(), "stray.json")
	if err := os.WriteFile(stray, []byte(`{"Samples":3,"Cases":[{"Case":"not-a-case"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/nonexistent/x.json", notJSON, stray} {
		opt := options{rescore: path, getenv: func(string) string { return "" }, stdout: &strings.Builder{}}
		if missed, err := run(context.Background(), opt); err == nil || missed {
			t.Errorf("-rescore %s: missed=%v err=%v, want an error that is not a miss", path, missed, err)
		}
	}
}

// -confirm consolidates a finished report with one more call per group, on
// the client this run builds, and writes its own report.
func TestRun_ConfirmConsolidatesASampledReport(t *testing.T) {
	src := sampledReport(t, "s.json")
	client, seeded := perfectFindings(t)
	opt := baseOptions(t, client)
	opt.confirm = src
	opt.consolidatePath = "../../internal/evals/briefs/merge.txt"
	opt.confirmGroups = 0 // below one is one
	opt.jsonOut = filepath.Join(t.TempDir(), "c.json")
	if missed, err := run(context.Background(), opt); err != nil || missed {
		t.Fatalf("missed=%v err=%v", missed, err)
	}
	blob, err := os.ReadFile(opt.jsonOut)
	if err != nil {
		t.Fatal(err)
	}
	var got evals.Confirmed
	if err := json.Unmarshal(blob, &got); err != nil {
		t.Fatal(err)
	}
	if got.Seeded != seeded || got.Confirmer == "" || got.Groups != 1 {
		t.Errorf("seeded %d, confirmer %q, groups %d", got.Seeded, got.Confirmer, got.Groups)
	}
	for _, v := range got.Views {
		if v.AllCaught != 1 || v.ControlsClean != 1 {
			t.Errorf("%s: all-caught %.2f, controls clean %.2f on a perfect reviewer", v.Name, v.AllCaught, v.ControlsClean)
		}
	}

	notJSON := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(notJSON, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*options){
		"missing report": func(o *options) { o.confirm = "/nonexistent/x.json" },
		"not json":       func(o *options) { o.confirm = notJSON },
		"missing brief":  func(o *options) { o.consolidatePath = "/nonexistent/brief.txt" },
		"empty report":   func(o *options) { o.confirm = writeFile(t, `{"Samples":3}`) },
	} {
		bad := baseOptions(t, client)
		bad.confirm, bad.consolidatePath = src, opt.consolidatePath
		mutate(&bad)
		if missed, err := run(context.Background(), bad); err == nil || missed {
			t.Errorf("%s: missed=%v err=%v, want an error that is not a miss", name, missed, err)
		}
	}
}

func writeFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "f.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// -cases runs only what it names, and a name the corpus does not have is an
// error, never a smaller run.
func TestRun_CasesLimitsTheRun(t *testing.T) {
	client, _ := perfectFindings(t)
	opt := baseOptions(t, client)
	opt.only = "nil-deref, clean-rename"
	opt.jsonOut = filepath.Join(t.TempDir(), "r.json")
	if missed, err := run(context.Background(), opt); err != nil || missed {
		t.Fatalf("missed=%v err=%v", missed, err)
	}
	blob, err := os.ReadFile(opt.jsonOut)
	if err != nil {
		t.Fatal(err)
	}
	var score evals.Score
	if err := json.Unmarshal(blob, &score); err != nil {
		t.Fatal(err)
	}
	if len(score.Results) != 2 || score.Seeded != 1 {
		t.Errorf("%d case(s), %d seeded, want 2 and 1", len(score.Results), score.Seeded)
	}

	opt.only = "nil-deref,no-such-case"
	if missed, err := run(context.Background(), opt); err == nil || missed || !strings.Contains(err.Error(), "no-such-case") {
		t.Errorf("missed=%v err=%v, want the unknown case named", missed, err)
	}
}
