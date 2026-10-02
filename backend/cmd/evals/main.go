// Command evals scores prompts/review.txt against the seeded-bug corpus.
//
// It is part of the product, not a throwaway: the corpus is embedded, the
// scoring lives in internal/evals with its own tests, and a run is one
// command. Re-run it after any edit to prompts/review.txt, to the assembly
// in internal/inference, or when changing model or reasoning effort.
//
//	EVAL_BASE_URL=https://llm.example/v1 EVAL_API_KEY=... \
//	  go run ./cmd/evals -prompt ../prompts/review.txt
//
// No model or level is a default in code: a series is only comparable at
// one pair, which internal/evals/corpus/series.yaml names and
// internal/evals/results/README.md records. EVAL_MODEL unset is the series
// model; -effort unset is the series level on that model and the balanced
// level on any other. A passed -effort, empty included, always wins, and is
// checked against the profile at startup, before any request.
//
// EVAL_MODEL must be an llmwire PROFILE id, not whatever name the endpoint
// answers to: llmwire's registry is fixed and rejects an unknown id before
// any request is sent. The gateway alias defaults to that id, which is what
// a plain OpenAI-compatible host serves; EVAL_ALIAS covers one that renames.
//
// The endpoint only has to speak the OpenAI chat-completions API. It reaches
// the gateway through llmwire's own environment, the same path production
// uses, so an eval exercises the real client: real schema, real parse, real
// cost accounting. Nothing about the review path is stubbed.
//
// -samples N reviews each case N times and scores what merging those reviews
// would post (2-of-3 vote, union, a 2+1 adaptive rule) next to the single
// run, with each view's cost against one review. It is an experiment report
// under internal/evals/results/sampled/, never a row in the series, and its
// exit code judges the vote view. Pass a -timeout that fits N times the calls.
//
// -mix primary.json,secondary.json reads two sampled reports and scores a vote
// of one primary review with secondary ones (the production level once, a
// cheaper level or model for the rest). It makes no calls and needs no
// credentials; a failure is exit 2.
//
// -confirm sampled.json takes a finished sampled report, unions its reviews
// in groups of three and asks this run's model to consolidate each group in
// one call (internal/evals/briefs/merge.txt appended to the review prompt):
// merge the duplicates, drop nothing. The model may differ from the
// one that wrote the reviews. A failure is exit 2.
//
// Exit codes: 0 every seeded bug caught and the clean controls stayed clean,
// 1 the prompt got worse (a seeded bug was missed, or a finding was invented
// on a case that seeds none), 2 the run could not happen (no credentials,
// unreachable gateway, a case that never completed). Both halves of 1 are one
// code because both are a regression; anything else exits 2 and says why,
// rather than passing quietly.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/tokens"
)

// options is what a run needs. Grouped rather than passed as seven
// parameters so the test can drive the whole orchestration - the flag
// handling, the prompt read, the JSON write and both exit paths - without a
// gateway.
type options struct {
	promptPath string
	effort     string
	// effortSet is whether -effort was passed at all, so an explicit empty
	// (the balanced level) is told apart from no flag (the series level).
	effortSet bool
	jsonOut   string
	timeout   time.Duration
	window    int
	// samples is how many times each case is reviewed. 1 is the series run;
	// more is a sampled run, a different report that is never a series row.
	samples  int
	parallel int
	// mix is "primary.json,secondary.json": two sampled reports scored as
	// one mixed vote, without a gateway.
	mix string
	// rescore is a sampled report to score again from the samples it holds.
	rescore string
	// confirm is a sampled report whose reviews are unioned in groups of
	// three and consolidated by one more call each, on the model this run
	// names. consolidatePath is that call's brief, confirmGroups how many
	// groups per case.
	// only limits the run to the named cases, comma-separated. A run on a
	// costly model pays per case, so it must be able to buy only the cases
	// that answer its question.
	only            string
	confirm         string
	consolidatePath string
	confirmGroups   int

	// getenv and stdout are injected so a test does not have to mutate the
	// process environment or capture os.Stdout.
	getenv func(string) string
	stdout io.Writer
	// newClient builds the reviewer. The default reaches a real gateway;
	// a test supplies one that does not.
	newClient func(evals.Settings, string) (evals.Reviewer, error)
}

// logLevel is the client logger's level. -usage lowers it to Debug, where
// llmwire logs every call's token accounting: the series report carries no
// tokens, and a run on a costly model should not need a second one to learn
// what it spent.
var logLevel slog.LevelVar

func main() {
	logLevel.Set(slog.LevelWarn)
	usage := flag.Bool("usage", false, "log every call's token usage to stderr")
	opt := options{getenv: os.Getenv, stdout: os.Stdout, newClient: dialGateway}
	flag.StringVar(&opt.only, "cases", "", "comma-separated case names; unset runs the whole corpus")
	flag.StringVar(&opt.confirm, "confirm", "",
		"sampled.json: union its reviews in groups of three and consolidate each with one call on this run's model")
	flag.StringVar(&opt.consolidatePath, "consolidate", "internal/evals/briefs/merge.txt", "consolidation brief appended to the review prompt")
	flag.IntVar(&opt.confirmGroups, "confirm-groups", 3, "groups of three consolidated per case")
	flag.StringVar(&opt.promptPath, "prompt", "../prompts/review.txt", "review prompt template")
	// Unset is the series level (corpus/series.yaml) on the series model.
	// Keep a series at the level its earlier runs used: less thinking scores
	// worse on the same prompt.
	flag.StringVar(&opt.effort, "effort", "",
		"reasoning level (unset: the series level on the series model; empty: the model's balanced level)")
	flag.StringVar(&opt.jsonOut, "json", "", "also write the full result as JSON to this path")
	flag.IntVar(&opt.window, "context-window", 0, "override the gateway's max_input_tokens (0 = resolve it)")
	flag.DurationVar(&opt.timeout, "timeout", 10*time.Minute, "whole-run timeout")
	flag.IntVar(&opt.samples, "samples", 1,
		"reviews per case; above 1 scores merged views (vote, union, adaptive) instead of the series run, and needs a longer -timeout")
	flag.IntVar(&opt.parallel, "parallel", 3, "sampled run: reviews in flight after a case's first one has completed")
	flag.StringVar(&opt.mix, "mix", "",
		"primary.json,secondary.json: score one primary review voted with secondary ones from two sampled reports; makes no calls")
	flag.StringVar(&opt.rescore, "rescore", "",
		"sampled.json: recompute every view from the samples a sampled report holds; makes no calls")
	flag.Parse()
	opt.effortSet = wasSet(flag.CommandLine, "effort")
	if *usage {
		logLevel.Set(slog.LevelDebug)
	}

	missed, err := run(context.Background(), opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "evals: "+err.Error())
		if missed {
			os.Exit(1)
		}
		os.Exit(2)
	}
}

// wasSet reports whether the flag was passed at all, which is what tells an
// explicit `-effort ""` (the balanced level) from no flag (the series level).
func wasSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

// dialGateway builds the real client and proves the gateway serves the
// profile before any case is scored.
func dialGateway(s evals.Settings, effort string) (evals.Reviewer, error) {
	env := evals.GatewayEnv(s)
	client, err := inference.New(inference.Options{
		Model:           s.Model,
		ReasoningEffort: effort,
		APIKey:          s.APIKey,
		ContextWindow:   s.ContextWindow,
		Logger:          slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: &logLevel})),
		Env: func(name string) (string, bool) {
			v, ok := env[name]
			return v, ok
		},
	})
	if err != nil {
		return nil, fmt.Errorf("client: %w", err)
	}
	// Startup resolves the context window and checks the profile is served.
	// Doing it here rather than in run keeps run free of the network.
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if err := client.Startup(ctx); err != nil {
		return nil, fmt.Errorf("gateway unreachable or profile not served: %w", err)
	}
	return client, nil
}

// header names the model and the level the run used. A started client's label
// carries the level llmwire actually sent, so two runs at different resolved
// levels read differently even with -effort unset.
func header(client evals.Reviewer, model, effort string) string {
	if l, ok := client.(interface{ Label() string }); ok {
		return l.Label()
	}
	if effort == "" {
		effort = "model balanced"
	}
	return model + ", effort " + effort
}

// run returns (missed, err): missed distinguishes a prompt regression from a
// run that could not happen, which the exit code has to tell apart.
func run(ctx context.Context, opt options) (bool, error) {
	// The zero value is the series run, so a caller that sets neither field
	// gets today's behaviour.
	if opt.samples == 0 {
		opt.samples = 1
	}
	if opt.samples != 1 && opt.samples < evals.MinSamples {
		return false, fmt.Errorf("-samples %d: 1 for the series run, or at least %d for a sampled one",
			opt.samples, evals.MinSamples)
	}
	if opt.parallel < 1 {
		opt.parallel = 1
	}
	if opt.mix != "" {
		return false, mix(opt)
	}
	if opt.rescore != "" {
		return false, rescore(opt)
	}
	settings, err := evals.ResolveSettings(opt.getenv, opt.window)
	if err != nil {
		return false, err
	}
	opt.effort = evals.DefaultSeries().EffortFor(settings.Model, opt.effort, opt.effortSet)
	template, err := os.ReadFile(opt.promptPath) //nolint:gosec // G304: an operator-supplied flag in a dev tool
	if err != nil {
		return false, fmt.Errorf("read prompt: %w", err)
	}
	cases, err := evals.LoadCorpus()
	if err != nil {
		return false, fmt.Errorf("load corpus: %w", err)
	}
	if cases, err = selectCases(cases, opt.only); err != nil {
		return false, err
	}
	counter, err := tokens.New()
	if err != nil {
		return false, fmt.Errorf("tokenizer: %w", err)
	}
	client, err := opt.newClient(settings, opt.effort)
	if err != nil {
		return false, err
	}

	ctx, cancel := context.WithTimeout(ctx, opt.timeout)
	defer cancel()

	// Write errors are dropped: this is the progress line on stdout, and a
	// report that cannot be printed changes nothing about the verdict.
	_, _ = fmt.Fprintf(opt.stdout, "model %s via %s, %d case(s)\n\n",
		header(client, settings.Model, opt.effort), settings.BaseURL, len(cases))
	if opt.confirm != "" {
		return false, confirm(ctx, opt, client, string(template), cases, counter.Count,
			header(client, settings.Model, opt.effort))
	}
	if opt.samples > 1 {
		sampled := evals.RunSampled(ctx, client, string(template), cases, counter.Count, opt.samples, opt.parallel, opt.stdout)
		sampled.Model, sampled.Effort = settings.Model, opt.effort
		if sampled.Effort == "" {
			sampled.Effort = "balanced"
		}
		sampled.Report(opt.stdout)
		if err := writeJSON(opt, sampled); err != nil {
			return false, err
		}
		if err := sampled.ErrIncomplete(); err != nil {
			return false, err
		}
		if err := sampled.ErrRegressed(); err != nil {
			return true, err
		}
		return false, nil
	}
	score := evals.Run(ctx, client, string(template), cases, counter.Count)
	score.Report(opt.stdout)
	if err := writeJSON(opt, score); err != nil {
		return false, err
	}

	if err := score.ErrIncomplete(); err != nil {
		return false, err
	}
	// Missed and invented are both "the prompt got worse", so both take the
	// exit code CI reads for a regression. Only ErrIncomplete, which means the
	// run could not happen, is the other one. Joined rather than returned in
	// turn: a run that both missed a bug and invented one should say so once.
	if err := errors.Join(score.ErrMissed(), score.ErrInvented()); err != nil {
		return true, err
	}
	return false, nil
}

// writeJSON writes the report to -json, when one was asked for. It runs
// before the verdict, so a failed run is still a committed one.
func writeJSON(opt options, report any) error {
	if opt.jsonOut == "" {
		return nil
	}
	blob, err := json.MarshalIndent(report, "", " ")
	if err != nil {
		return err
	}
	// 0o644: a score report the operator reads and may commit; it holds
	// no secret, and the prompt and corpus it scores are already public.
	if err := os.WriteFile(opt.jsonOut, append(blob, '\n'), 0o644); err != nil { //nolint:gosec // G306
		return err
	}
	_, _ = fmt.Fprintln(opt.stdout, "wrote "+opt.jsonOut)
	return nil
}

// mix scores two sampled reports as one mixed vote. It reads files only: the
// samples are already paid for.
func mix(opt options) error {
	primaryPath, secondaryPath, ok := strings.Cut(opt.mix, ",")
	if !ok || primaryPath == "" || secondaryPath == "" {
		return fmt.Errorf("-mix %q: want primary.json,secondary.json", opt.mix)
	}
	var runs [2]evals.Sampled
	for i, path := range []string{primaryPath, secondaryPath} {
		blob, err := os.ReadFile(path) //nolint:gosec // G304: an operator-supplied flag in a dev tool
		if err != nil {
			return err
		}
		if err := json.Unmarshal(blob, &runs[i]); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	cases, err := evals.LoadCorpus()
	if err != nil {
		return fmt.Errorf("load corpus: %w", err)
	}
	mixed, err := evals.Mix(cases, runs[0], runs[1])
	if err != nil {
		return err
	}
	mixed.Report(opt.stdout)
	return writeJSON(opt, mixed)
}

// rescore recomputes a sampled report's views from its own samples, so a
// view added after the run was paid for is scored on the same reviews.
func rescore(opt options) error {
	blob, err := os.ReadFile(opt.rescore) //nolint:gosec // G304: an operator-supplied flag in a dev tool
	if err != nil {
		return err
	}
	var sampled evals.Sampled
	if err := json.Unmarshal(blob, &sampled); err != nil {
		return fmt.Errorf("%s: %w", opt.rescore, err)
	}
	cases, err := evals.LoadCorpus()
	if err != nil {
		return fmt.Errorf("load corpus: %w", err)
	}
	if err := sampled.Rescore(cases); err != nil {
		return fmt.Errorf("%s: %w", opt.rescore, err)
	}
	sampled.Report(opt.stdout)
	return writeJSON(opt, sampled)
}

// confirm consolidates the reviews of a finished sampled report.
func confirm(ctx context.Context, opt options, client evals.Reviewer, template string, cases []evals.Case, count inference.CountFunc, label string) error {
	blob, err := os.ReadFile(opt.confirm) //nolint:gosec // G304: an operator-supplied flag in a dev tool
	if err != nil {
		return err
	}
	var sampled evals.Sampled
	if err := json.Unmarshal(blob, &sampled); err != nil {
		return fmt.Errorf("%s: %w", opt.confirm, err)
	}
	suffix, err := os.ReadFile(opt.consolidatePath) //nolint:gosec // G304: an operator-supplied flag in a dev tool
	if err != nil {
		return fmt.Errorf("read consolidation brief: %w", err)
	}
	groups := opt.confirmGroups
	if groups < 1 {
		groups = 1
	}
	confirmed, err := evals.RunConfirm(ctx, client, template, string(suffix), cases, sampled, count, groups, opt.stdout)
	if err != nil {
		return fmt.Errorf("%s: %w", opt.confirm, err)
	}
	confirmed.Confirmer = label
	confirmed.Report(opt.stdout)
	if err := writeJSON(opt, confirmed); err != nil {
		return err
	}
	return confirmed.ErrIncomplete()
}

// selectCases keeps the named cases, in corpus order. An unknown name is an
// error: a typo must not quietly run a smaller corpus and report its score
// as the answer.
func selectCases(cases []evals.Case, only string) ([]evals.Case, error) {
	if strings.TrimSpace(only) == "" {
		return cases, nil
	}
	want := map[string]bool{}
	for _, n := range strings.Split(only, ",") {
		want[strings.TrimSpace(n)] = true
	}
	var out []evals.Case
	for _, c := range cases {
		if want[c.Name] {
			out = append(out, c)
			delete(want, c.Name)
		}
	}
	for n := range want {
		return nil, fmt.Errorf("-cases: no case named %q", n)
	}
	return out, nil
}
