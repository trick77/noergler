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

	// getenv and stdout are injected so a test does not have to mutate the
	// process environment or capture os.Stdout.
	getenv func(string) string
	stdout io.Writer
	// newClient builds the reviewer. The default reaches a real gateway;
	// a test supplies one that does not.
	newClient func(evals.Settings, string) (evals.Reviewer, error)
}

func main() {
	opt := options{getenv: os.Getenv, stdout: os.Stdout, newClient: dialGateway}
	flag.StringVar(&opt.promptPath, "prompt", "../prompts/review.txt", "review prompt template")
	// Unset is the series level (corpus/series.yaml) on the series model.
	// Keep a series at the level its earlier runs used: less thinking scores
	// worse on the same prompt.
	flag.StringVar(&opt.effort, "effort", "",
		"reasoning level (unset: the series level on the series model; empty: the model's balanced level)")
	flag.StringVar(&opt.jsonOut, "json", "", "also write the full result as JSON to this path")
	flag.IntVar(&opt.window, "context-window", 0, "override the gateway's max_input_tokens (0 = resolve it)")
	flag.DurationVar(&opt.timeout, "timeout", 10*time.Minute, "whole-run timeout")
	flag.Parse()
	opt.effortSet = wasSet(flag.CommandLine, "effort")

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
		Logger:          slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})),
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
	score := evals.Run(ctx, client, string(template), cases, counter.Count)
	score.Report(opt.stdout)

	if opt.jsonOut != "" {
		blob, err := json.MarshalIndent(score, "", " ")
		if err != nil {
			return false, err
		}
		// 0o644: a score report the operator reads and may commit; it holds
		// no secret, and the prompt and corpus it scores are already public.
		if err := os.WriteFile(opt.jsonOut, append(blob, '\n'), 0o644); err != nil { //nolint:gosec // G306
			return false, err
		}
		_, _ = fmt.Fprintln(opt.stdout, "wrote "+opt.jsonOut)
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
