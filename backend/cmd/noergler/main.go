// Command noergler is the Bitbucket PR review bridge.
//
//	noergler serve     run the service (default)
//	noergler migrate   apply pending database migrations and exit
//	noergler version   print the version and exit
//	noergler -healthcheck  probe /healthz, exit 0 on a 2xx (liveness: the container healthcheck)
//	noergler -readycheck   probe /readyz, exit 0 on a 2xx (readiness: 503 while no team is enabled)
//
// Both probes run inside the container: kit's /healthz and /readyz answer a
// loopback caller only and 404 anyone else, so an orchestrator's HTTP probe
// from the node never passes. Deploy them as exec probes.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/trick77/ship/kit/httpx"
	"github.com/trick77/ship/kit/logging"
	"github.com/trick77/ship/kit/run"
	"github.com/trick77/ship/kit/version"

	"github.com/trick77/noergler/internal/api"
	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/secretbox"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/noergler/internal/tokens"
)

func main() {
	cmd := "serve"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	var err error
	switch cmd {
	case "serve":
		err = withLog(serve)
	case "migrate":
		err = withLog(migrate)
	case "version":
		fmt.Println(version.String())
	case "-healthcheck":
		// The container probes: the distroless image has no shell and no wget.
		os.Exit(httpx.Healthcheck(httpx.LocalURL(listenAddr(), "/healthz")))
	case "-readycheck":
		os.Exit(httpx.Healthcheck(httpx.LocalURL(listenAddr(), "/readyz")))
	case "-h", "--help", "help":
		fmt.Fprintln(os.Stderr, usage)
	default:
		fmt.Fprintf(os.Stderr, "noergler: unknown command %q\n%s\n", cmd, usage)
		os.Exit(2)
	}
	if err != nil {
		slog.Error("startup aborted", logging.Err(err))
		os.Exit(1)
	}
}

const usage = "usage: noergler [serve|migrate|version|-healthcheck|-readycheck]"

// withLog sets up logging before anything logs (config loading does), names
// the build, then runs. An unknown LOG_LEVEL still fails startup: logging
// starts at info so the fault surfaces with the rest of the config's (serve)
// or on its own (migrate), never as a knob that quietly does nothing.
func withLog(run func(*slog.Logger) error) error {
	env := os.Getenv("NOERGLER_ENV")
	if env == "" {
		env = "dev"
	}
	level := os.Getenv("LOG_LEVEL")
	if _, err := logging.ParseLevel(level); err != nil {
		level = ""
	}
	log, err := logging.Setup(logging.Options{
		Level:   level,
		Format:  logging.Splunk,
		Writer:  os.Stdout,
		Service: "noergler",
		Env:     env,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "noergler:", err)
		os.Exit(1)
	}
	log.Info("noergler version: " + version.String())
	return run(log)
}

// listenAddr is SERVER_HOST:SERVER_PORT as serve binds it, read without the
// rest of the config: the healthcheck needs nothing else.
func listenAddr() string {
	port := strings.TrimSpace(os.Getenv("SERVER_PORT"))
	if port == "" {
		port = "8080"
	}
	return net.JoinHostPort(strings.TrimSpace(os.Getenv("SERVER_HOST")), port)
}

// migrate applies pending migrations and exits: the init container's job.
// Only DATABASE_URL is needed, so a teams.yaml is not required here.
func migrate(log *slog.Logger) error {
	if _, err := logging.ParseLevel(os.Getenv("LOG_LEVEL")); err != nil {
		return fmt.Errorf("LOG_LEVEL: %w", err)
	}
	dsn, ok := os.LookupEnv("DATABASE_URL")
	if !ok || strings.TrimSpace(dsn) == "" {
		return errors.New("environment variable DATABASE_URL is not set")
	}
	ctx := context.Background()
	db, err := store.Open(ctx, dsn, log)
	if err != nil {
		return err
	}
	defer db.Close()
	if err := db.Migrate(ctx); err != nil {
		return fmt.Errorf("migrate: %w", err)
	}
	log.Info("migrations up to date")
	return nil
}

func serve(log *slog.Logger) error {
	app, err := config.Load(config.OSLookup)
	if err != nil {
		return err
	}
	config.Dump(app, log)
	for _, name := range config.RemovedPromptSettings(config.OSLookup) {
		log.Warn(name + " is " + config.RemovedPromptReason)
	}

	// SIGINT/SIGTERM end startup and then serving; the queue drain below runs
	// on. run.Context releases the handler shortly after ctx ends, so a second
	// signal kills the process mid-drain, review in flight included.
	ctx, stop := run.Context(context.Background())
	defer stop()

	// Shared layer: any failure here aborts boot, because none of it belongs to
	// one team. Every check runs before the abort so the log names all the
	// failures at once rather than one per restart.
	checks := []string{}
	db, err := store.Open(ctx, app.Database.URL, log)
	if err == nil {
		defer db.Close()
		err = db.SchemaCurrent(ctx)
	}
	if err != nil {
		log.Error("Database: " + err.Error())
		checks = append(checks, "Database")
	} else {
		log.Info("Database: OK")
	}

	bb, err := bitbucket.New(app.Bitbucket, log)
	if err == nil {
		err = bb.CheckConnectivity(ctx)
	}
	if err != nil {
		log.Error("Bitbucket: " + err.Error())
		checks = append(checks, "Bitbucket")
	}

	if err == nil {
		log.Info("Bot username: " + bb.BotUsername())
		log.Info("Bitbucket: OK")
	}

	jr, err := jira.New(app.Jira, log)
	if err == nil {
		err = jr.CheckConnectivity(ctx)
	}
	if err != nil {
		log.Error("Jira: " + err.Error())
		checks = append(checks, "Jira")
	} else {
		log.Info("Jira: OK")
	}
	if len(checks) > 0 {
		return fmt.Errorf("startup aborted: %d connection(s) failed: %s", len(checks), strings.Join(checks, ", "))
	}

	// The tokenizer's vocabulary is compiled in but cold; warming it here
	// keeps the first review off the critical path.
	counter, err := tokens.New()
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	counter.Warm()

	// Per-team startup. Reconciliation runs inside Boot and before any
	// Reviewer is built, because review.New copies the review config by value.
	// LoadInstance has checked the key's length, the one thing New refuses.
	box, err := secretbox.New(app.SecretsKey)
	if err != nil {
		return fmt.Errorf("NOERGLER_SECRETS_KEY: %w", err)
	}
	reg, err := teams.Boot(ctx, app, teams.Deps{
		Claims:      db,
		ReviewStore: db,
		Bitbucket:   bb,
		Tokens:      counter,
		Log:         log,
		Credentials: db,
		Box:         box,
	})
	if err != nil {
		return err
	}

	// The worker runs on its own context, never one a signal cancels.
	// queue.run derives every job's context from what Start was given, so a
	// cancelled one would fail the review that Stop then waits for: its diff
	// fetch, its LLM call and its store writes would all fail with context
	// canceled, and safeDB would swallow them, losing the run row and its
	// cost. Cancellation reaches the worker through Stop alone; killing the
	// process (SIGKILL, or a second signal) ends it sooner.
	queueCtx, stopQueue := context.WithCancel(context.WithoutCancel(ctx))
	defer stopQueue()
	q := queue.New(reg.Review, app.Queue.InferenceConcurrency, app.Queue.InferenceConcurrencyPerTeam, log)
	q.Start(queueCtx)

	srv := httpapi.New(reg.Status, log)
	api.Register(srv, api.Deps{
		Teams:       reg,
		Queue:       q,
		Store:       db,
		Claims:      db,
		Bitbucket:   api.Bot(bb),
		Log:         log,
		BotUsername: bb.BotUsername(),
		PublicURL:   app.Server.PublicURL,
		// Trimmed once, here, so the page never joins a double slash.
		BitbucketURL: strings.TrimRight(app.Bitbucket.BaseURL, "/"),
		// The operator dashboard. Read-only and unauthenticated, so the
		// ingress is the boundary.
		Dashboard:      q,
		DashboardStore: db,
	})

	enabled, _ := reg.Status()
	addr := net.JoinHostPort(app.Server.Host, strconv.Itoa(app.Server.Port))
	log.Info("Bridge service started", "base_url", app.LLM.BaseURL, "teams", len(enabled))
	log.Info("listening", "addr", addr)
	// Serve returns on SIGINT/SIGTERM once in-flight requests have drained; a
	// second signal kills that drain too.
	runErr := httpx.Serve(ctx, httpx.NewServer(addr, srv.Handler()), 15*time.Second)

	// Drain in order: the server has already stopped accepting, so the queue
	// finishes the review in flight before the pool closes under it. Not a
	// defer, because `defer db.Close()` above is registered earlier and would
	// otherwise run first. Stop has no timeout: a long review holds shutdown,
	// which the pod's termination grace period has to allow for: its SIGKILL
	// ends the drain, and so does any second SIGINT/SIGTERM, pod or not. With
	// the inference pool that budget is now prepare + pool wait + the
	// gateway call timeout (REVIEW_INFERENCE_CALL_TIMEOUT) + posting, for
	// every review still in flight.
	q.Stop()
	if runErr != nil {
		return runErr
	}
	log.Info("Bridge service stopped")
	return nil
}
