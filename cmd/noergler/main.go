// Command noergler is the Bitbucket PR review bridge.
//
//	noergler serve     run the service (default)
//	noergler migrate   apply pending database migrations and exit
//	noergler version   print the version and exit
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/trick77/noergler-go/internal/buildinfo"
	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/httpapi"
	"github.com/trick77/noergler-go/internal/logging"
	"github.com/trick77/noergler-go/internal/store"
)

func main() {
	cmd := "serve"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	var err error
	switch cmd {
	case "serve":
		err = serve(logging.Setup())
	case "migrate":
		err = migrate(logging.Setup())
	case "version":
		fmt.Println(buildinfo.Version())
	case "-h", "--help", "help":
		fmt.Fprintln(os.Stderr, "usage: noergler [serve|migrate|version]")
	default:
		fmt.Fprintf(os.Stderr, "noergler: unknown command %q\nusage: noergler [serve|migrate|version]\n", cmd)
		os.Exit(2)
	}
	if err != nil {
		slog.Error("startup aborted", "error", err)
		os.Exit(1)
	}
}

// migrate applies pending migrations and exits: the init container's job.
// Only DATABASE_URL is needed, so a teams.yaml is not required here.
func migrate(log *slog.Logger) error {
	log.Info("noergler version: " + buildinfo.Version())
	dsn, ok := os.LookupEnv("DATABASE_URL")
	if !ok || strings.TrimSpace(dsn) == "" {
		return errors.New("Environment variable DATABASE_URL is not set")
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
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
	log.Info("noergler version: " + buildinfo.Version())
	app, err := config.Load(config.OSLookup)
	if err != nil {
		return err
	}
	config.Dump(app, log)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Shared layer: any failure here aborts boot. The Bitbucket and Jira
	// checks join in Phase 3; all checks run before the abort so the log
	// names every failure at once.
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
	if len(checks) > 0 {
		return fmt.Errorf("Startup aborted: %d connection(s) failed: %s", len(checks), strings.Join(checks, ", "))
	}

	// Per-team startup (inference check, riptide ping, store reconciliation)
	// arrives with the later phases; for now a team the file resolves is a
	// team that takes traffic, so the probes already show the right slugs.
	enabled := append([]string(nil), app.Order...)
	sort.Strings(enabled)
	disabled := make([]string, 0, len(app.Disabled))
	for slug := range app.Disabled {
		disabled = append(disabled, slug)
	}
	sort.Strings(disabled)
	summary := fmt.Sprintf("teams_ready enabled=%s disabled=%s", pyList(enabled), pyList(disabled))
	if len(disabled) > 0 {
		log.Warn(summary)
	} else {
		log.Info(summary)
	}
	if len(enabled) == 0 {
		log.Error("no team is enabled; /ready reports 503 until the config is fixed")
	}

	status := func() ([]string, []string) { return enabled, disabled }
	srv := httpapi.New(status, log)

	addr := net.JoinHostPort(app.Server.Host, strconv.Itoa(app.Server.Port))
	log.Info("Bridge service started", "base_url", app.LLM.BaseURL, "teams", len(enabled))
	if err := httpapi.Run(ctx, addr, srv.Handler(), log); err != nil {
		return err
	}
	log.Info("Bridge service stopped")
	return nil
}

// pyList renders a slug list the way the Python service logged it, so the
// Splunk alert on `teams_ready` keeps matching.
func pyList(items []string) string {
	out := "["
	for i, s := range items {
		if i > 0 {
			out += ", "
		}
		out += "'" + s + "'"
	}
	return out + "]"
}
