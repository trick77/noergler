// Package store is the Postgres layer: pool, embedded migrations and every
// query the service runs. One function per query, positional pgx args, and a
// transaction wherever more than one statement must land together.
package store

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// migrateLockID is the advisory-lock key held for the whole of Migrate. Fixed:
// pg_advisory_lock namespaces by value alone. From "noer".
const migrateLockID int64 = 0x6E6F6572

// migrateLockTimeout bounds how long one migrate waits for another's.
// pg_advisory_lock blocks forever; a wedged holder would otherwise leave the
// init container hanging with no log line.
var migrateLockTimeout = 5 * time.Minute

// Store owns the pool.
type Store struct {
	pool *pgxpool.Pool
	log  *slog.Logger
}

// Open connects with a pool of 2..10 and logs the server version at startup.
func Open(ctx context.Context, dsn string, log *slog.Logger) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse DATABASE_URL: %w", err)
	}
	cfg.MinConns = 2
	cfg.MaxConns = 10
	return openConfig(ctx, cfg, log)
}

func openConfig(ctx context.Context, cfg *pgxpool.Config, log *slog.Logger) (*Store, error) {
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create pool: %w", err)
	}
	var version string
	if err := pool.QueryRow(ctx, `SELECT version()`).Scan(&version); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect: %w", err)
	}
	log.Info("PostgreSQL connection pool created (" + version + ")")
	return &Store{pool: pool, log: log}, nil
}

// Close releases the pool.
func (s *Store) Close() {
	s.pool.Close()
	s.log.Info("PostgreSQL connection pool closed")
}

// Migrate applies every pending migration in filename order, exactly once,
// each in its own transaction, serialised by a session advisory lock on a
// dedicated connection. Idempotent: a second run applies nothing.
func (s *Store) Migrate(ctx context.Context) error {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire migration connection: %w", err)
	}
	defer conn.Release()

	lockCtx, cancelLock := context.WithTimeout(ctx, migrateLockTimeout)
	defer cancelLock()
	if _, err := conn.Exec(lockCtx, `SELECT pg_advisory_lock($1)`, migrateLockID); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("timed out after %s waiting for the migration lock (key %d in pg_locks): %w",
				migrateLockTimeout, migrateLockID, err)
		}
		return fmt.Errorf("acquire migration lock: %w", err)
	}
	defer func() {
		_, _ = conn.Exec(context.WithoutCancel(ctx), `SELECT pg_advisory_unlock($1)`, migrateLockID)
	}()

	if _, err := conn.Exec(ctx, `CREATE TABLE IF NOT EXISTS schema_migrations (
		name TEXT PRIMARY KEY,
		applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
	)`); err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	entries, err := migrationFS.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)

	for _, name := range names {
		var applied bool
		if err := conn.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)`, name).Scan(&applied); err != nil {
			return fmt.Errorf("check %s: %w", name, err)
		}
		if applied {
			continue
		}
		body, err := migrationFS.ReadFile("migrations/" + name)
		if err != nil {
			return fmt.Errorf("read %s: %w", name, err)
		}
		tx, err := conn.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin %s: %w", name, err)
		}
		if _, err := tx.Exec(ctx, string(body)); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("apply %s: %w", name, err)
		}
		if _, err := tx.Exec(ctx, `INSERT INTO schema_migrations (name) VALUES ($1)`, name); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("record %s: %w", name, err)
		}
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit %s: %w", name, err)
		}
		s.log.Info("migration applied: " + name)
	}
	return nil
}

// Ping is the startup connectivity check.
func (s *Store) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

// SchemaCurrent reports whether every embedded migration has been applied.
// The startup gate uses it so a deploy that skipped the migrate init
// container fails at boot with a name, not at the first query.
func (s *Store) SchemaCurrent(ctx context.Context) error {
	entries, err := migrationFS.ReadDir("migrations")
	if err != nil {
		return err
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT to_regclass('schema_migrations') IS NOT NULL`).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return errors.New("database schema is empty: run `noergler migrate` first")
	}
	var missing []string
	for _, e := range entries {
		var applied bool
		if err := s.pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)`, e.Name()).Scan(&applied); err != nil {
			return err
		}
		if !applied {
			missing = append(missing, e.Name())
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("database schema is behind, pending %v: run `noergler migrate` first", missing)
	}
	return nil
}
