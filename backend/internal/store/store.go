// Package store is the Postgres layer: pool, embedded migrations and every
// query the service runs. One function per query, positional pgx args, and a
// transaction wherever more than one statement must land together.
package store

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log/slog"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/trick77/ship/kit/migrate"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// migrateLockID is the advisory-lock key held for the whole of Migrate. Fixed:
// pg_advisory_lock namespaces by value alone. From "noer".
const migrateLockID int64 = 0x6E6F6572

// migrations is the embedded SQL with the directory stripped, as kit's
// migrate reads it.
func migrations() fs.FS {
	sub, err := fs.Sub(migrationFS, "migrations")
	if err != nil {
		panic(err) // unreachable: the embed pattern guarantees the directory
	}
	return sub
}

// Store owns the pool.
type Store struct {
	pool *pgxpool.Pool
	log  *slog.Logger
}

// queryAll runs sql and reads every row through scan into a slice. No rows
// is a nil slice, as the hand loops it replaced left it (a JSON null, never
// []); pgx.CollectRows would start from []T{}, which is why it is not used.
func queryAll[T any](ctx context.Context, pool *pgxpool.Pool, sql string, args []any,
	scan func(row pgx.CollectableRow, v *T) error) ([]T, error) {
	rows, err := pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	return pgx.AppendRows([]T(nil), rows, func(row pgx.CollectableRow) (T, error) {
		var v T
		return v, scan(row, &v)
	})
}

// inTx runs fn in one transaction: committed when fn returns nil, rolled
// back otherwise.
func (s *Store) inTx(ctx context.Context, fn func(tx pgx.Tx) error) error {
	return pgx.BeginFunc(ctx, s.pool, fn)
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
	// date_trunc('day') buckets in the session TimeZone. Pinned, so a server
	// configured for another zone cannot label every daily bucket a day off;
	// the dashboard computes its windows in UTC to match.
	cfg.ConnConfig.RuntimeParams["timezone"] = "UTC"
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

// Migrate applies every pending migration in file order, exactly once, each
// in its own transaction, serialised by a session advisory lock on a
// dedicated connection. Idempotent: a second run applies nothing. Same
// schema_migrations table as before kit, so no data step.
func (s *Store) Migrate(ctx context.Context) error {
	db := stdlib.OpenDBFromPool(s.pool)
	defer func() { _ = db.Close() }()
	_, err := migrate.Up(ctx, db, migrations(), s.migrateOptions())
	return err
}

func (s *Store) migrateOptions() migrate.Options {
	return migrate.Options{Dialect: migrate.Postgres(migrateLockID), Log: s.log}
}

// Ping is the startup connectivity check.
func (s *Store) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

// SchemaCurrent reports whether every embedded migration has been applied.
// The startup gate uses it so a deploy that skipped the migrate init
// container fails at boot with a name, not at the first query.
func (s *Store) SchemaCurrent(ctx context.Context) error {
	db := stdlib.OpenDBFromPool(s.pool)
	defer func() { _ = db.Close() }()
	pending, err := migrate.Pending(ctx, db, migrations(), s.migrateOptions())
	if err != nil {
		return err
	}
	if len(pending) > 0 {
		return fmt.Errorf("database schema is behind, pending %v: run `noergler migrate` first", pending)
	}
	return nil
}
