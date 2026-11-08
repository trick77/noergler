package store

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jackc/pgx/v5/stdlib"
	"github.com/trick77/ship/kit/schemadoc"
)

// TestSchemaDoc keeps docs/schema.md in step with the migrations and the
// model.go row types. testStore migrates a fresh schema, the search_path
// current_schema() names, so the doc lists the migrations' tables only.
func TestSchemaDoc(t *testing.T) {
	// An asked-for rewrite without a database would skip and pass, leaving
	// the doc stale until the gate.
	if os.Getenv("SCHEMADOC_UPDATE") != "" && os.Getenv("SHIP_TEST_DSN") == "" {
		t.Fatal("SCHEMADOC_UPDATE needs SHIP_TEST_DSN (a Postgres to migrate)")
	}
	s := testStore(t)
	db := stdlib.OpenDBFromPool(s.pool)
	t.Cleanup(func() { _ = db.Close() })
	sch, err := schemadoc.Describe(t.Context(), db, schemadoc.Postgres)
	if err != nil {
		t.Fatal(err)
	}
	es, err := schemadoc.Entities(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	doc, err := schemadoc.Render(sch, es)
	if err != nil {
		t.Fatal(err)
	}
	schemadoc.Golden(t, doc, filepath.Join("..", "..", "..", "docs", "schema.md"))
}
