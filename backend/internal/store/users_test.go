package store

import (
	"context"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

// A user's language is keyed by the lower-cased name: Bitbucket matches
// authors case-insensitively, so "Alice" and "alice" are one person.
func TestUserLanguage_RoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	if got, err := s.UserLanguage(ctx, "alice"); err != nil || got != "" {
		t.Fatalf("unset = %q, %v, want empty", got, err)
	}
	if err := s.SetUserLanguage(ctx, "Alice", lang.EN); err != nil {
		t.Fatal(err)
	}
	if got, err := s.UserLanguage(ctx, "alice"); err != nil || got != lang.EN {
		t.Fatalf("after set = %q, %v, want en", got, err)
	}
	if err := s.SetUserLanguage(ctx, "alice", lang.DE); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.UserLanguage(ctx, "ALICE"); got != lang.DE {
		t.Fatalf("after overwrite = %q, want de", got)
	}
	if _, err := s.pool.Exec(ctx, `INSERT INTO user_settings (user_name, language) VALUES ('bob', 'fr')`); err == nil {
		t.Error("the table accepted a language noergler does not write")
	}
}
