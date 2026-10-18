package store

import (
	"context"
	"errors"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/trick77/noergler/internal/lang"
)

// UserLanguage is the user's own language, "" when they never set one.
func (s *Store) UserLanguage(ctx context.Context, userName string) (lang.Lang, error) {
	var v string
	err := s.pool.QueryRow(ctx, `SELECT language FROM user_settings WHERE user_name = $1`,
		strings.ToLower(userName)).Scan(&v)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	l, _ := lang.Parse(v)
	return l, nil
}

// SetUserLanguage upserts the user's language.
func (s *Store) SetUserLanguage(ctx context.Context, userName string, l lang.Lang) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO user_settings (user_name, language, updated_at) VALUES ($1, $2, now())
		ON CONFLICT (user_name) DO UPDATE SET language = EXCLUDED.language, updated_at = now()`,
		strings.ToLower(userName), string(l))
	return err
}
