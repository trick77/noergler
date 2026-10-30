package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// SealedCredential is one team_credentials row: the token as ciphertext only.
// Sealing and opening are the caller's (internal/teams); the store never sees
// a token in the clear.
type SealedCredential struct {
	Team       string
	Kind       string
	Username   string
	Nonce      []byte
	Ciphertext []byte
}

// ListCredentials returns every team's pushed credentials.
func (s *Store) ListCredentials(ctx context.Context) ([]SealedCredential, error) {
	return queryAll(ctx, s.pool, `SELECT team_slug, kind, COALESCE(username, ''), nonce, ciphertext FROM team_credentials ORDER BY team_slug, kind`,
		nil, func(row pgx.CollectableRow, c *SealedCredential) error {
			return row.Scan(&c.Team, &c.Kind, &c.Username, &c.Nonce, &c.Ciphertext)
		})
}

// PutCredentials upserts team credentials, all or none: a push of both kinds
// never lands half. The log lines name the team, the kind and the username,
// never the token.
func (s *Store) PutCredentials(ctx context.Context, creds []SealedCredential, updatedBy string) error {
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		for _, c := range creds {
			var username *string
			if c.Username != "" {
				username = &c.Username
			}
			if _, err := tx.Exec(ctx, `
			INSERT INTO team_credentials (team_slug, kind, username, nonce, ciphertext, updated_by, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, now())
			ON CONFLICT (team_slug, kind) DO UPDATE SET
				username = EXCLUDED.username,
				nonce = EXCLUDED.nonce,
				ciphertext = EXCLUDED.ciphertext,
				updated_by = EXCLUDED.updated_by,
				updated_at = now()`,
				c.Team, c.Kind, username, c.Nonce, c.Ciphertext, updatedBy); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	for _, c := range creds {
		s.log.Info(fmt.Sprintf("team credentials updated team=%s kind=%s username=%s by=%s",
			c.Team, c.Kind, c.Username, updatedBy), "team", c.Team)
	}
	return nil
}

// DeleteCredential removes one team credential; false when there was none.
func (s *Store) DeleteCredential(ctx context.Context, team, kind, updatedBy string) (bool, error) {
	tag, err := s.pool.Exec(ctx, `DELETE FROM team_credentials WHERE team_slug = $1 AND kind = $2`, team, kind)
	if err != nil {
		return false, err
	}
	if tag.RowsAffected() == 0 {
		return false, nil
	}
	s.log.Info(fmt.Sprintf("team credentials removed team=%s kind=%s by=%s", team, kind, updatedBy), "team", team)
	return true, nil
}

// KeyCheck reads the sealed key-check value; found false when none is stored.
func (s *Store) KeyCheck(ctx context.Context) (nonce, ciphertext []byte, found bool, err error) {
	err = s.pool.QueryRow(ctx, `SELECT nonce, ciphertext FROM secrets_key_check`).Scan(&nonce, &ciphertext)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil, false, nil
	}
	if err != nil {
		return nil, nil, false, err
	}
	return nonce, ciphertext, true, nil
}

// PutKeyCheck stores the sealed key-check value, replacing any.
func (s *Store) PutKeyCheck(ctx context.Context, nonce, ciphertext []byte) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO secrets_key_check (id, nonce, ciphertext) VALUES (TRUE, $1, $2)
		ON CONFLICT (id) DO UPDATE SET nonce = EXCLUDED.nonce, ciphertext = EXCLUDED.ciphertext`,
		nonce, ciphertext)
	return err
}
