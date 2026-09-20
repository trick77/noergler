package store

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/trick77/noergler-go/internal/config"
)

// What a team changes on its own: project/repo claims and the review lists.
// teams.yaml only seeds an empty DB.
//
// Ownership guarantee: team_claims holds a project or a repo for exactly one
// team (partial unique indexes). The one overlap the indexes cannot express,
// a whole-project claim against another team's repo claims on that project,
// is checked inside a transaction that locks the project's rows.

// ClaimConflict: the target is held by another team; nothing was written.
type ClaimConflict struct {
	Project   string
	Repo      *string
	OtherTeam string
}

func (c *ClaimConflict) Error() string {
	target := c.Project
	if c.Repo != nil {
		target += "/" + *c.Repo
	}
	return fmt.Sprintf("%s is claimed by team %s", target, c.OtherTeam)
}

// TeamSettings are the per-team review lists held in the DB.
type TeamSettings struct {
	AutoReviewAuthors []string
	IgnoreAuthors     []string
	ExcludeRepos      []string
}

// Excludes is the case-insensitive glob match of a repo slug.
func (t TeamSettings) Excludes(repoSlug string) bool {
	return config.ExcludesRepo(t.ExcludeRepos, repoSlug)
}

type claimRow struct {
	team string
	key  string
	repo *string
}

// scopesFromRows turns (project_key, repo_slug) rows into scopes: whole
// projects first, repo claims grouped per project, both in first-claimed
// order.
func scopesFromRows(rows []claimRow) []config.ProjectScope {
	var whole []string
	wholeSet := map[string]bool{}
	var repoOrder []string
	repos := map[string][]string{}
	for _, r := range rows {
		if r.repo == nil {
			if !wholeSet[r.key] {
				wholeSet[r.key] = true
				whole = append(whole, r.key)
			}
			continue
		}
		if _, seen := repos[r.key]; !seen {
			repoOrder = append(repoOrder, r.key)
		}
		repos[r.key] = append(repos[r.key], *r.repo)
	}
	scopes := make([]config.ProjectScope, 0, len(whole)+len(repoOrder))
	for _, k := range whole {
		scopes = append(scopes, config.ProjectScope{Key: k})
	}
	for _, k := range repoOrder {
		if !wholeSet[k] {
			scopes = append(scopes, config.ProjectScope{Key: k, Repos: repos[k]})
		}
	}
	return scopes
}

// ListClaims returns the team's scopes.
func (s *Store) ListClaims(ctx context.Context, teamSlug string) ([]config.ProjectScope, error) {
	rows, err := s.pool.Query(ctx, `SELECT project_key, repo_slug FROM team_claims WHERE team_slug = $1 ORDER BY id`, teamSlug)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []claimRow
	for rows.Next() {
		var r claimRow
		if err := rows.Scan(&r.key, &r.repo); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return scopesFromRows(out), rows.Err()
}

// ListAllClaims returns every team's scopes keyed by slug.
func (s *Store) ListAllClaims(ctx context.Context) (map[string][]config.ProjectScope, error) {
	rows, err := s.pool.Query(ctx, `SELECT team_slug, project_key, repo_slug FROM team_claims ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	byTeam := map[string][]claimRow{}
	for rows.Next() {
		var r claimRow
		if err := rows.Scan(&r.team, &r.key, &r.repo); err != nil {
			return nil, err
		}
		byTeam[r.team] = append(byTeam[r.team], r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	out := make(map[string][]config.ProjectScope, len(byTeam))
	for slug, rs := range byTeam {
		out[slug] = scopesFromRows(rs)
	}
	return out, nil
}

// AddClaims claims scopes for the team. All or nothing: a conflict with
// another team returns *ClaimConflict and writes nothing. Returns the targets
// that are new (`KEY` or `KEY/repo`); a repo already covered by the team's
// own whole-project claim is not new, a whole-project claim replaces the
// team's own repo claims on that project.
func (s *Store) AddClaims(ctx context.Context, teamSlug string, scopes []config.ProjectScope, claimedBy string) ([]string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	added := []string{}
	if err := s.addClaims(ctx, tx, teamSlug, scopes, claimedBy, &added); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	if len(added) > 0 {
		s.log.Info(fmt.Sprintf("claims added team=%s by=%s targets=%v", teamSlug, claimedBy, added), "team", teamSlug)
	}
	return added, nil
}

func (s *Store) addClaims(ctx context.Context, tx pgx.Tx, teamSlug string, scopes []config.ProjectScope, claimedBy string, added *[]string) (err error) {
	// FOR UPDATE locks nothing when the project has no rows yet, and under
	// READ COMMITTED a waiter does not see rows another transaction inserted
	// after its own read. A whole-project claim racing a repo claim on an
	// unclaimed project would then both commit, distinct keys on both partial
	// indexes. One transaction-scoped advisory lock per project key, taken
	// in sorted order (no deadlock between callers naming the same keys),
	// serialises the check and the inserts whatever the table holds.
	keys := make([]string, 0, len(scopes))
	for _, scope := range scopes {
		keys = append(keys, scope.Key)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtext('team_claims:' || $1))`, k); err != nil {
			return err
		}
	}
	var current config.ProjectScope
	defer func() {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			// The index decided against a claim that slipped past the lock
			// (a transaction from before this build, or a hand-written row).
			// State is fine, the transaction rolls back; name the scope that
			// lost rather than the first one.
			err = &ClaimConflict{Project: current.Key, OtherTeam: "another team (concurrent claim)"}
		}
	}()
	for _, scope := range scopes {
		current = scope
		rows, err := tx.Query(ctx, `SELECT team_slug, repo_slug FROM team_claims WHERE project_key = $1 FOR UPDATE`, scope.Key)
		if err != nil {
			return err
		}
		var held []claimRow
		for rows.Next() {
			var r claimRow
			if err := rows.Scan(&r.team, &r.repo); err != nil {
				rows.Close()
				return err
			}
			held = append(held, r)
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return err
		}

		if scope.Repos == nil {
			ownWhole := false
			for _, r := range held {
				if r.team != teamSlug {
					return &ClaimConflict{Project: scope.Key, Repo: r.repo, OtherTeam: r.team}
				}
				if r.repo == nil {
					ownWhole = true
				}
			}
			if ownWhole {
				continue // already ours as a whole
			}
			// Our repo claims on this project are covered by the project
			// claim now.
			if _, err := tx.Exec(ctx, `DELETE FROM team_claims WHERE project_key = $1 AND team_slug = $2`, scope.Key, teamSlug); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, NULL, $3)`,
				teamSlug, scope.Key, claimedBy); err != nil {
				return err
			}
			*added = append(*added, scope.Key)
			continue
		}

		var whole *claimRow
		owners := map[string]string{}
		for i := range held {
			if held[i].repo == nil {
				whole = &held[i]
			} else {
				owners[*held[i].repo] = held[i].team
			}
		}
		if whole != nil && whole.team != teamSlug {
			return &ClaimConflict{Project: scope.Key, OtherTeam: whole.team}
		}
		if whole != nil {
			continue // our whole-project claim already covers every repo
		}
		for _, repo := range scope.Repos {
			owner, held := owners[repo]
			if held && owner != teamSlug {
				r := repo
				return &ClaimConflict{Project: scope.Key, Repo: &r, OtherTeam: owner}
			}
			if held {
				continue
			}
			if _, err := tx.Exec(ctx, `INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, $3, $4)`,
				teamSlug, scope.Key, repo, claimedBy); err != nil {
				return err
			}
			*added = append(*added, scope.Key+"/"+repo)
		}
	}
	return nil
}

// RemoveClaims drops the team's claims named in scopes. A whole-project scope
// drops the project claim and any repo claims of the team on it. Returns what
// was dropped.
func (s *Store) RemoveClaims(ctx context.Context, teamSlug string, scopes []config.ProjectScope) ([]string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	removed := []string{}
	for _, scope := range scopes {
		if scope.Repos == nil {
			rows, err := tx.Query(ctx, `DELETE FROM team_claims WHERE team_slug = $1 AND project_key = $2 RETURNING repo_slug`, teamSlug, scope.Key)
			if err != nil {
				return nil, err
			}
			for rows.Next() {
				var repo *string
				if err := rows.Scan(&repo); err != nil {
					rows.Close()
					return nil, err
				}
				if repo == nil {
					removed = append(removed, scope.Key)
				} else {
					removed = append(removed, scope.Key+"/"+*repo)
				}
			}
			rows.Close()
			if err := rows.Err(); err != nil {
				return nil, err
			}
			continue
		}
		for _, repo := range scope.Repos {
			tag, err := tx.Exec(ctx, `DELETE FROM team_claims WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3`, teamSlug, scope.Key, repo)
			if err != nil {
				return nil, err
			}
			if tag.RowsAffected() > 0 {
				removed = append(removed, scope.Key+"/"+repo)
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	if len(removed) > 0 {
		s.log.Info(fmt.Sprintf("claims removed team=%s targets=%v", teamSlug, removed), "team", teamSlug)
	}
	return removed, nil
}

// PurgeProject deletes every PR record the team holds on the target (runs
// and findings cascade). The team guard keeps a team from purging data
// another team wrote on a project that changed hands. Returns the count.
func (s *Store) PurgeProject(ctx context.Context, teamSlug, projectKey string, repoSlug *string) (int, error) {
	var tag pgconn.CommandTag
	var err error
	if repoSlug == nil {
		tag, err = s.pool.Exec(ctx, `DELETE FROM pull_requests WHERE team_slug = $1 AND project_key = $2`, teamSlug, projectKey)
	} else {
		tag, err = s.pool.Exec(ctx, `DELETE FROM pull_requests WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3`, teamSlug, projectKey, *repoSlug)
	}
	if err != nil {
		return 0, err
	}
	n := int(tag.RowsAffected())
	if n > 0 {
		target := projectKey + "/*"
		if repoSlug != nil {
			target = projectKey + "/" + *repoSlug
		}
		s.log.Info(fmt.Sprintf("purged %d PR record(s) team=%s target=%s", n, teamSlug, target), "team", teamSlug)
	}
	return n, nil
}

// CountProjectPRs is what PurgeProject would drop (for dry runs).
func (s *Store) CountProjectPRs(ctx context.Context, teamSlug, projectKey string, repoSlug *string) (int, error) {
	var n int
	var err error
	if repoSlug == nil {
		err = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM pull_requests WHERE team_slug = $1 AND project_key = $2`, teamSlug, projectKey).Scan(&n)
	} else {
		err = s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM pull_requests WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3`, teamSlug, projectKey, *repoSlug).Scan(&n)
	}
	return n, err
}

// GetSettings returns the team's row, or nil when the DB has none.
func (s *Store) GetSettings(ctx context.Context, teamSlug string) (*TeamSettings, error) {
	var t TeamSettings
	err := s.pool.QueryRow(ctx, `SELECT auto_review_authors, ignore_authors, exclude_repos FROM team_settings WHERE team_slug = $1`, teamSlug).
		Scan(&t.AutoReviewAuthors, &t.IgnoreAuthors, &t.ExcludeRepos)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &t, nil
}

// GetAllSettings returns every team's row keyed by slug.
func (s *Store) GetAllSettings(ctx context.Context) (map[string]TeamSettings, error) {
	rows, err := s.pool.Query(ctx, `SELECT team_slug, auto_review_authors, ignore_authors, exclude_repos FROM team_settings`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]TeamSettings{}
	for rows.Next() {
		var slug string
		var t TeamSettings
		if err := rows.Scan(&slug, &t.AutoReviewAuthors, &t.IgnoreAuthors, &t.ExcludeRepos); err != nil {
			return nil, err
		}
		out[slug] = t
	}
	return out, rows.Err()
}

// PutSettings upserts the team's lists.
func (s *Store) PutSettings(ctx context.Context, teamSlug string, t TeamSettings, updatedBy string) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO team_settings (team_slug, auto_review_authors, ignore_authors, exclude_repos, updated_by, updated_at)
		VALUES ($1, $2, $3, $4, $5, now())
		ON CONFLICT (team_slug) DO UPDATE SET
			auto_review_authors = EXCLUDED.auto_review_authors,
			ignore_authors = EXCLUDED.ignore_authors,
			exclude_repos = EXCLUDED.exclude_repos,
			updated_by = EXCLUDED.updated_by,
			updated_at = now()`,
		teamSlug, nonNil(t.AutoReviewAuthors), nonNil(t.IgnoreAuthors), nonNil(t.ExcludeRepos), updatedBy)
	if err != nil {
		return err
	}
	s.log.Info(fmt.Sprintf("settings updated team=%s by=%s auto_review_authors=%v ignore_authors=%v exclude_repos=%v",
		teamSlug, updatedBy, t.AutoReviewAuthors, t.IgnoreAuthors, t.ExcludeRepos), "team", teamSlug)
	return nil
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
