package store

import "context"

// VisualPair identifies one screenshot pair by content, migrations/
// 0023_visual_changes.sql. BeforeID is "" for an added screenshot.
type VisualPair struct {
	Path     string
	BeforeID string
	AfterID  string
}

// VisualChanges is every line stored for an open PR's screenshot pairs; ""
// for a pair stored as unchanged pixels.
func (s *Store) VisualChanges(ctx context.Context, k PRKey) (map[VisualPair]string, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT v.path, v.before_id, v.after_id, v.change_text
		FROM visual_changes v JOIN pull_requests p ON v.pull_request_id = p.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND p.`+activeFilter,
		k.Project, k.Repo, k.PRID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[VisualPair]string{}
	for rows.Next() {
		var p VisualPair
		var text string
		if err := rows.Scan(&p.Path, &p.BeforeID, &p.AfterID, &text); err != nil {
			return nil, err
		}
		out[p] = text
	}
	return out, rows.Err()
}

// InsertVisualChange stores what the model said about one pair, "" for
// unchanged pixels. The first line stored for a pair stands: a rerun of the
// same pair is never sent.
func (s *Store) InsertVisualChange(ctx context.Context, k PRKey, p VisualPair, text, model string) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO visual_changes (pull_request_id, path, before_id, after_id, change_text, model)
		SELECT id, $4, $5, $6, $7, $8 FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
		ON CONFLICT DO NOTHING`,
		k.Project, k.Repo, k.PRID, p.Path, p.BeforeID, p.AfterID, text, model)
	return err
}
