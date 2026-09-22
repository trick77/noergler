package bitbucket

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// pageLimit is the page size for every paged collection.
const pageLimit = 100

// CheckConnectivity proves the token works and logs which server answered.
// Any non-2xx is an error: this runs at startup and a failure aborts boot.
func (c *Client) CheckConnectivity(ctx context.Context) error {
	var props struct {
		Version     string `json:"version"`
		DisplayName string `json:"displayName"`
	}
	if err := c.do(ctx, http.MethodGet, apiBase+"/application-properties", nil, nil, &props); err != nil {
		return err
	}
	name, version := props.DisplayName, props.Version
	if name == "" {
		name = "?"
	}
	if version == "" {
		version = "?"
	}
	c.log.Info("Bitbucket Server " + name + " v" + version)
	return nil
}

// prPath is the base path of one pull request.
func prPath(project, repo string, prID int) string {
	return fmt.Sprintf("%s/projects/%s/repos/%s/pull-requests/%d", apiBase, project, repo, prID)
}

// FetchPRDiff returns the PR's full diff as text. contextLines is omitted
// entirely when not positive, which is what Bitbucket wants for a bare diff.
func (c *Client) FetchPRDiff(ctx context.Context, project, repo string, prID, contextLines int) (string, error) {
	var query url.Values
	if contextLines > 0 {
		query = url.Values{"contextLines": {strconv.Itoa(contextLines)}}
	}
	what := fmt.Sprintf("%s/%s#%d diff", project, repo, prID)
	return c.getTextCapped(ctx, prPath(project, repo, prID)+"/diff", query, "text/plain", what, c.maxDiffBytes)
}

// FetchCommitDiff returns the diff between two commits, for an incremental
// review. A 406 means the histories cannot be compared (usually a rebase) and
// comes back as ErrIncrementalDiffUnavailable so the caller can fall back to a
// full review instead of treating it as a transport failure.
func (c *Client) FetchCommitDiff(ctx context.Context, project, repo, fromCommit, toCommit string) (string, error) {
	path := fmt.Sprintf("%s/projects/%s/repos/%s/compare/diff", apiBase, project, repo)
	query := url.Values{"from": {fromCommit}, "to": {toCommit}}
	what := fmt.Sprintf("%s/%s compare %s..%s", project, repo, shortSHA(fromCommit), shortSHA(toCommit))

	text, err := c.getTextCapped(ctx, path, query, "text/plain", what, c.maxDiffBytes)
	if Status(err) == http.StatusNotAcceptable {
		return "", fmt.Errorf("compare/diff 406 for %s..%s (unreachable history, likely a rebase): %w",
			shortSHA(fromCommit), shortSHA(toCommit), ErrIncrementalDiffUnavailable)
	}
	return text, err
}

// FetchPRChanges returns the paths of every file in a PR, in the order
// Bitbucket reports them.
//
// This is the file list only, no diff bodies, so it stays cheap on a PR whose
// diff is far too large to fetch. That is its one caller: the too-large log
// path, where the cap tripped at the socket and the truncated head covers only
// the files that fit inside it.
//
// Decoding is deliberately tolerant. Bitbucket nests the path under
// `path.toString`; an entry that does not carry one is skipped rather than
// failing the call, so an unexpected shape degrades to a short list instead of
// turning a failure path into a second failure.
func (c *Client) FetchPRChanges(ctx context.Context, project, repo string, prID int) ([]string, error) {
	values, err := c.paged(ctx, prPath(project, repo, prID)+"/changes")
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(values))
	for _, v := range values {
		// Directories carry nodeType DIRECTORY; only files have diffs.
		if nodeType, ok := v["nodeType"].(string); ok && !strings.EqualFold(nodeType, "FILE") {
			continue
		}
		path, _ := v["path"].(map[string]any)
		if path == nil {
			continue
		}
		if s, ok := path["toString"].(string); ok && s != "" {
			paths = append(paths, s)
		}
	}
	// Entries that all failed to parse mean the response shape is not what
	// this assumes. Erroring is the point: returning an empty list with no
	// error would have the caller report head-only scope as if the call had
	// never been made, hiding the very breakage the caller exists to surface.
	if len(values) > 0 && len(paths) == 0 {
		return nil, fmt.Errorf("changes: %d entries, none carried path.toString", len(values))
	}
	return paths, nil
}

func shortSHA(s string) string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

// FetchFileContent returns one file at one commit. Unlike the diff endpoints
// this goes out with the default JSON Accept.
func (c *Client) FetchFileContent(ctx context.Context, project, repo, commit, path string) (string, error) {
	raw := fmt.Sprintf("%s/projects/%s/repos/%s/raw/%s", apiBase, project, repo, path)
	return c.getTextCapped(ctx, raw, url.Values{"at": {commit}}, "", path, c.maxFileBytes)
}

// PostInlineComment anchors a finished comment body to one line of one file.
//
// body is rendered by the caller: this client decides only where the comment
// goes. Bitbucket rejects an ADDED anchor on a line the diff shows as context,
// and there is no way to tell which from the diff alone, so a 400 is retried
// once as CONTEXT. Only a 400 retries; anything else fails on the first try.
func (c *Client) PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error) {
	path := prPath(project, repo, prID) + "/comments"
	anchorPath := stripDiffPrefix(file)

	var lastErr error
	for _, lineType := range [...]string{"ADDED", "CONTEXT"} {
		payload := map[string]any{
			"text": body,
			"anchor": map[string]any{
				"path":     anchorPath,
				"line":     line,
				"fileType": "TO",
				"lineType": lineType,
			},
		}
		var created struct {
			ID int `json:"id"`
		}
		err := c.do(ctx, http.MethodPost, path, nil, payload, &created)
		if err == nil {
			c.log.InfoContext(ctx, fmt.Sprintf("Posted inline comment on %s:%d", file, line))
			return created.ID, nil
		}
		lastErr = err
		if Status(err) != http.StatusBadRequest {
			return 0, err
		}
	}
	// Both anchors rejected: log what Bitbucket objected to, then fail. The
	// caller counts the miss and carries on with the other findings.
	c.log.WarnContext(ctx, fmt.Sprintf("Inline comment rejected for %s:%d: %v", file, line, lastErr))
	return 0, lastErr
}

// stripDiffPrefix removes one leading "a/" or "b/" from a diff path. Only the
// first is removed: a real file called "b/thing" under "a/" keeps its name.
func stripDiffPrefix(p string) string {
	if strings.HasPrefix(p, "a/") || strings.HasPrefix(p, "b/") {
		return p[2:]
	}
	return p
}

// PostPRComment posts a top-level comment and returns its id and version. The
// version is needed to edit it later; Bitbucket omits it on a fresh comment,
// which means 0.
func (c *Client) PostPRComment(ctx context.Context, project, repo string, prID int, text string) (id, version int, err error) {
	var created struct {
		ID      int `json:"id"`
		Version int `json:"version"`
	}
	if err := c.do(ctx, http.MethodPost, prPath(project, repo, prID)+"/comments", nil,
		map[string]any{"text": text}, &created); err != nil {
		return 0, 0, err
	}
	c.log.InfoContext(ctx, fmt.Sprintf("Posted summary comment on PR %d", prID))
	return created.ID, created.Version, nil
}

// ReplyToComment answers one comment in its thread.
func (c *Client) ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) error {
	payload := map[string]any{"text": text, "parent": map[string]any{"id": parentCommentID}}
	if err := c.do(ctx, http.MethodPost, prPath(project, repo, prID)+"/comments", nil, payload, nil); err != nil {
		return err
	}
	c.log.InfoContext(ctx, fmt.Sprintf("Replied to comment %d on PR %d", parentCommentID, prID))
	return nil
}

// Comment is one PR comment as Bitbucket returns it.
type Comment struct {
	ID      int    `json:"id"`
	Version int    `json:"version"`
	Text    string `json:"text"`
}

// FetchPRComment reads one comment, for its current text and version.
func (c *Client) FetchPRComment(ctx context.Context, project, repo string, prID, commentID int) (*Comment, error) {
	var out Comment
	path := fmt.Sprintf("%s/comments/%d", prPath(project, repo, prID), commentID)
	if err := c.do(ctx, http.MethodGet, path, nil, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// UpdatePRComment edits a comment, holding the version it was read at. A
// concurrent edit (409) returns ErrVersionConflict: the caller posts a new
// comment rather than clobbering whatever landed.
func (c *Client) UpdatePRComment(ctx context.Context, project, repo string, prID, commentID, version int, text string) (int, error) {
	path := fmt.Sprintf("%s/comments/%d", prPath(project, repo, prID), commentID)
	var updated struct {
		Version int `json:"version"`
	}
	err := c.do(ctx, http.MethodPut, path, nil, map[string]any{"text": text, "version": version}, &updated)
	if Status(err) == http.StatusConflict {
		c.log.WarnContext(ctx, fmt.Sprintf(
			"Version conflict updating comment %d on PR %d, falling back to a new comment", commentID, prID))
		return 0, ErrVersionConflict
	}
	if err != nil {
		return 0, err
	}
	c.log.InfoContext(ctx, fmt.Sprintf("Updated summary comment %d on PR %d", commentID, prID))
	return updated.Version, nil
}

// targetPath addresses a project (repo == "") or one repo inside it. Project
// webhooks (Bitbucket DC 8.8+) share body shape and paging with repo webhooks;
// only the path differs.
func targetPath(project, repo string) string {
	base := apiBase + "/projects/" + project
	if repo == "" {
		return base
	}
	return base + "/repos/" + repo
}

// page is one page of a paged collection.
type page struct {
	Values        []map[string]any `json:"values"`
	IsLastPage    *bool            `json:"isLastPage"`
	NextPageStart *int             `json:"nextPageStart"`
}

// paged walks every page of a collection. It stops on isLastPage (a response
// without the field counts as the last), and also when nextPageStart fails to
// advance, so a server echoing an offset cannot spin this forever.
func (c *Client) paged(ctx context.Context, path string) ([]map[string]any, error) {
	var out []map[string]any
	start := 0
	for {
		var p page
		query := url.Values{"start": {strconv.Itoa(start)}, "limit": {strconv.Itoa(pageLimit)}}
		if err := c.do(ctx, http.MethodGet, path, query, nil, &p); err != nil {
			return nil, err
		}
		out = append(out, p.Values...)
		if p.IsLastPage == nil || *p.IsLastPage {
			return out, nil
		}
		if p.NextPageStart == nil || *p.NextPageStart <= start {
			return out, nil
		}
		start = *p.NextPageStart
	}
}

// ListRepos returns every repository in a project.
func (c *Client) ListRepos(ctx context.Context, project string) ([]map[string]any, error) {
	return c.paged(ctx, apiBase+"/projects/"+project+"/repos")
}

// Webhook is a webhook as Bitbucket stores it.
type Webhook struct {
	ID            int      `json:"id,omitempty"`
	Name          string   `json:"name"`
	URL           string   `json:"url"`
	Active        bool     `json:"active"`
	Events        []string `json:"events"`
	Configuration struct {
		Secret string `json:"secret,omitempty"`
	} `json:"configuration"`
	SSLVerificationRequired bool `json:"sslVerificationRequired"`
}

// ListWebhooks returns the webhooks on a project or one repo.
func (c *Client) ListWebhooks(ctx context.Context, project, repo string) ([]map[string]any, error) {
	return c.paged(ctx, targetPath(project, repo)+"/webhooks")
}

// CreateWebhook registers a webhook and returns it with its assigned id.
func (c *Client) CreateWebhook(ctx context.Context, project, repo string, body Webhook) (*Webhook, error) {
	var out Webhook
	if err := c.do(ctx, http.MethodPost, targetPath(project, repo)+"/webhooks", nil, body, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// UpdateWebhook replaces a webhook's settings.
func (c *Client) UpdateWebhook(ctx context.Context, project, repo string, webhookID int, body Webhook) (*Webhook, error) {
	var out Webhook
	path := fmt.Sprintf("%s/webhooks/%d", targetPath(project, repo), webhookID)
	if err := c.do(ctx, http.MethodPut, path, nil, body, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteWebhook removes a webhook.
func (c *Client) DeleteWebhook(ctx context.Context, project, repo string, webhookID int) error {
	path := fmt.Sprintf("%s/webhooks/%d", targetPath(project, repo), webhookID)
	return c.do(ctx, http.MethodDelete, path, nil, nil, nil)
}

// GrantUserPermission gives username a permission (PROJECT_WRITE, REPO_WRITE)
// on the target. Needs admin rights there. Bitbucket takes this as query
// parameters, not a body.
func (c *Client) GrantUserPermission(ctx context.Context, project, repo, username, permission string) error {
	query := url.Values{"name": {username}, "permission": {permission}}
	return c.do(ctx, http.MethodPut, targetPath(project, repo)+"/permissions/users", query, nil, nil)
}

// UserPermission is a user's effective permission on a target, "" when the
// user has none there.
//
// The names are Bitbucket's: PROJECT_READ / PROJECT_WRITE / PROJECT_ADMIN on
// a project, REPO_READ / REPO_WRITE / REPO_ADMIN on a repository.
type UserPermission string

// CanWrite reports whether the permission allows posting a comment. ADMIN
// includes WRITE, so both pass; READ and "" do not.
func (p UserPermission) CanWrite() bool {
	return strings.HasSuffix(string(p), "_WRITE") || strings.HasSuffix(string(p), "_ADMIN")
}

// UserPermissionOn reads username's effective permission on the target.
//
// noergler needs WRITE, not read: it posts review comments. Proving the bot
// can GET the repository proves nothing about that, which is why this exists
// beside GetRepo rather than instead of it.
//
// Bitbucket's `filter` is a substring match over users, so the response can
// carry several rows and the one asked for may not be first. The username is
// compared exactly, case-insensitively, and anything else is ignored.
func (c *Client) UserPermissionOn(ctx context.Context, project, repo, username string) (UserPermission, error) {
	var out struct {
		Values []struct {
			User struct {
				Name string `json:"name"`
			} `json:"user"`
			Permission string `json:"permission"`
		} `json:"values"`
	}
	query := url.Values{"filter": {username}, "limit": {"100"}}
	path := targetPath(project, repo) + "/permissions/users"
	if err := c.do(ctx, http.MethodGet, path, query, nil, &out); err != nil {
		return "", err
	}
	for _, v := range out.Values {
		if strings.EqualFold(v.User.Name, username) {
			return UserPermission(v.Permission), nil
		}
	}
	// No row for this user is not an error: it is the answer, and it means
	// the bot has no permission of its own on the target.
	return "", nil
}

// GetProject reads a project, to prove it exists and is readable.
func (c *Client) GetProject(ctx context.Context, project string) (map[string]any, error) {
	var out map[string]any
	if err := c.do(ctx, http.MethodGet, apiBase+"/projects/"+project, nil, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// GetRepo reads one repository.
func (c *Client) GetRepo(ctx context.Context, project, repo string) (map[string]any, error) {
	var out map[string]any
	if err := c.do(ctx, http.MethodGet, apiBase+"/projects/"+project+"/repos/"+repo, nil, nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}
