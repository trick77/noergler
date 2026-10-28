package bitbucket

import (
	"context"
	"fmt"
	"io"
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

// Identity proves the token and returns the account Bitbucket says it belongs
// to: the X-AUSERNAME header, else the applinks whoami servlet (plain text),
// "" when neither names one (a proxy stripping the header, the servlet
// absent). Logs nothing; the caller decides what a failure means.
func (c *Client) Identity(ctx context.Context) (string, error) {
	path := apiBase + "/application-properties"
	who, err := c.identityGet(ctx, path, "application/json")
	if err != nil {
		return "", err
	}
	if who.header != "" {
		return who.header, nil
	}
	// Only a name: a failure here proves nothing the call above did not.
	if w, err := c.identityGet(ctx, "/plugins/servlet/applinks/whoami", "text/plain"); err == nil {
		return strings.TrimSpace(w.body), nil
	}
	return "", nil
}

type identityAnswer struct{ header, body string }

// identityGet GETs path for Identity: its X-AUSERNAME and its first 256 bytes.
func (c *Client) identityGet(ctx context.Context, path, accept string) (identityAnswer, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url(path, nil), nil)
	if err != nil {
		return identityAnswer{}, err
	}
	req.Header.Set("Accept", accept)
	resp, err := c.send(req)
	if err != nil {
		return identityAnswer{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return identityAnswer{}, statusError(http.MethodGet, path, resp)
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
	_, _ = io.Copy(io.Discard, resp.Body)
	return identityAnswer{header: resp.Header.Get("X-AUSERNAME"), body: string(body)}, nil
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

// FetchCommitDiff returns what one push changed on a PR: the PR diff from
// fromCommit to toCommit, raw text like FetchPRDiff.
//
// The diff holds this push's lines only while fromCommit is still one of the
// PR's commits and no merge commit lies between the two. A rebase or force
// push drops fromCommit; a merge of the target branch brings its lines in.
// Either way a diff would pass upstream lines off as the push's, evidence
// for a wrong "fixed", so both come back as ErrIncrementalDiffUnavailable
// and the caller reviews the whole PR. Not compare/diff: it serves JSON only
// and answers text/plain with 406, every time.
func (c *Client) FetchCommitDiff(ctx context.Context, project, repo string, prID int, fromCommit, toCommit string) (string, error) {
	onPR, err := c.prHasCommit(ctx, project, repo, prID, fromCommit)
	if err != nil {
		return "", err
	}
	if !onPR {
		return "", fmt.Errorf("%s is no longer a commit of the PR (history rewritten): %w",
			shortSHA(fromCommit), ErrIncrementalDiffUnavailable)
	}
	merge, err := c.firstMerge(ctx, project, repo, fromCommit, toCommit)
	if err != nil {
		return "", err
	}
	if merge != "" {
		return "", fmt.Errorf("merge commit %s in %s..%s: %w",
			shortSHA(merge), shortSHA(fromCommit), shortSHA(toCommit), ErrIncrementalDiffUnavailable)
	}
	query := url.Values{"sinceId": {fromCommit}, "untilId": {toCommit}}
	what := fmt.Sprintf("%s/%s#%d diff %s..%s", project, repo, prID, shortSHA(fromCommit), shortSHA(toCommit))
	return c.getTextCapped(ctx, prPath(project, repo, prID)+"/diff", query, "text/plain", what, c.maxDiffBytes)
}

// prHasCommit reports whether commit is among the PR's commits, reading no
// page past the one that has it. Exact match: both sides are full SHAs, and
// a prefix could match an unrelated commit.
func (c *Client) prHasCommit(ctx context.Context, project, repo string, prID int, commit string) (bool, error) {
	onPR := false
	err := c.scan(ctx, prPath(project, repo, prID)+"/commits", func(v map[string]any) bool {
		id, _ := v["id"].(string)
		onPR = id == commit
		return onPR
	})
	return onPR && err == nil, err
}

// firstMerge returns the id of a merge commit reachable from until but not
// from since, or "" when there is none. One commit is enough, so one page of
// one.
func (c *Client) firstMerge(ctx context.Context, project, repo, since, until string) (string, error) {
	var p page
	query := url.Values{"since": {since}, "until": {until}, "merges": {"only"}, "limit": {"1"}}
	if err := c.do(ctx, http.MethodGet, targetPath(project, repo)+"/commits", query, nil, &p); err != nil {
		return "", err
	}
	if len(p.Values) == 0 {
		return "", nil
	}
	id, _ := p.Values[0]["id"].(string)
	if id == "" {
		id = "?"
	}
	return id, nil
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
	cs, err := c.FetchPRChangeSet(ctx, project, repo, prID)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(cs.Changes))
	for _, ch := range cs.Changes {
		paths = append(paths, ch.Path)
	}
	return paths, nil
}

// ChangeSet is a PR's changed files with the two commits its diff is taken
// between: FromHash is the PR's base (the merge base, not the target's tip),
// ToHash its head. Either is "" when Bitbucket leaves it out.
type ChangeSet struct {
	FromHash, ToHash string
	Changes          []Change
}

// Change is one changed file. Type is Bitbucket's (ADD, DELETE, MODIFY,
// MOVE, COPY); SrcPath the old path of a move or copy; ContentID and
// FromContentID the git blob ids of the two versions, "" when absent.
type Change struct {
	Path, SrcPath, Type, ContentID, FromContentID string
}

// FetchPRChangeSet is the PR's changed files with what a screenshot
// comparison needs: the commits on either side and each file's blob ids, so
// an unchanged pair is known without reading it. FetchPRChanges is its path
// list.
//
// Decoding is tolerant per entry, strict over the whole: an entry without
// `path.toString` is skipped, but entries that ALL fail mean the response
// shape is not what this assumes, and that is an error. An empty set with no
// error would read as "nothing changed" and hide the breakage.
func (c *Client) FetchPRChangeSet(ctx context.Context, project, repo string, prID int) (ChangeSet, error) {
	var cs ChangeSet
	entries := 0
	err := c.pages(ctx, prPath(project, repo, prID)+"/changes", func(p page) bool {
		if cs.FromHash == "" {
			cs.FromHash, cs.ToHash = p.FromHash, p.ToHash
		}
		for _, v := range p.Values {
			entries++
			// Directories carry nodeType DIRECTORY; only files have diffs.
			if nodeType, ok := v["nodeType"].(string); ok && !strings.EqualFold(nodeType, "FILE") {
				continue
			}
			ch := Change{Path: pathString(v["path"]), SrcPath: pathString(v["srcPath"])}
			ch.Type, _ = v["type"].(string)
			ch.ContentID, _ = v["contentId"].(string)
			ch.FromContentID, _ = v["fromContentId"].(string)
			if ch.Path != "" {
				cs.Changes = append(cs.Changes, ch)
			}
		}
		return false
	})
	if err != nil {
		return ChangeSet{}, err
	}
	if entries > 0 && len(cs.Changes) == 0 {
		return ChangeSet{}, fmt.Errorf("changes: %d entries, none carried path.toString", entries)
	}
	return cs, nil
}

// pathString is a change's path object as one string, "" when absent.
func pathString(v any) string {
	m, _ := v.(map[string]any)
	s, _ := m["toString"].(string)
	return s
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

// FetchFileBytes is FetchFileContent for a binary file: the bytes as served,
// under the same cap.
func (c *Client) FetchFileBytes(ctx context.Context, project, repo, commit, path string) ([]byte, error) {
	raw := fmt.Sprintf("%s/projects/%s/repos/%s/raw/%s", apiBase, project, repo, path)
	return c.getCapped(ctx, raw, url.Values{"at": {commit}}, "", path, c.maxFileBytes)
}

// PostInlineComment anchors a finished comment body to one line of one file.
//
// body is rendered by the caller: this client decides only where the comment
// goes. Bitbucket rejects an ADDED anchor on a line the diff shows as context,
// and there is no way to tell which from the diff alone, so a 400 is retried
// once as CONTEXT. Only a 400 retries; anything else fails on the first try.
func (c *Client) PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error) {
	path := prPath(project, repo, prID) + "/comments"

	var lastErr error
	for _, lineType := range [...]string{"ADDED", "CONTEXT"} {
		payload := map[string]any{
			"text": body,
			"anchor": map[string]any{
				"path":     file,
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

// ReplyToComment answers one comment in its thread and returns the new
// comment's id, so a later reply under it can be told apart from any other.
func (c *Client) ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) (int, error) {
	var created struct {
		ID int `json:"id"`
	}
	payload := map[string]any{"text": text, "parent": map[string]any{"id": parentCommentID}}
	if err := c.do(ctx, http.MethodPost, prPath(project, repo, prID)+"/comments", nil, payload, &created); err != nil {
		return 0, err
	}
	c.log.InfoContext(ctx, fmt.Sprintf("Replied to comment %d on PR %d", parentCommentID, prID))
	return created.ID, nil
}

// Comment is one PR comment as Bitbucket returns it.
type Comment struct {
	ID      int    `json:"id"`
	Version int    `json:"version"`
	Text    string `json:"text"`
	Author  struct {
		Name        string `json:"name"`
		DisplayName string `json:"displayName"`
	} `json:"author"`
	// Anchor is set on a file comment, nil on a general one. Bitbucket puts
	// it on the thread's root only, not on its replies.
	Anchor *CommentAnchor `json:"anchor"`
	// ThreadResolved is set on a thread's root once someone resolved it.
	ThreadResolved bool `json:"threadResolved"`
}

// CommentAnchor is where a file comment sits. Line is 0 on a comment on the
// whole file. FileType "FROM" puts Line in the old file (a removed line, or
// the left side of a side-by-side view), "TO" in the new one.
type CommentAnchor struct {
	Path     string `json:"path"`
	Line     int    `json:"line"`
	FileType string `json:"fileType"`
	// Orphaned: the anchored line is gone from the PR's current diff, so
	// Line points at code that has since changed.
	Orphaned bool `json:"orphaned"`
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

// ResolveCommentThread resolves the thread a root comment starts, as the
// UI's Resolve does: PUT the comment with its version and threadResolved.
// A reply's id is refused (400), so it takes the root's.
//
// A resolved thread takes no replies: a caller with something to say posts
// it first. The version is read fresh, and a 409 (a reply or edit landed in
// between) is retried once with the version read again.
func (c *Client) ResolveCommentThread(ctx context.Context, project, repo string, prID, commentID int) error {
	path := fmt.Sprintf("%s/comments/%d", prPath(project, repo, prID), commentID)
	var err error
	for range 2 {
		var cur *Comment
		if cur, err = c.FetchPRComment(ctx, project, repo, prID, commentID); err != nil {
			return err
		}
		err = c.do(ctx, http.MethodPut, path, nil,
			map[string]any{"version": cur.Version, "threadResolved": true}, nil)
		if Status(err) != http.StatusConflict {
			break
		}
	}
	if err != nil {
		return err
	}
	c.log.InfoContext(ctx, fmt.Sprintf("Resolved comment thread %d on PR %d", commentID, prID))
	return nil
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
	// FromHash and ToHash: a PR changes page only.
	FromHash string `json:"fromHash"`
	ToHash   string `json:"toHash"`
}

// paged walks every page of a collection.
func (c *Client) paged(ctx context.Context, path string) ([]map[string]any, error) {
	var out []map[string]any
	err := c.scan(ctx, path, func(v map[string]any) bool {
		out = append(out, v)
		return false
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// scan walks a collection page by page until found returns true, fetching no
// page past that one. It stops on isLastPage (a response without the field
// counts as the last), and also when nextPageStart fails to advance, so a
// server echoing an offset cannot spin this forever.
func (c *Client) scan(ctx context.Context, path string, found func(map[string]any) bool) error {
	return c.pages(ctx, path, func(p page) bool {
		for _, v := range p.Values {
			if found(v) {
				return true
			}
		}
		return false
	})
}

// pages is scan a page at a time, for a caller that reads the page's own
// fields too (a PR's changes carry its base and head there). done stops it.
func (c *Client) pages(ctx context.Context, path string, done func(page) bool) error {
	start := 0
	for {
		var p page
		query := url.Values{"start": {strconv.Itoa(start)}, "limit": {strconv.Itoa(pageLimit)}}
		if err := c.do(ctx, http.MethodGet, path, query, nil, &p); err != nil {
			return err
		}
		if done(p) {
			return nil
		}
		if p.IsLastPage == nil || *p.IsLastPage {
			return nil
		}
		if p.NextPageStart == nil || *p.NextPageStart <= start {
			return nil
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

// UserPermissionOn reads username's EFFECTIVE permission on the target: the
// one granted on the target itself, or inherited from the project when the
// target is a repository.
//
// noergler needs WRITE, not read: it posts review comments. Proving the bot
// can GET the repository proves nothing about that, which is why this exists
// beside GetRepo rather than instead of it.
//
// The project fallback is the part that is easy to get wrong. Bitbucket's
// per-repository permission listing returns only what was granted ON THAT
// REPOSITORY. A bot holding PROJECT_WRITE can comment on every repository
// under the project and appears in none of their listings, so checking the
// repository alone reports "cannot write" for a bot that can - and
// grant-bot grants PROJECT_WRITE for a whole-project target, which makes
// that the normal setup rather than a corner.
//
// A permission granted to a GROUP the bot belongs to is still not visible
// here; that needs /permissions/groups and the bot's group membership,
// which the caller's token may not be able to read. A false "cannot write"
// remains possible for a group-granted bot, and shows up as an unhealthy
// status rather than as a silent failure to post.
func (c *Client) UserPermissionOn(ctx context.Context, project, repo, username string) (UserPermission, error) {
	if repo != "" {
		perm, err := c.userPermissionAt(ctx, targetPath(project, repo), username)
		if err != nil || perm != "" {
			return perm, err
		}
		// Nothing on the repository itself: the project may still carry it.
	}
	return c.userPermissionAt(ctx, targetPath(project, ""), username)
}

// userPermissionAt reads one permission listing, following its pages.
//
// `filter` is a SUBSTRING match over users, so the wanted row need not be
// first and need not even be on the first page: a bot sharing a prefix with
// enough other accounts pushes its own row past the limit. Paging until the
// match is found, or until Bitbucket says the page is the last one, is what
// keeps that from reading as "no permission".
func (c *Client) userPermissionAt(ctx context.Context, path, username string) (UserPermission, error) {
	start := 0
	// Bounded: a listing that never sets isLastPage and never advances must
	// not spin the review worker.
	for page := 0; page < 20; page++ {
		var out struct {
			Values []struct {
				User struct {
					Name string `json:"name"`
				} `json:"user"`
				Permission string `json:"permission"`
			} `json:"values"`
			IsLastPage    bool `json:"isLastPage"`
			NextPageStart int  `json:"nextPageStart"`
		}
		query := url.Values{"filter": {username}, "limit": {strconv.Itoa(pageLimit)}}
		if start > 0 {
			query.Set("start", strconv.Itoa(start))
		}
		if err := c.do(ctx, http.MethodGet, path+"/permissions/users", query, nil, &out); err != nil {
			return "", err
		}
		for _, v := range out.Values {
			if strings.EqualFold(v.User.Name, username) {
				return UserPermission(v.Permission), nil
			}
		}
		if out.IsLastPage || out.NextPageStart <= start || len(out.Values) == 0 {
			break
		}
		start = out.NextPageStart
	}
	// No row for this user is not an error: it is the answer, and it means
	// the bot has no permission of its own here.
	return "", nil
}

// GetProject reads a project, to prove it exists and is readable. The body
// is not decoded: the answer is the status alone.
func (c *Client) GetProject(ctx context.Context, project string) error {
	return c.do(ctx, http.MethodGet, apiBase+"/projects/"+project, nil, nil, nil)
}

// GetRepo reads one repository, to prove the same.
func (c *Client) GetRepo(ctx context.Context, project, repo string) error {
	return c.do(ctx, http.MethodGet, apiBase+"/projects/"+project+"/repos/"+repo, nil, nil, nil)
}
