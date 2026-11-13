package bitbucket

import (
	"context"
	"errors"
	"net/http"
	"net/url"

	"github.com/trick77/noergler/internal/gitrepo"
)

// MergeStatus is Bitbucket's verdict on merging a PR now.
type MergeStatus struct {
	CanMerge   bool `json:"canMerge"`
	Conflicted bool `json:"conflicted"`
}

// MergeStatus asks whether the PR merges, and whether a conflict is why
// not. Bitbucket computes it on the request (a test merge), so it is read
// once per request, never polled.
func (c *Client) MergeStatus(ctx context.Context, project, repo string, prID int) (MergeStatus, error) {
	var out MergeStatus
	err := c.do(ctx, http.MethodGet, prPath(project, repo, prID)+"/merge", nil, nil, &out)
	return out, err
}

// ErrNoHTTPClone is a repository whose links name no HTTP clone URL.
var ErrNoHTTPClone = errors.New("no HTTP clone URL")

// GitRemote is the repository for git over HTTP as this client's account:
// the clone URL Bitbucket names (never one built here: a context path or a
// proxy rewrites it), the account's token, and the adapter's git HTTP
// client: its rules (no redirects, no client timeout) without the header
// timeout, which upload-pack's pack computation would trip. maxBytes
// bounds a fetch.
func (c *Client) GitRemote(ctx context.Context, project, repo string, maxBytes int64) (gitrepo.Remote, error) {
	var r struct {
		Links struct {
			Clone []struct {
				Href string `json:"href"`
				Name string `json:"name"`
			} `json:"clone"`
		} `json:"links"`
	}
	if err := c.do(ctx, http.MethodGet, apiBase+"/projects/"+project+"/repos/"+repo, nil, nil, &r); err != nil {
		return gitrepo.Remote{}, err
	}
	for _, l := range r.Links.Clone {
		if l.Name != "http" && l.Name != "https" {
			continue
		}
		u, err := url.Parse(l.Href)
		if err != nil {
			return gitrepo.Remote{}, err
		}
		// The link carries the account's name as userinfo: auth goes
		// in the header, never in the URL.
		u.User = nil
		return gitrepo.Remote{URL: u.String(), Username: c.botUsername, Token: c.token, HTTP: c.git, MaxBytes: maxBytes}, nil
	}
	return gitrepo.Remote{}, ErrNoHTTPClone
}

// BotIdentity is the account's display name and email, a commit's author:
// a commit-checker hook compares the author with the pusher.
func (c *Client) BotIdentity(ctx context.Context) (name, email string, err error) {
	var u struct {
		Name         string `json:"name"`
		DisplayName  string `json:"displayName"`
		EmailAddress string `json:"emailAddress"`
	}
	// c.url escapes the path.
	if err := c.do(ctx, http.MethodGet, apiBase+"/users/"+c.botUsername, nil, nil, &u); err != nil {
		return "", "", err
	}
	name = u.DisplayName
	if name == "" {
		name = c.botUsername
	}
	return name, u.EmailAddress, nil
}
