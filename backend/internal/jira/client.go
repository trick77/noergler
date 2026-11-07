// Package jira reads tickets for the acceptance-criteria check: a ticket's
// summary, description and criteria go into the review prompt as context.
//
// Jira never stops a review or a boot. A missing ticket is noise: the reference
// in a branch name may be a false positive or in a project the service account
// cannot see. Jira down is an error the review turns into a footer notice; at
// startup it is a warning.
package jira

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/forward"
	"github.com/trick77/noergler/internal/httpstats"
)

// maxDescriptionLength caps a description before markup is stripped. Long
// descriptions are mostly changelogs and templates, and the prompt has a budget.
const maxDescriptionLength = 5000

// requestTimeout bounds one Jira call. Responses are small, so unlike the
// Bitbucket client a total timeout is the right shape here.
const requestTimeout = 30 * time.Second

// errBodyLimit caps how much of an error body is kept for a message.
const errBodyLimit = 2000

// fieldsQuery is the exact field list the service asks for. Sent verbatim:
// url.Values would escape the commas.
const fieldsQuery = "fields=summary,description,labels,subtasks,issuetype,status,parent"

// StatusError is a non-2xx response from Jira.
type StatusError struct {
	Method string
	Path   string
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("%s %s: HTTP %d", e.Method, e.Path, e.Status)
	}
	return fmt.Sprintf("%s %s: HTTP %d: %s", e.Method, e.Path, e.Status, e.Body)
}

// Status returns the HTTP status of err, or 0 if err is not a StatusError.
func Status(err error) int {
	var se *StatusError
	if errors.As(err, &se) {
		return se.Status
	}
	return 0
}

// Ticket is one Jira issue, as much of it as the review prompt needs.
type Ticket struct {
	Key                string
	Title              string
	Description        string
	Labels             []string
	AcceptanceCriteria string
	Subtasks           []string
	URL                string
	IssueType          string
	Status             string
	ParentKey          string
}

// Client reads issues as one Jira user.
type Client struct {
	base     string
	token    string
	prefixes []string
	http     *http.Client
	log      *slog.Logger
	// observe, when set, sees the outcome of every request: the status, or
	// 0 with the transport error. A team's own token uses it to track
	// whether it still works.
	observe func(status int, err error)
}

// WithToken returns a copy reading as another Jira user, for a team's own
// token. The transport (and its pool) is shared; the observer is not.
func (c *Client) WithToken(token string) *Client {
	clone := *c
	clone.token = token
	clone.observe = nil
	return &clone
}

// Observed returns a copy that reports every request's outcome to observe.
func (c *Client) Observed(observe func(status int, err error)) *Client {
	clone := *c
	clone.observe = observe
	return &clone
}

// New builds a client for the shared read-only Jira user.
func New(cfg config.Jira, log *slog.Logger) (*Client, error) {
	base := strings.TrimSuffix(cfg.URL, "/")
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("parse JIRA_URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("parse JIRA_URL: %q is not an absolute URL", cfg.URL)
	}
	return &Client{
		base:     base,
		token:    cfg.Token,
		prefixes: cfg.AcceptanceCriteriaPrefixes,
		http: &http.Client{
			Timeout: requestTimeout,
			Transport: httpstats.Transport("jira", &http.Transport{
				Proxy:               http.ProxyFromEnvironment,
				DialContext:         (&net.Dialer{Timeout: 30 * time.Second}).DialContext,
				TLSHandshakeTimeout: 30 * time.Second,
				ForceAttemptHTTP2:   true,
			}),
			// Redirects are not followed: a 3xx would replay the bearer token
			// at the new host. Non-2xx, so a redirect cannot read as success.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		log: log,
	}, nil
}

// get issues a GET and decodes JSON into out.
func (c *Client) get(ctx context.Context, rawURL, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(req)
	if c.observe != nil {
		if err != nil {
			c.observe(0, err)
		} else {
			c.observe(resp.StatusCode, nil)
		}
	}
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &StatusError{
			Method: http.MethodGet,
			Path:   path,
			Status: resp.StatusCode,
			Body:   forward.ReadErrorBody(resp.Body, errBodyLimit),
		}
	}
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// CheckConnectivity proves the token works. It runs at startup; a failure
// there is a warning, never an abort.
func (c *Client) CheckConnectivity(ctx context.Context) error {
	var me struct {
		Name        string `json:"name"`
		DisplayName string `json:"displayName"`
	}
	path := "/rest/api/2/myself"
	if err := c.get(ctx, c.base+path, path, &me); err != nil {
		return err
	}
	who := me.DisplayName
	if who == "" {
		who = me.Name
	}
	if who == "" {
		who = "?"
	}
	c.log.Info("Jira authenticated as " + who)
	return nil
}

// issue is the wire shape of one issue.
type issue struct {
	Key    string `json:"key"`
	Fields struct {
		Summary     string   `json:"summary"`
		Description *string  `json:"description"`
		Labels      []string `json:"labels"`
		Subtasks    []struct {
			Key    string `json:"key"`
			Fields struct {
				Summary string `json:"summary"`
			} `json:"fields"`
		} `json:"subtasks"`
		IssueType *struct {
			Name string `json:"name"`
		} `json:"issuetype"`
		Status *struct {
			Name string `json:"name"`
		} `json:"status"`
		Parent *struct {
			Key string `json:"key"`
		} `json:"parent"`
	} `json:"fields"`
}

// FetchTicket reads one ticket, or returns nil when there is none to read.
//
// A missing ticket (404) or one the user may not see (403) returns (nil, nil):
// the ticket reference came out of a branch name and may well be noise. Any
// other failure (refused, DNS, timeout, another status, bad JSON) is Jira not
// serving the call and comes back as an error, so the summary can say the
// check was skipped instead of "no ticket".
func (c *Client) FetchTicket(ctx context.Context, ticketID string) (*Ticket, error) {
	path := "/rest/api/2/issue/" + ticketID
	rawURL := c.base + path + "?" + fieldsQuery

	var got issue
	err := c.get(ctx, rawURL, path, &got)
	switch {
	case err == nil:
	case Status(err) == http.StatusNotFound || Status(err) == http.StatusForbidden:
		c.log.InfoContext(ctx, fmt.Sprintf("Jira ticket %s not readable (HTTP %d)", ticketID, Status(err)))
		return nil, nil
	default:
		return nil, err
	}

	key := got.Key
	if key == "" {
		key = ticketID
	}

	// Truncate the raw markup first, then strip: the cap is about how much text
	// is carried, and stripping afterwards only ever shortens it.
	description := ""
	if got.Fields.Description != nil {
		description = *got.Fields.Description
	}
	if len(description) > maxDescriptionLength {
		description = description[:maxDescriptionLength] + "..."
	}
	if description != "" {
		description = stripMarkup(description)
	}

	var subtasks []string
	for _, st := range got.Fields.Subtasks {
		if st.Key == "" {
			continue
		}
		subtasks = append(subtasks, st.Key+": "+st.Fields.Summary)
	}

	t := &Ticket{
		Key:                key,
		Title:              got.Fields.Summary,
		Description:        description,
		Labels:             got.Fields.Labels,
		AcceptanceCriteria: acceptanceCriteria(description, c.prefixes),
		Subtasks:           subtasks,
		URL:                c.base + "/browse/" + key,
	}
	if got.Fields.IssueType != nil {
		t.IssueType = got.Fields.IssueType.Name
	}
	if got.Fields.Status != nil {
		t.Status = got.Fields.Status.Name
	}
	if got.Fields.Parent != nil {
		t.ParentKey = got.Fields.Parent.Key
	}
	return t, nil
}

// FetchTicketWithParent reads a ticket and, for a subtask, its parent. The
// acceptance criteria often live on the parent story, not the subtask being
// worked on. A missing parent (404/403) is no error: the child still counts.
// A parent Jira failed to serve comes back as the error BESIDE the good
// child: the caller keeps the child and still tells the author Jira failed,
// since the criteria on the parent were never checked.
func (c *Client) FetchTicketWithParent(ctx context.Context, ticketID string) (ticket, parent *Ticket, err error) {
	ticket, err = c.FetchTicket(ctx, ticketID)
	if err != nil || ticket == nil || ticket.ParentKey == "" {
		return ticket, nil, err
	}
	parent, err = c.FetchTicket(ctx, ticket.ParentKey)
	if err != nil {
		return ticket, nil, fmt.Errorf("parent %s: %w", ticket.ParentKey, err)
	}
	return ticket, parent, nil
}
